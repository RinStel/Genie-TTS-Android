#include <jni.h>
#include <android/log.h>
#include <vector>
#include <cstring>
#include "onnxruntime_c_api.h"
#include "native_semantic_slicing.h"

#define TAG "NativeDecoderLoop"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static const OrtApi* g_ort = nullptr;

static const OrtApi* getOrtApi() {
    if (g_ort) return g_ort;
    const OrtApiBase* base = OrtGetApiBase();
    if (!base) { LOGE("OrtGetApiBase() returned null"); return nullptr; }
    g_ort = base->GetApi(ORT_API_VERSION);
    if (!g_ort) { LOGE("GetApi(%d) returned null", ORT_API_VERSION); return nullptr; }
    return g_ort;
}

static const char* jstrToCstr(JNIEnv* env, jstring jstr) {
    return env->GetStringUTFChars(jstr, nullptr);
}

static void releaseJstr(JNIEnv* env, jstring jstr, const char* cstr) {
    env->ReleaseStringUTFChars(jstr, cstr);
}

static const OrtMemoryInfo* tryGetQnnMemoryInfo(const OrtApi* api, OrtEnv* env) {
    // Use the memory descriptor exported by the registered EP device. A
    // hard-coded CreateMemoryInfo("QNN", ...) does not prove that this ORT
    // build has a usable QNN allocator and caused false fallback reports.
    if (!env || !api->GetEpDevices || !api->EpDevice_EpName || !api->EpDevice_MemoryInfo) {
        return nullptr;
    }

    const OrtEpDevice* const* devices = nullptr;
    size_t deviceCount = 0;
    OrtStatus* status = api->GetEpDevices(env, &devices, &deviceCount);
    if (status) {
        LOGE("GetEpDevices failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return nullptr;
    }

    for (size_t i = 0; i < deviceCount; ++i) {
        const char* epName = api->EpDevice_EpName(devices[i]);
        if (!epName || (std::strcmp(epName, "QNN") != 0 &&
                        std::strcmp(epName, "QNNExecutionProvider") != 0)) {
            continue;
        }
        const OrtMemoryInfo* memoryInfo = api->EpDevice_MemoryInfo(
            devices[i], OrtDeviceMemoryType_DEFAULT);
        if (memoryInfo) {
            LOGI("Using QNN EP device memory info from registered device");
            return memoryInfo;
        }
        LOGI("QNN EP device exposes CPU memory only");
    }
    return nullptr;
}

static bool bindOutputToDevice(
    const OrtApi* api,
    OrtIoBinding* binding,
    const char* name,
    const OrtMemoryInfo* memoryInfo
) {
    OrtStatus* status = api->BindOutputToDevice(binding, name, memoryInfo);
    if (!status) return true;
    LOGE("BindOutputToDevice(%s) failed: %s", name, api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return false;
}

static jlongArray extractTokensFromOutput(JNIEnv* env, const OrtApi* api, OrtValue* yValue, int generatedSteps) {
    if (!yValue || generatedSteps <= 0) return nullptr;

    void* yData = nullptr;
    OrtStatus* status = api->GetTensorMutableData(yValue, &yData);
    if (status) {
        LOGE("GetTensorMutableData(y) failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return nullptr;
    }

    OrtTensorTypeAndShapeInfo* shapeInfo = nullptr;
    status = api->GetTensorTypeAndShape(yValue, &shapeInfo);
    if (status) {
        LOGE("GetTensorTypeAndShape failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return nullptr;
    }

    size_t numElements = 0;
    status = api->GetTensorShapeElementCount(shapeInfo, &numElements);
    api->ReleaseTensorTypeAndShapeInfo(shapeInfo);
    if (status) {
        LOGE("GetTensorShapeElementCount failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return nullptr;
    }

    int64_t* yTokens = reinterpret_cast<int64_t*>(yData);
    // Clamp to the actual tensor length so early/failed stops do not underflow.
    size_t tokenCount = (size_t)generatedSteps < numElements ? (size_t)generatedSteps : numElements;
    size_t startIdx = numElements - tokenCount;

    std::vector<int64_t> tokens(yTokens + startIdx, yTokens + startIdx + tokenCount);
    const std::vector<int64_t> semanticTokens =
        SliceSemanticTokens(tokens, generatedSteps, 0);
    jlongArray result = env->NewLongArray((jsize)semanticTokens.size());
    if (!result || semanticTokens.empty()) return result;
    env->SetLongArrayRegion(result, 0, (jsize)semanticTokens.size(),
        reinterpret_cast<const jlong*>(semanticTokens.data()));
    return result;
}

static void notifyDecoderProgress(
    JNIEnv* env,
    jobject progressCallback,
    jmethodID progressMethod,
    int generatedSteps,
    int maxSteps
) {
    if (!progressCallback || !progressMethod) return;
    env->CallVoidMethod(progressCallback, progressMethod, (jint)generatedSteps, (jint)maxSteps);
    if (env->ExceptionCheck()) {
        LOGE("Decoder progress callback threw; suppressing progress callback failure");
        env->ExceptionClear();
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_dev_rinstel_genie_1tts_inference_NativeDecoderLoop_nativeDecoderLoop(
    JNIEnv* env, jclass cls,
    jlong environmentPtr,
    jlong sessionPtr,
    jobjectArray inputNamesJ,
    jobjectArray outputNamesJ,
    jlongArray yDataJ, jlongArray yShapeJ,
    jfloatArray yEmbDataJ, jlongArray yEmbShapeJ,
    jfloatArray kvDataJ, jlongArray kvShapeJ,
    jint numKvTensors,
    jintArray outputMappingJ,
    jboolean useQnnIoBindingJ,
    jint maxSteps,
    jobject progressCallback
) {
    (void)cls;

    const OrtApi* api = getOrtApi();
    if (!api || !sessionPtr) {
        LOGE("ORT API or session is null");
        return nullptr;
    }

    OrtEnv* ortEnvironment = reinterpret_cast<OrtEnv*>(environmentPtr);
    OrtSession* session = reinterpret_cast<OrtSession*>(sessionPtr);
    int numInputs = env->GetArrayLength(inputNamesJ);
    int numOutputs = env->GetArrayLength(outputNamesJ);
    jmethodID progressMethod = nullptr;
    if (progressCallback) {
        jclass callbackClass = env->GetObjectClass(progressCallback);
        progressMethod = env->GetMethodID(callbackClass, "onProgress", "(II)V");
        env->DeleteLocalRef(callbackClass);
        if (!progressMethod && env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }

    // --- Convert input/output names to C strings ---
    std::vector<const char*> inputNames(numInputs);
    std::vector<const char*> outputNames(numOutputs);
    std::vector<jstring> inputJStrs(numInputs);
    std::vector<jstring> outputJStrs(numOutputs);

    for (int i = 0; i < numInputs; i++) {
        inputJStrs[i] = (jstring) env->GetObjectArrayElement(inputNamesJ, i);
        inputNames[i] = jstrToCstr(env, inputJStrs[i]);
    }
    for (int i = 0; i < numOutputs; i++) {
        outputJStrs[i] = (jstring) env->GetObjectArrayElement(outputNamesJ, i);
        outputNames[i] = jstrToCstr(env, outputJStrs[i]);
    }

    // --- Get output mapping ---
    jint* mapping = env->GetIntArrayElements(outputMappingJ, nullptr);

    // Keep all acquired JNI/ORT resources in one cleanup path. Inference errors
    // are expected on device while probing provider capabilities, so an error
    // must not leave pinned Java arrays or native tensors behind.
    jlong* yShapeArr = nullptr;
    jlong* yDataRaw = nullptr;
    jlong* yEmbShapeArr = nullptr;
    jfloat* yEmbDataRaw = nullptr;
    jlong* kvShapeArr = nullptr;
    jfloat* kvDataRaw = nullptr;
    OrtMemoryInfo* cpuMemInfo = nullptr;
    OrtRunOptions* runOptions = nullptr;
    OrtValue* yValue = nullptr;
    OrtValue* yEmbValue = nullptr;
    std::vector<OrtValue*> kvValues(numKvTensors, nullptr);

    auto releaseJavaResources = [&]() {
        if (yShapeArr) {
            env->ReleaseLongArrayElements(yShapeJ, yShapeArr, JNI_ABORT);
            yShapeArr = nullptr;
        }
        if (yDataRaw) {
            env->ReleaseLongArrayElements(yDataJ, yDataRaw, JNI_ABORT);
            yDataRaw = nullptr;
        }
        if (yEmbShapeArr) {
            env->ReleaseLongArrayElements(yEmbShapeJ, yEmbShapeArr, JNI_ABORT);
            yEmbShapeArr = nullptr;
        }
        if (yEmbDataRaw) {
            env->ReleaseFloatArrayElements(yEmbDataJ, yEmbDataRaw, JNI_ABORT);
            yEmbDataRaw = nullptr;
        }
        if (kvShapeArr) {
            env->ReleaseLongArrayElements(kvShapeJ, kvShapeArr, JNI_ABORT);
            kvShapeArr = nullptr;
        }
        if (kvDataRaw) {
            env->ReleaseFloatArrayElements(kvDataJ, kvDataRaw, JNI_ABORT);
            kvDataRaw = nullptr;
        }
        if (mapping) {
            env->ReleaseIntArrayElements(outputMappingJ, mapping, JNI_ABORT);
            mapping = nullptr;
        }
        for (int i = 0; i < numInputs; ++i) {
            if (inputJStrs[i]) {
                if (inputNames[i]) releaseJstr(env, inputJStrs[i], inputNames[i]);
                env->DeleteLocalRef(inputJStrs[i]);
                inputJStrs[i] = nullptr;
                inputNames[i] = nullptr;
            }
        }
        for (int i = 0; i < numOutputs; ++i) {
            if (outputJStrs[i]) {
                if (outputNames[i]) releaseJstr(env, outputJStrs[i], outputNames[i]);
                env->DeleteLocalRef(outputJStrs[i]);
                outputJStrs[i] = nullptr;
                outputNames[i] = nullptr;
            }
        }
    };

    auto releaseOrtResources = [&]() {
        if (yValue) {
            api->ReleaseValue(yValue);
            yValue = nullptr;
        }
        if (yEmbValue) {
            api->ReleaseValue(yEmbValue);
            yEmbValue = nullptr;
        }
        for (OrtValue*& value : kvValues) {
            if (value) {
                api->ReleaseValue(value);
                value = nullptr;
            }
        }
        if (runOptions) {
            api->ReleaseRunOptions(runOptions);
            runOptions = nullptr;
        }
        if (cpuMemInfo) {
            api->ReleaseMemoryInfo(cpuMemInfo);
            cpuMemInfo = nullptr;
        }
    };

    auto fail = [&]() -> jlongArray {
        releaseOrtResources();
        releaseJavaResources();
        return nullptr;
    };

    // --- Create memory info ---
    OrtStatus* status = api->CreateCpuMemoryInfo(
        OrtDeviceAllocator, OrtMemTypeDefault, &cpuMemInfo);
    if (status) {
        LOGE("CreateCpuMemoryInfo failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return fail();
    }

    // Try to create QNN device memory info for IO binding.
    const OrtMemoryInfo* qnnMemInfo = nullptr;
    bool useIoBinding = false;
    if (useQnnIoBindingJ == JNI_TRUE && ortEnvironment) {
        qnnMemInfo = tryGetQnnMemoryInfo(api, ortEnvironment);
        useIoBinding = (qnnMemInfo != nullptr);
    }
    if (useQnnIoBindingJ == JNI_TRUE && useIoBinding) {
        LOGI("QNN device memory available - using IO Binding");
    } else if (useQnnIoBindingJ == JNI_TRUE) {
        LOGI("QNN device memory unavailable - using array-based Run");
    }

    // --- Create run options ---
    status = api->CreateRunOptions(&runOptions);
    if (status) {
        LOGE("CreateRunOptions failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return fail();
    }

    // --- Create initial state tensors ---
    int yShapeLen = env->GetArrayLength(yShapeJ);
    yShapeArr = env->GetLongArrayElements(yShapeJ, nullptr);
    int yDataLen = env->GetArrayLength(yDataJ);
    yDataRaw = env->GetLongArrayElements(yDataJ, nullptr);
    if (!yShapeArr || !yDataRaw) {
        LOGE("Failed to access initial y arrays");
        return fail();
    }

    status = api->CreateTensorWithDataAsOrtValue(
        cpuMemInfo, yDataRaw, yDataLen * sizeof(int64_t),
        yShapeArr, yShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &yValue);
    if (status) {
        LOGE("CreateTensor y failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return fail();
    }

    int yEmbShapeLen = env->GetArrayLength(yEmbShapeJ);
    yEmbShapeArr = env->GetLongArrayElements(yEmbShapeJ, nullptr);
    int yEmbDataLen = env->GetArrayLength(yEmbDataJ);
    yEmbDataRaw = env->GetFloatArrayElements(yEmbDataJ, nullptr);
    if (!yEmbShapeArr || !yEmbDataRaw) {
        LOGE("Failed to access initial y_emb arrays");
        return fail();
    }

    status = api->CreateTensorWithDataAsOrtValue(
        cpuMemInfo, yEmbDataRaw, yEmbDataLen * sizeof(float),
        yEmbShapeArr, yEmbShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &yEmbValue);
    if (status) {
        LOGE("CreateTensor y_emb failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return fail();
    }

    int kvShapeLen = env->GetArrayLength(kvShapeJ);
    kvShapeArr = env->GetLongArrayElements(kvShapeJ, nullptr);
    kvDataRaw = env->GetFloatArrayElements(kvDataJ, nullptr);
    if (!kvShapeArr || !kvDataRaw) {
        LOGE("Failed to access initial KV arrays");
        return fail();
    }

    size_t kvPerTensor = 1;
    for (int i = 0; i < kvShapeLen; i++) {
        kvPerTensor *= (size_t)kvShapeArr[i];
    }
    size_t kvBytesPerTensor = kvPerTensor * sizeof(float);

    for (int i = 0; i < numKvTensors; i++) {
        void* dataPtr = kvDataRaw + (i * kvPerTensor);
        status = api->CreateTensorWithDataAsOrtValue(
            cpuMemInfo, dataPtr, kvBytesPerTensor,
            kvShapeArr, kvShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &kvValues[i]);
        if (status) {
            LOGE("CreateTensor kv[%d] failed: %s", i, api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            return fail();
        }
    }

    // --- Build initial input array ---
    std::vector<OrtValue*> inputs(numInputs);
    inputs[0] = yValue;
    inputs[1] = yEmbValue;
    for (int i = 0; i < numKvTensors; i++) {
        inputs[2 + i] = kvValues[i];
    }

    int generatedSteps = 0;
    bool stopped = false;
    OrtAllocator* defaultAlloc = nullptr;
    if (useIoBinding) {
        status = api->GetAllocatorWithDefaultOptions(&defaultAlloc);
        if (status || !defaultAlloc) {
            LOGE("GetAllocatorWithDefaultOptions failed: %s",
                 status ? api->GetErrorMessage(status) : "allocator is null");
            if (status) api->ReleaseStatus(status);
            useIoBinding = false;
        }
    }

    // =====================================================================
    // IO BINDING PATH - KV cache stays on QNN device between steps
    // =====================================================================
    if (useIoBinding) {
        OrtIoBinding* binding = nullptr;
        status = api->CreateIoBinding(session, &binding);
        if (status) {
            LOGE("CreateIoBinding failed: %s", api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            useIoBinding = false;
        } else {
            // Bind outputs to QNN device memory except stop_condition, which is read on CPU.
            bool initialOutputsBound = true;
            for (int i = 0; i < numOutputs; i++) {
                const OrtMemoryInfo* memoryInfo = i == 2 ? cpuMemInfo : qnnMemInfo;
                if (!bindOutputToDevice(api, binding, outputNames[i], memoryInfo)) {
                    initialOutputsBound = false;
                    break;
                }
            }

            if (initialOutputsBound) {
                std::vector<OrtValue*> prevOutputs(numOutputs, nullptr);
                bool bindingFailed = false;

                for (int step = 0; step < maxSteps && !stopped; step++) {
                    // Rebind inputs every step because they point to the previous iteration outputs.
                    api->ClearBoundInputs(binding);
                    bool inputsBound = true;
                    for (int i = 0; i < numInputs; i++) {
                        status = api->BindInput(binding, inputNames[i], inputs[i]);
                        if (status) {
                            LOGE("BindInput[%d] failed: %s", i, api->GetErrorMessage(status));
                            api->ReleaseStatus(status);
                            inputsBound = false;
                            break;
                        }
                    }
                    if (!inputsBound) {
                        bindingFailed = true;
                        break;
                    }

                    // Run with binding so QNN can keep large KV tensors resident when supported.
                    status = api->RunWithBinding(session, runOptions, binding);
                    if (status) {
                        LOGE("RunWithBinding failed at step %d: %s", step, api->GetErrorMessage(status));
                        api->ReleaseStatus(status);
                        bindingFailed = true;
                        break;
                    }

                    // Ensure bound outputs are materialized before reading stop_condition.
                    status = api->SynchronizeBoundOutputs(binding);
                    if (status) {
                        LOGE("SynchronizeBoundOutputs failed: %s", api->GetErrorMessage(status));
                        api->ReleaseStatus(status);
                        bindingFailed = true;
                        break;
                    }

                    generatedSteps++;
                    notifyDecoderProgress(env, progressCallback, progressMethod, generatedSteps, maxSteps);

                    // ORT allocates the pointer array; OrtValue ownership moves into prevOutputs below.
                    OrtValue** outputValues = nullptr;
                    size_t numOut = 0;
                    status = api->GetBoundOutputValues(binding, defaultAlloc, &outputValues, &numOut);
                    if (status || numOut != (size_t)numOutputs) {
                        LOGE("GetBoundOutputValues failed: %s",
                             status ? api->GetErrorMessage(status) : "output count mismatch");
                        if (status) api->ReleaseStatus(status);
                        if (outputValues) {
                            for (size_t i = 0; i < numOut; ++i) {
                                if (outputValues[i]) api->ReleaseValue(outputValues[i]);
                            }
                            defaultAlloc->Free(defaultAlloc, outputValues);
                        }
                        bindingFailed = true;
                        break;
                    }

                    // stop_condition is bound on CPU so it can be checked without copying KV tensors back.
                    void* stopData = nullptr;
                    status = api->GetTensorMutableData(outputValues[2], &stopData);
                    if (status) {
                        LOGE("GetTensorMutableData(stop) failed: %s", api->GetErrorMessage(status));
                        api->ReleaseStatus(status);
                        for (size_t i = 0; i < numOut; i++) {
                            if (outputValues[i]) api->ReleaseValue(outputValues[i]);
                        }
                        defaultAlloc->Free(defaultAlloc, outputValues);
                        bindingFailed = true;
                        break;
                    }
                    bool stop = *reinterpret_cast<bool*>(stopData);

                    // Release outputs from the previous iteration after the current run no longer references them.
                    if (step > 0) {
                        for (int i = 0; i < numOutputs; i++) {
                            if (prevOutputs[i]) {
                                api->ReleaseValue(prevOutputs[i]);
                                prevOutputs[i] = nullptr;
                            }
                        }
                    }

                    // Move current outputs to prevOutputs and refresh output device bindings.
                    for (int i = 0; i < numOutputs; i++) {
                        prevOutputs[i] = outputValues[i];
                    }
                    bool outputsBound = true;
                    for (int i = 0; i < numOutputs; i++) {
                        const OrtMemoryInfo* memoryInfo = i == 2 ? cpuMemInfo : qnnMemInfo;
                        if (!bindOutputToDevice(api, binding, outputNames[i], memoryInfo)) {
                            outputsBound = false;
                            break;
                        }
                    }
                    defaultAlloc->Free(defaultAlloc, outputValues);
                    if (!outputsBound) {
                        bindingFailed = true;
                        break;
                    }

                    // Map model outputs back to the next step inputs; output[2] is stop_condition.
                    for (int i = 0; i < numInputs; i++) {
                        inputs[i] = prevOutputs[mapping[i]];
                    }

                    if (stop) {
                        stopped = true;
                        break;
                    }
                }

                // Never expose a partial sequence after an IO binding failure. The
                // array path is safe only before binding has executed any step.
                jlongArray result = bindingFailed
                    ? nullptr
                    : extractTokensFromOutput(env, api, inputs[0], generatedSteps);

                // Cleanup IO binding resources and tensors created for the initial step.
                for (int i = 0; i < numOutputs; i++) {
                    if (prevOutputs[i]) api->ReleaseValue(prevOutputs[i]);
                }
                api->ClearBoundInputs(binding);
                api->ClearBoundOutputs(binding);
                api->ReleaseIoBinding(binding);

                // Release initial state tensors.
                api->ReleaseValue(yValue);
                api->ReleaseValue(yEmbValue);
                for (int i = 0; i < numKvTensors; i++) api->ReleaseValue(kvValues[i]);
                api->ReleaseRunOptions(runOptions);
                api->ReleaseMemoryInfo(cpuMemInfo);

                // Release Java arrays and strings.
                env->ReleaseLongArrayElements(yShapeJ, yShapeArr, JNI_ABORT);
                env->ReleaseLongArrayElements(yDataJ, yDataRaw, JNI_ABORT);
                env->ReleaseLongArrayElements(yEmbShapeJ, yEmbShapeArr, JNI_ABORT);
                env->ReleaseFloatArrayElements(yEmbDataJ, yEmbDataRaw, JNI_ABORT);
                env->ReleaseLongArrayElements(kvShapeJ, kvShapeArr, JNI_ABORT);
                env->ReleaseFloatArrayElements(kvDataJ, kvDataRaw, JNI_ABORT);
                env->ReleaseIntArrayElements(outputMappingJ, mapping, JNI_ABORT);
                for (int i = 0; i < numInputs; i++) {
                    releaseJstr(env, inputJStrs[i], inputNames[i]);
                    env->DeleteLocalRef(inputJStrs[i]);
                }
                for (int i = 0; i < numOutputs; i++) {
                    releaseJstr(env, outputJStrs[i], outputNames[i]);
                    env->DeleteLocalRef(outputJStrs[i]);
                }

                return result;
            }

            // A provider may advertise a device but reject one of the model
            // output types. Re-enter the CPU-array path instead of returning
            // an empty or partially generated token sequence.
            api->ClearBoundInputs(binding);
            api->ClearBoundOutputs(binding);
            api->ReleaseIoBinding(binding);
            useIoBinding = false;
        }
    }

    // =====================================================================
    // FALLBACK: Array-based Run (no IO binding)
    // =====================================================================
    {
        std::vector<OrtValue*> outputs(numOutputs, nullptr);
        std::vector<OrtValue*> prevOutputs(numOutputs, nullptr);
        bool runFailed = false;

        for (int step = 0; step < maxSteps && !stopped; step++) {
            for (int i = 0; i < numOutputs; i++) outputs[i] = nullptr;

            status = api->Run(
                session, runOptions,
                inputNames.data(), (const OrtValue* const*)inputs.data(), (size_t)numInputs,
                outputNames.data(), (size_t)numOutputs, outputs.data());

            if (status) {
                LOGE("Run failed at step %d: %s", step, api->GetErrorMessage(status));
                api->ReleaseStatus(status);
                runFailed = true;
                break;
            }

            generatedSteps++;
            notifyDecoderProgress(env, progressCallback, progressMethod, generatedSteps, maxSteps);

            // stop_condition is a CPU scalar in the array-based path.
            void* stopData = nullptr;
            if (numOutputs <= 2 || !outputs[2]) {
                LOGE("Array Run returned no stop_condition output at step %d", step);
                runFailed = true;
                break;
            }
            status = api->GetTensorMutableData(outputs[2], &stopData);
            if (status) {
                LOGE("GetTensorMutableData(stop) failed: %s", api->GetErrorMessage(status));
                api->ReleaseStatus(status);
                runFailed = true;
                break;
            }
            bool stop = *reinterpret_cast<bool*>(stopData);
            if (stop) {
                stopped = true;
                break;
            }

            // Release outputs from two iterations ago after inputs no longer reference them.
            if (step > 0) {
                for (int i = 0; i < numOutputs; i++) {
                    if (prevOutputs[i]) {
                        api->ReleaseValue(prevOutputs[i]);
                        prevOutputs[i] = nullptr;
                    }
                }
            }

            // Transfer ownership from outputs to prevOutputs for the next decoder step.
            for (int i = 0; i < numOutputs; i++) {
                prevOutputs[i] = outputs[i];
                // Prevent cleanup from releasing the same OrtValue twice if maxSteps is reached.
                outputs[i] = nullptr;
            }

            // Map model outputs back to the next step inputs.
            for (int i = 0; i < numInputs; i++) {
                inputs[i] = prevOutputs[mapping[i]];
            }
        }

        // If the loop stopped immediately, outputs[0] still owns the final y tensor.
        OrtValue* finalY = outputs[0] ? outputs[0] : prevOutputs[0];
        // Do not turn a provider/runtime error after an earlier successful step
        // into a shorter, apparently valid utterance.
        jlongArray result = runFailed
            ? nullptr
            : extractTokensFromOutput(env, api, finalY, generatedSteps);

        // Cleanup fallback tensors and Java resources.
        for (int i = 0; i < numOutputs; i++) {
            if (outputs[i]) api->ReleaseValue(outputs[i]);
            if (prevOutputs[i]) api->ReleaseValue(prevOutputs[i]);
        }

        api->ReleaseValue(yValue);
        api->ReleaseValue(yEmbValue);
        for (int i = 0; i < numKvTensors; i++) api->ReleaseValue(kvValues[i]);
        api->ReleaseRunOptions(runOptions);
        api->ReleaseMemoryInfo(cpuMemInfo);

        env->ReleaseLongArrayElements(yShapeJ, yShapeArr, JNI_ABORT);
        env->ReleaseLongArrayElements(yDataJ, yDataRaw, JNI_ABORT);
        env->ReleaseLongArrayElements(yEmbShapeJ, yEmbShapeArr, JNI_ABORT);
        env->ReleaseFloatArrayElements(yEmbDataJ, yEmbDataRaw, JNI_ABORT);
        env->ReleaseLongArrayElements(kvShapeJ, kvShapeArr, JNI_ABORT);
        env->ReleaseFloatArrayElements(kvDataJ, kvDataRaw, JNI_ABORT);
        env->ReleaseIntArrayElements(outputMappingJ, mapping, JNI_ABORT);

        for (int i = 0; i < numInputs; i++) {
            releaseJstr(env, inputJStrs[i], inputNames[i]);
            env->DeleteLocalRef(inputJStrs[i]);
        }
        for (int i = 0; i < numOutputs; i++) {
            releaseJstr(env, outputJStrs[i], outputNames[i]);
            env->DeleteLocalRef(outputJStrs[i]);
        }

        return result;
    }
}
