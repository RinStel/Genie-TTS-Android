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

static OrtMemoryInfo* tryCreateQnnMemoryInfo(const OrtApi* api) {
    // Try QNN provider memory so KV cache can stay on device when supported.
    OrtMemoryInfo* info = nullptr;
    OrtStatus* s = api->CreateMemoryInfo("QNN", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info);
    if (s) {
        api->ReleaseStatus(s);
        return nullptr;
    }
    return info;
}

static jlongArray extractTokensFromOutput(JNIEnv* env, const OrtApi* api, OrtValue* yValue, int generatedSteps) {
    if (!yValue || generatedSteps <= 0) return env->NewLongArray(0);

    void* yData = nullptr;
    OrtStatus* status = api->GetTensorMutableData(yValue, &yData);
    if (status) {
        LOGE("GetTensorMutableData(y) failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return env->NewLongArray(0);
    }

    OrtTensorTypeAndShapeInfo* shapeInfo = nullptr;
    status = api->GetTensorTypeAndShape(yValue, &shapeInfo);
    if (status) {
        LOGE("GetTensorTypeAndShape failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return env->NewLongArray(0);
    }

    size_t numElements = 0;
    status = api->GetTensorShapeElementCount(shapeInfo, &numElements);
    api->ReleaseTensorTypeAndShapeInfo(shapeInfo);
    if (status) {
        LOGE("GetTensorShapeElementCount failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return env->NewLongArray(0);
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

    OrtSession* session = (OrtSession*) sessionPtr;
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

    // --- Create memory info ---
    OrtMemoryInfo* cpuMemInfo = nullptr;
    api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpuMemInfo);

    // Try to create QNN device memory info for IO binding.
    OrtMemoryInfo* qnnMemInfo = nullptr;
    bool useIoBinding = false;
    if (useQnnIoBindingJ == JNI_TRUE) {
        qnnMemInfo = tryCreateQnnMemoryInfo(api);
        useIoBinding = (qnnMemInfo != nullptr);
    }
    if (useQnnIoBindingJ == JNI_TRUE && useIoBinding) {
        LOGI("QNN device memory available - using IO Binding");
    } else if (useQnnIoBindingJ == JNI_TRUE) {
        LOGI("QNN device memory unavailable - using array-based Run");
    }

    // --- Create run options ---
    OrtRunOptions* runOptions = nullptr;
    api->CreateRunOptions(&runOptions);

    // --- Create initial state tensors ---
    jlong* yShapeArr = env->GetLongArrayElements(yShapeJ, nullptr);
    int yShapeLen = env->GetArrayLength(yShapeJ);
    jlong* yDataRaw = env->GetLongArrayElements(yDataJ, nullptr);
    int yDataLen = env->GetArrayLength(yDataJ);

    OrtValue* yValue = nullptr;
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        cpuMemInfo, yDataRaw, yDataLen * sizeof(int64_t),
        yShapeArr, yShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &yValue);
    if (status) {
        LOGE("CreateTensor y failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return nullptr;
    }

    jlong* yEmbShapeArr = env->GetLongArrayElements(yEmbShapeJ, nullptr);
    int yEmbShapeLen = env->GetArrayLength(yEmbShapeJ);
    jfloat* yEmbDataRaw = env->GetFloatArrayElements(yEmbDataJ, nullptr);
    int yEmbDataLen = env->GetArrayLength(yEmbDataJ);

    OrtValue* yEmbValue = nullptr;
    status = api->CreateTensorWithDataAsOrtValue(
        cpuMemInfo, yEmbDataRaw, yEmbDataLen * sizeof(float),
        yEmbShapeArr, yEmbShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &yEmbValue);
    if (status) {
        LOGE("CreateTensor y_emb failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return nullptr;
    }

    jlong* kvShapeArr = env->GetLongArrayElements(kvShapeJ, nullptr);
    int kvShapeLen = env->GetArrayLength(kvShapeJ);
    jfloat* kvDataRaw = env->GetFloatArrayElements(kvDataJ, nullptr);

    size_t kvPerTensor = 1;
    for (int i = 0; i < kvShapeLen; i++) {
        kvPerTensor *= (size_t)kvShapeArr[i];
    }
    size_t kvBytesPerTensor = kvPerTensor * sizeof(float);

    std::vector<OrtValue*> kvValues(numKvTensors, nullptr);
    for (int i = 0; i < numKvTensors; i++) {
        void* dataPtr = kvDataRaw + (i * kvPerTensor);
        status = api->CreateTensorWithDataAsOrtValue(
            cpuMemInfo, dataPtr, kvBytesPerTensor,
            kvShapeArr, kvShapeLen, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &kvValues[i]);
        if (status) {
            LOGE("CreateTensor kv[%d] failed: %s", i, api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            api->ReleaseValue(yValue);
            api->ReleaseValue(yEmbValue);
            for (int j = 0; j < i; j++) api->ReleaseValue(kvValues[j]);
            return nullptr;
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
    api->GetAllocatorWithDefaultOptions(&defaultAlloc);

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
            for (int i = 0; i < numOutputs; i++) {
                if (i == 2) {
                    api->BindOutputToDevice(binding, outputNames[i], cpuMemInfo);
                } else {
                    api->BindOutputToDevice(binding, outputNames[i], qnnMemInfo);
                }
            }

            std::vector<OrtValue*> prevOutputs(numOutputs, nullptr);

            for (int step = 0; step < maxSteps && !stopped; step++) {
                // Rebind inputs every step because they point to the previous iteration outputs.
                api->ClearBoundInputs(binding);
                for (int i = 0; i < numInputs; i++) {
                    status = api->BindInput(binding, inputNames[i], inputs[i]);
                    if (status) {
                        LOGE("BindInput[%d] failed: %s", i, api->GetErrorMessage(status));
                        api->ReleaseStatus(status);
                        break;
                    }
                }

                // Run with binding so QNN can keep large KV tensors resident when supported.
                status = api->RunWithBinding(session, runOptions, binding);
                if (status) {
                    LOGE("RunWithBinding failed at step %d: %s", step, api->GetErrorMessage(status));
                    api->ReleaseStatus(status);
                    break;
                }

                // Ensure bound outputs are materialized before reading stop_condition.
                status = api->SynchronizeBoundOutputs(binding);
                if (status) {
                    LOGE("SynchronizeBoundOutputs failed: %s", api->GetErrorMessage(status));
                    api->ReleaseStatus(status);
                }

                generatedSteps++;
                notifyDecoderProgress(env, progressCallback, progressMethod, generatedSteps, maxSteps);

                // ORT allocates the pointer array; OrtValue ownership moves into prevOutputs below.
                OrtValue** outputValues = nullptr;
                size_t numOut = 0;
                status = api->GetBoundOutputValues(binding, defaultAlloc, &outputValues, &numOut);
                if (status || numOut != (size_t)numOutputs) {
                    LOGE("GetBoundOutputValues failed: %s", status ? api->GetErrorMessage(status) : "output count mismatch");
                    if (status) api->ReleaseStatus(status);
                    break;
                }

                // stop_condition is bound on CPU so it can be checked without copying KV tensors back.
                void* stopData = nullptr;
                status = api->GetTensorMutableData(outputValues[2], &stopData);
                if (status) {
                    LOGE("GetTensorMutableData(stop) failed: %s", api->GetErrorMessage(status));
                    api->ReleaseStatus(status);
                    // Release output values before breaking.
                    for (size_t i = 0; i < numOut; i++) {
                        if (outputValues[i]) api->ReleaseValue(outputValues[i]);
                    }
                    defaultAlloc->Free(defaultAlloc, outputValues);
                    break;
                }
                bool stop = *reinterpret_cast<bool*>(stopData);

                // Release outputs from two iterations ago after inputs no longer reference them.
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
                    if (i == 2) {
                        api->BindOutputToDevice(binding, outputNames[i], cpuMemInfo);
                    } else {
                        api->BindOutputToDevice(binding, outputNames[i], qnnMemInfo);
                    }
                }

                // Map model outputs back to the next step inputs; output[2] is stop_condition.
                for (int i = 0; i < numInputs; i++) {
                    inputs[i] = prevOutputs[mapping[i]];
                }

                // Free only the OrtValue* array; the OrtValue objects are owned by prevOutputs.
                defaultAlloc->Free(defaultAlloc, outputValues);

                if (stop) {
                    stopped = true;
                    break;
                }
            }

            // inputs[0] points to the final y tensor through the output mapping.
            jlongArray result = extractTokensFromOutput(env, api, inputs[0], generatedSteps);

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
            api->ReleaseMemoryInfo(qnnMemInfo);

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
    }

    // =====================================================================
    // FALLBACK: Array-based Run (no IO binding)
    // =====================================================================
    {
        std::vector<OrtValue*> outputs(numOutputs, nullptr);
        std::vector<OrtValue*> prevOutputs(numOutputs, nullptr);

        for (int step = 0; step < maxSteps && !stopped; step++) {
            for (int i = 0; i < numOutputs; i++) outputs[i] = nullptr;

            status = api->Run(
                session, runOptions,
                inputNames.data(), (const OrtValue* const*)inputs.data(), (size_t)numInputs,
                outputNames.data(), (size_t)numOutputs, outputs.data());

            if (status) {
                LOGE("Run failed at step %d: %s", step, api->GetErrorMessage(status));
                api->ReleaseStatus(status);
                break;
            }

            generatedSteps++;
            notifyDecoderProgress(env, progressCallback, progressMethod, generatedSteps, maxSteps);

            // stop_condition is a CPU scalar in the array-based path.
            void* stopData = nullptr;
            status = api->GetTensorMutableData(outputs[2], &stopData);
            if (status) {
                LOGE("GetTensorMutableData(stop) failed: %s", api->GetErrorMessage(status));
                api->ReleaseStatus(status);
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
        jlongArray result = extractTokensFromOutput(env, api, finalY, generatedSteps);

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
        if (qnnMemInfo) api->ReleaseMemoryInfo(qnnMemInfo);

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
