package dev.rinstel.genie_tts.inference

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.lang.reflect.Array
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

class OrtCpuBackend(
    override val backend: ExecutionBackend = ExecutionBackend.CPU,
    private val configureSessionOptions: (OrtSession.SessionOptions) -> Unit = {},
    private val traceLogger: InferenceTraceLogger = InferenceTraceLogger.None,
    private val configureRoleSessionOptions: ((OrtSession.SessionOptions, InferenceModelRole) -> Unit)? = null,
    // Current CPU device measurements favor the FP32 shells. Keep the FP16
    // path available for explicit device-specific experiments.
    private val preferFp16Graphs: Boolean = false,
) : InferenceBackend, PreparedTtsBackend {

    private val environment: OrtEnvironment by lazy(LazyThreadSafetyMode.NONE) {
        OrtEnvironment.getEnvironment()
    }
    private val sessions: MutableMap<String, OrtSession> = linkedMapOf()
    private val sessionFiles: MutableMap<String, File> = linkedMapOf()
    private var modelFiles: RequiredModelFiles? = null
    private var multiReferenceCapabilityResolved = false
    private var cachedMultiReferenceCapability: MultiReferenceCapability? = null

    @Volatile
    private var retainT2SSessionsAfterGeneration = false

    @Volatile
    internal var runtimeCallbacks: BackendRuntimeCallbacks? = null

    internal fun reportProgress(
        stage: GenerationStage,
        message: String,
        progressPercent: Int?,
        progressLabel: String?,
    ) {
        runtimeCallbacks?.onStage(stage, message, progressPercent, progressLabel)
    }

    // Initialization only validates and indexes model files. Heavy ORT
    // sessions are opened on demand so runtime feature models do not overlap
    // with the T2S/vocoder working set.
    override fun isInitialized(): Boolean = sessionFiles.isNotEmpty()

    fun hasPromptEncoder(): Boolean =
        sessionFiles.keys.any { it.contains("prompt_encoder", ignoreCase = true) }

    fun multiReferenceCapability(): MultiReferenceCapability? {
        if (multiReferenceCapabilityResolved) return cachedMultiReferenceCapability
        val resolved = when {
            !hasPromptEncoder() -> null
            !sessionFiles.keys.any { matchesSessionName(it, MULTI_REFERENCE_PROMPT_ENCODER) } ->
                MultiReferenceCapability(supported = false)
            else -> runCatching {
                MultiReferenceCapability.fromSession(session(MULTI_REFERENCE_PROMPT_ENCODER))
            }.getOrDefault(MultiReferenceCapability(supported = false))
        }
        multiReferenceCapabilityResolved = true
        cachedMultiReferenceCapability = resolved
        return resolved
    }

    data class CpuThreadPlan(
        val intraOpThreads: Int,
        val interOpThreads: Int,
    )

    fun applySessionOptions(options: OrtSession.SessionOptions) {
        configureSessionOptions(options)
    }

    fun applySessionOptions(
        options: OrtSession.SessionOptions,
        role: InferenceModelRole,
    ) {
        configureRoleSessionOptions?.invoke(options, role) ?: configureSessionOptions(options)
    }

    override fun initialize(modelRoot: File, characterModel: CharacterModel): SessionSummary {
        close()

        val modelDirectory = File(modelRoot, characterModel.relativeModelDirectory)
        val sessionModelNames = characterModel.modelFiles.sessionModelNames(
            modelDirectory,
            preferFp16 = preferFp16Graphs,
        )
        for (modelName in sessionModelNames) {
            val modelFile = File(modelDirectory, modelName)
            require(modelFile.exists()) { "Model file not found: ${modelFile.absolutePath}" }
            sessionFiles[modelName] = modelFile
        }
        modelFiles = characterModel.modelFiles

        // Do not create any sessions here. RoBERTa, HuBERT, speaker encoder,
        // and the character graphs are loaded in disjoint inference phases.
        return SessionSummary(
            backend = backend,
            sessionNames = sessionFiles.keys.toList(),
            inputNames = emptyList(),
            outputNames = emptyList(),
        )
    }

    fun warmUp() {
        for ((_, session) in sessions) {
            warmUpSession(environment, session)
        }
    }

    private fun createSession(
        modelFile: File,
        role: InferenceModelRole,
    ): OrtSession {
        val options = OrtSession.SessionOptions()
        return try {
            applySessionOptions(options, role)
            // Prompt weights are already materialized as the model's normal
            // FP32 external-data file during model import. Let ORT read that
            // file directly so the app heap never holds a second Tensor set.
            environment.createSession(modelFile.absolutePath, options)
        } finally {
            options.close()
        }
    }

    override fun generatePrepared(input: TtsPreparedInput, maxDecoderSteps: Int): TtsGenerationResult {
        require(isInitialized()) { "CPU backend is not initialized." }

        val timer = InferenceTimer(traceLogger)
        val ownedTensors = mutableListOf<OnnxTensor>()
        var encoderResult: OrtSession.Result? = null
        var previousDecoderResult: OrtSession.Result? = null
        var finalDecoderResult: OrtSession.Result? = null
        var generationSucceeded = false

        try {
            val refSeq = createLongTensor(input.refSeq).also(ownedTensors::add)
            val textSeq = createLongTensor(input.textSeq).also(ownedTensors::add)
            val refBert = createFloatTensor(input.refBert).also(ownedTensors::add)
            val textBert = createFloatTensor(input.textBert).also(ownedTensors::add)
            val sslContent = createFloatTensor(input.sslContent).also(ownedTensors::add)

            encoderResult = timer.measure("t2s_encoder_ms") {
                session(T2S_ENCODER).run(
                    mapOf(
                        "ref_seq" to refSeq,
                        "text_seq" to textSeq,
                        "ref_bert" to refBert,
                        "text_bert" to textBert,
                        "ssl_content" to sslContent,
                    ),
                )
            }

            val currentEncoderResult = requireNotNull(encoderResult)
            reportProgress(
                GenerationStage.RUNNING_INFERENCE,
                "running_inference",
                GenerationProgress.FIRST_DECODER,
                "First decoder",
            )
            previousDecoderResult = timer.measure("t2s_first_decoder_ms") {
                session(T2S_FIRST_STAGE_DECODER).run(
                    mapOf(
                        "x" to tensorLike(currentEncoderResult, 0),
                        "prompts" to tensorLike(currentEncoderResult, 1),
                    ),
                )
            }
            encoderResult.close()
            encoderResult = null

            val stageSession = session(T2S_STAGE_DECODER)
            val stageInputNames = stageSession.inputNames.toList()
            val decoderProgress = decoderProgressCallback(maxDecoderSteps)

            // The CPU production path intentionally uses the regular ORT array
            // loop. The native QNN decoder remains under the accelerator archive.
            val decoderInputs = mutableListOf<OnnxTensorLike>()
            val feed = mutableMapOf<String, OnnxTensorLike>()
            val runOptions = OrtSession.RunOptions().apply {
                setLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL)
            }
            var generatedSteps = 0
            var presentStartIndex = 2

            try {
                timer.measure("decoder_loop_ms") {
                    while (generatedSteps < maxDecoderSteps) {
                        val previous = requireNotNull(previousDecoderResult)
                        decoderInputs.clear()
                        decoderInputs.add(tensorLike(previous, 0))
                        decoderInputs.add(tensorLike(previous, 1))
                        for (index in presentStartIndex until previous.size()) {
                            decoderInputs.add(tensorLike(previous, index))
                        }

                        feed.clear()
                        for (i in stageInputNames.indices) {
                            feed[stageInputNames[i]] = decoderInputs[i]
                        }

                        val stageResult = stageSession.run(feed, runOptions)
                        previous.close()
                        previousDecoderResult = stageResult
                        generatedSteps += 1
                        presentStartIndex = 3
                        decoderProgress(generatedSteps, maxDecoderSteps)

                        if (booleanValue(stageResult.get(2))) {
                            break
                        }
                    }
                }
            } finally {
                runOptions.close()
            }

            finalDecoderResult = requireNotNull(previousDecoderResult)
            previousDecoderResult = null
            val predSemantic = semanticTensorFromDecoderOutput(finalDecoderResult, generatedSteps)
                .also(ownedTensors::add)
            finalDecoderResult.close()
            finalDecoderResult = null

            reportProgress(
                GenerationStage.RUNNING_INFERENCE,
                "running_inference",
                GenerationProgress.DECODER_END,
                "Decoder complete",
            )

            // Decoder results are copied into predSemantic above. Release the
            // encoder before VITS; a memory-approved hot path keeps only the
            // two stateless decoder graphs for the next sentence.
            if (!retainT2SSessionsAfterGeneration) {
                closeSessions { roleForSession(it) == InferenceModelRole.T2S }
            } else {
                releaseLoadedSessionsKeepingReusableT2S()
            }

            val vocoderInputs = if (hasPromptEncoder()) {
                timer.event(InferenceTraceEvent.TensorCount(InferenceTensorMetric.SEMANTIC_TOKENS, predSemantic.info.numElements))
                timer.event(InferenceTraceEvent.TensorShape(InferenceTensorMetric.SEMANTIC_SHAPE, predSemantic.info.shape))
                timer.event(
                    InferenceTraceEvent.TensorHash(
                        InferenceTensorMetric.SEMANTIC_HASH,
                        InferenceTraceHash.sha256(longTensorValues(predSemantic)),
                    ),
                )
                val ge = createFloatTensor(input.globalEmbedding).also(ownedTensors::add)
                val geAdvanced = createFloatTensor(input.advancedGlobalEmbedding).also(ownedTensors::add)
                mapOf(
                    "text_seq" to textSeq,
                    "pred_semantic" to predSemantic,
                    "ge" to ge,
                    "ge_advanced" to geAdvanced,
                )
            } else {
                timer.event(InferenceTraceEvent.TensorCount(InferenceTensorMetric.SEMANTIC_TOKENS, predSemantic.info.numElements))
                timer.event(InferenceTraceEvent.TensorShape(InferenceTensorMetric.SEMANTIC_SHAPE, predSemantic.info.shape))
                timer.event(
                    InferenceTraceEvent.TensorHash(
                        InferenceTensorMetric.SEMANTIC_HASH,
                        InferenceTraceHash.sha256(longTensorValues(predSemantic)),
                    ),
                )
                val refAudio = createFloatTensor(
                    requireNotNull(input.refAudio32k) { "V2 vocoder requires refAudio32k in TtsPreparedInput." }
                ).also(ownedTensors::add)
                mapOf(
                    "text_seq" to textSeq,
                    "pred_semantic" to predSemantic,
                    "ref_audio" to refAudio,
                )
            }

            // ORT exposes no callback inside one VITS graph execution. Use an
            // indeterminate bar here instead of claiming a false percentage.
            reportProgress(
                GenerationStage.RUNNING_INFERENCE,
                "running_inference",
                null,
                "Vocoder",
            )
            val vocoderResult = timer.measure("vocoder_ms") {
                session(VITS).run(vocoderInputs)
            }
            val generationResult = vocoderResult.use {
                val audioTensor = vocoderResult.get(0) as OnnxTensor
                val audio = FloatArray(audioTensor.info.numElements.toInt())
                audioTensor.floatBuffer.get(audio)
                TtsGenerationResult(
                    audio = audio,
                    shape = audioTensor.info.shape,
                )
            }
            reportProgress(
                GenerationStage.RUNNING_INFERENCE,
                "running_inference",
                GenerationProgress.VOCODER_COMPLETE,
                "Vocoder complete",
            )
            return generationResult.also { generationSucceeded = true }
        } finally {
            encoderResult?.close()
            previousDecoderResult?.close()
            finalDecoderResult?.close()
            ownedTensors.forEach(OnnxTensor::close)
            // The audio has already been copied out of ORT. VITS and the T2S
            // encoder are always released; only the optional decoder hot cache
            // survives a successful request.
            if (retainT2SSessionsAfterGeneration && generationSucceeded) {
                releaseLoadedSessionsKeepingReusableT2S()
            } else {
                releaseLoadedSessions()
            }
        }
    }

    fun encodePrompt(refAudio32k: FloatArray, speakerEmbedding: FloatTensorData): PromptEmbeddings {
        require(isInitialized()) { "CPU backend is not initialized." }
        return runPromptEncoder(PROMPT_ENCODER, refAudio32k, speakerEmbedding)
    }

    /** Release sessions used only while constructing reference conditioning. */
    fun releasePreparationSessions() {
        closeSessions { roleForSession(it) == InferenceModelRole.PROMPT_ENCODER }
    }

    /** Release all currently loaded graphs before the next feature phase. */
    fun releaseLoadedSessions() {
        closeSessions { true }
    }

    /** Release all T2S graphs when memory pressure makes the hot cache unsafe. */
    fun releaseT2SSessions() {
        closeSessions { roleForSession(it) == InferenceModelRole.T2S }
    }

    /** Keep only the decoder graphs that can be reused by the next sentence. */
    fun releaseLoadedSessionsKeepingReusableT2S() {
        closeSessions {
            roleForSession(it) != InferenceModelRole.T2S || !isReusableT2SSession(it)
        }
    }

    fun setRetainT2SSessionsAfterGeneration(retain: Boolean) {
        retainT2SSessionsAfterGeneration = retain
    }

    private fun runPromptEncoder(
        modelName: String,
        refAudio32k: FloatArray,
        speakerEmbedding: FloatTensorData,
    ): PromptEmbeddings {
        val refAudio = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(refAudio32k),
            longArrayOf(1L, refAudio32k.size.toLong()),
        )
        val svEmbedding = createFloatTensor(speakerEmbedding)
        refAudio.use {
            svEmbedding.use {
                session(modelName).run(
                    mapOf(
                        "ref_audio" to refAudio,
                        "sv_emb" to svEmbedding,
                    ),
                ).use { result ->
                    return PromptEmbeddings(
                        globalEmbedding = floatTensorData(result.get(0) as OnnxTensor),
                        advancedGlobalEmbedding = floatTensorData(result.get(1) as OnnxTensor),
                    )
                }
            }
        }
    }

    fun encodePromptBatch(
        referenceAudio32k: List<FloatArray>,
        speakerEmbeddings: List<FloatTensorData>,
    ): PromptEmbeddings {
        require(referenceAudio32k.isNotEmpty()) { "At least one reference audio is required." }
        require(referenceAudio32k.size == speakerEmbeddings.size) {
            "Reference audio and speaker embedding counts must match."
        }
        return runPromptBatchGraph(referenceAudio32k, speakerEmbeddings)
    }

    private fun runPromptBatchGraph(
        referenceAudio32k: List<FloatArray>,
        speakerEmbeddings: List<FloatTensorData>,
    ): PromptEmbeddings {
        val capability = requireNotNull(multiReferenceCapability()) {
            "The loaded model does not provide the multi-reference prompt encoder."
        }
        // This layer receives tensors rather than paths; validate the count without
        // inventing duplicate path values that would reject valid ordered slots.
        capability.validateReferenceCount(referenceAudio32k.size)
        val maxReferenceCount = capability.maxReferenceCount
        val placeholderAudioSamples = capability.placeholderAudioSamples
        val embeddingWidth = speakerEmbeddings.first().values.size
        require(speakerEmbeddings.all { it.values.size == embeddingWidth }) {
            "Speaker embedding dimensions must match."
        }
        val ownedTensors = mutableListOf<OnnxTensor>()
        val feed = linkedMapOf<String, OnnxTensorLike>()
        try {
            feed[REFERENCE_COUNT_INPUT] = OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(longArrayOf(referenceAudio32k.size.toLong())),
                longArrayOf(1L),
            ).also(ownedTensors::add)
            for (index in 0 until maxReferenceCount) {
                // Unused slots still execute the unrolled source graph. Give
                // STFT nodes a valid zero waveform; the graph masks its output
                // before reduction, so this does not affect conditioning.
                val audio = referenceAudio32k.getOrNull(index)
                    ?: FloatArray(placeholderAudioSamples)
                val embedding = speakerEmbeddings.getOrNull(index)
                    ?: FloatTensorData(FloatArray(embeddingWidth), longArrayOf(1L, embeddingWidth.toLong()))
                feed["ref_audio_$index"] = OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(audio),
                    longArrayOf(1L, audio.size.toLong()),
                ).also(ownedTensors::add)
                feed["sv_emb_$index"] = createFloatTensor(embedding).also(ownedTensors::add)
            }
            session(MULTI_REFERENCE_PROMPT_ENCODER).run(feed).use { result ->
                return PromptEmbeddings(
                    globalEmbedding = floatTensorData(result.get(0) as OnnxTensor),
                    advancedGlobalEmbedding = floatTensorData(result.get(1) as OnnxTensor),
                )
            }
        } finally {
            ownedTensors.forEach(OnnxTensor::close)
        }
    }

    override fun close() {
        sessions.values.forEach(OrtSession::close)
        sessions.clear()
        sessionFiles.clear()
        modelFiles = null
        multiReferenceCapabilityResolved = false
        cachedMultiReferenceCapability = null
        retainT2SSessionsAfterGeneration = false
    }

    private fun session(modelName: String): OrtSession {
        val loadedEntry = sessions.entries.firstOrNull {
            matchesSessionName(it.key, modelName)
        }
        if (loadedEntry != null) return loadedEntry.value

        val selectedName = sessionFiles.keys.firstOrNull {
            matchesSessionName(it, modelName)
        }
        requireNotNull(selectedName) { "Session is not available: $modelName" }
        val selectedFile = requireNotNull(sessionFiles[selectedName])
        val selectedRole = roleForSession(selectedName)
        val loaded = try {
            selectedName to createSession(selectedFile, selectedRole)
        } catch (error: Exception) {
            val fallbackName = modelFiles?.fallbackSessionModelName(selectedName)
            val fallbackFile = fallbackName?.let { File(selectedFile.parentFile, it) }
            if (fallbackName == null || fallbackFile == null || !fallbackFile.isFile) {
                throw error
            }
            val fallbackSession = createSession(fallbackFile, roleForSession(fallbackName))
            sessionFiles.remove(selectedName)
            sessionFiles[fallbackName] = fallbackFile
            fallbackName to fallbackSession
        }
        val loadedModelName = loaded.first
        val loadedSession = loaded.second
        sessions[loadedModelName] = loadedSession
        traceLogger.event(
            InferenceTraceEvent.Model(
                InferenceTraceRecord(
                    modelRole = roleForSession(loadedModelName),
                    provider = ModelRoleSessionOptions.providerFor(
                        backend,
                        roleForSession(loadedModelName),
                    ),
                ),
            ),
        )
        warmUpSession(environment, loadedSession)
        return loadedSession
    }

    private fun closeSessions(predicate: (String) -> Boolean) {
        sessions.keys.filter(predicate).forEach { name ->
            sessions.remove(name)?.close()
        }
    }

    private fun roleForSession(modelName: String): InferenceModelRole =
        ModelRoleSessionOptions.roleForModelFile(modelName)

    private fun isReusableT2SSession(modelName: String): Boolean =
        roleForSession(modelName) == InferenceModelRole.T2S &&
            (matchesSessionName(modelName, T2S_FIRST_STAGE_DECODER) ||
                matchesSessionName(modelName, T2S_STAGE_DECODER))

    private fun matchesSessionName(actualName: String, requestedName: String): Boolean {
        if (actualName.equals(requestedName, ignoreCase = true)) return true
        val keyword = requestedName
            .substringBefore("_fp32.onnx")
            .substringBefore("_fp16.onnx")
        return actualName.contains(keyword, ignoreCase = true)
    }

    private fun createLongTensor(data: LongTensorData): OnnxTensor =
        OnnxTensor.createTensor(environment, LongBuffer.wrap(data.values), data.shape)

    private fun createFloatTensor(data: FloatTensorData): OnnxTensor =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(data.values), data.shape)

    private fun tensorLike(result: OrtSession.Result, index: Int): OnnxTensorLike =
        result.get(index) as OnnxTensorLike

    private fun longTensorValues(tensor: OnnxTensor): LongArray {
        val values = LongArray(tensor.info.numElements.toInt())
        tensor.longBuffer.get(values)
        return values
    }

    private fun semanticTensorFromDecoderOutput(
        decoderResult: OrtSession.Result,
        generatedSteps: Int,
    ): OnnxTensor {
        val yTensor = decoderResult.get(0) as OnnxTensor
        val shape = yTensor.info.shape
        require(shape.size == 2) { "Expected decoder y shape [batch, tokens], got ${shape.contentToString()}" }
        require(shape[0] == 1L) { "Only batch size 1 is supported for prototype CPU generation." }

        val source = LongArray(yTensor.info.numElements.toInt())
        yTensor.longBuffer.get(source)
        val semantic = SemanticTokenSlicing.completedTokens(source, generatedSteps)

        return OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(semantic),
            longArrayOf(1L, 1L, semantic.size.toLong()),
        )
    }

    private fun booleanValue(value: OnnxValue): Boolean =
        scalarBoolean(value.value)

    private fun decoderProgressCallback(maxDecoderSteps: Int): (Int, Int) -> Unit {
        var lastPercent = -1
        val safeMaxSteps = maxDecoderSteps.coerceAtLeast(1)
        val callback: (Int, Int) -> Unit = { generatedSteps, _ ->
            val overallPercent = GenerationProgress.decoderPercent(generatedSteps, safeMaxSteps)
            if (overallPercent != lastPercent) {
                lastPercent = overallPercent
                reportProgress(
                    GenerationStage.RUNNING_INFERENCE,
                    "running_inference",
                    overallPercent,
                    "Decoder $generatedSteps/$safeMaxSteps",
                )
            }
        }
        callback(0, safeMaxSteps)
        return callback
    }

    private fun floatTensorData(tensor: OnnxTensor): FloatTensorData {
        val values = FloatArray(tensor.info.numElements.toInt())
        tensor.floatBuffer.get(values)
        return FloatTensorData(values, tensor.info.shape)
    }

    private fun scalarBoolean(value: Any?): Boolean {
        if (value is Boolean) {
            return value
        }
        val valueClass = value?.javaClass
        if (valueClass?.isArray == true && Array.getLength(value) > 0) {
            return scalarBoolean(Array.get(value, 0))
        }
        return false
    }

    companion object {
        private const val T2S_ENCODER = "t2s_encoder_fp32.onnx"
        private const val T2S_FIRST_STAGE_DECODER = "t2s_first_stage_decoder_fp32.onnx"
        private const val T2S_STAGE_DECODER = "t2s_stage_decoder_fp32.onnx"
        private const val PROMPT_ENCODER = "prompt_encoder_fp32.onnx"
        private const val MULTI_REFERENCE_PROMPT_ENCODER = "prompt_encoder_multi_fp32.onnx"
        private const val REFERENCE_COUNT_INPUT = "reference_count"
        private const val VITS = "vits_fp32.onnx"
        fun cpuThreadPlan(availableProcessors: Int): CpuThreadPlan {
            val processors = availableProcessors.coerceAtLeast(1)
            return CpuThreadPlan(
                intraOpThreads = processors.coerceAtMost(8),
                interOpThreads = 1,
            )
        }

        fun cpuThreadPlan(
            availableProcessors: Int,
            role: InferenceModelRole,
        ): CpuThreadPlan {
            val processors = availableProcessors.coerceAtLeast(1)
            // T2S executes one autoregressive token at a time; a smaller
            // intra-op pool avoids repeatedly scheduling eight workers.
            val maxIntraOpThreads = if (role == InferenceModelRole.T2S) 4 else 8
            return CpuThreadPlan(
                intraOpThreads = processors.coerceAtMost(maxIntraOpThreads),
                interOpThreads = 1,
            )
        }

        fun configureCpuSessionOptions(options: OrtSession.SessionOptions) {
            configureCpuSessionOptions(options, cpuThreadPlan(Runtime.getRuntime().availableProcessors()))
        }

        fun configureCpuSessionOptionsForRole(
            options: OrtSession.SessionOptions,
            role: InferenceModelRole,
        ) {
            configureCpuSessionOptions(
                options,
                cpuThreadPlan(Runtime.getRuntime().availableProcessors(), role),
            )
        }

        private fun configureCpuSessionOptions(
            options: OrtSession.SessionOptions,
            threadPlan: CpuThreadPlan,
        ) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            options.setMemoryPatternOptimization(true)
            options.setCPUArenaAllocator(true)
            options.setIntraOpNumThreads(threadPlan.intraOpThreads)
            options.setInterOpNumThreads(threadPlan.interOpThreads)
        }

        /** Prime ORT's CPU kernels; exported graphs may reject shape-1 dummies. */
        fun warmUpSession(environment: OrtEnvironment, session: OrtSession) {
            try {
                val inputInfo = session.inputInfo
                val dummyTensors = mutableListOf<OnnxTensor>()
                val feed = mutableMapOf<String, OnnxTensorLike>()

                for ((name, nodeInfo) in inputInfo) {
                    val tensorInfo = nodeInfo.info as? TensorInfo ?: continue
                    val shape = tensorInfo.shape.map { if (it <= 0) 1L else it }.toLongArray()
                    val numElements = shape.fold(1L) { acc, d -> acc * d }.toInt().coerceAtLeast(1)

                    val tensor = when (tensorInfo.type) {
                        OnnxJavaType.FLOAT -> OnnxTensor.createTensor(
                            environment, FloatBuffer.wrap(FloatArray(numElements)), shape,
                        )
                        OnnxJavaType.FLOAT16 -> OnnxTensor.createTensor(
                            environment,
                            ShortBuffer.wrap(ShortArray(numElements)),
                            shape,
                            OnnxJavaType.FLOAT16,
                        )
                        OnnxJavaType.INT64 -> OnnxTensor.createTensor(
                            environment, LongBuffer.wrap(LongArray(numElements)), shape,
                        )
                        else -> continue
                    }
                    dummyTensors.add(tensor)
                    feed[name] = tensor
                }

                if (feed.isNotEmpty()) {
                    val runOptions = OrtSession.RunOptions().apply {
                        setLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL)
                    }
                    try {
                        session.run(feed, runOptions).use { }
                    } finally {
                        runOptions.close()
                    }
                }
                dummyTensors.forEach(OnnxTensor::close)
            } catch (_: Exception) {
                // The real request owns the authoritative input shapes.
            }
        }

        fun sessionModelPaths(modelRoot: File, characterModel: CharacterModel): List<File> {
            val modelDirectory = File(modelRoot, characterModel.relativeModelDirectory)
            return characterModel.modelFiles.sessionModelNames(modelDirectory)
                .map { File(modelDirectory, it) }
        }

    }
}
