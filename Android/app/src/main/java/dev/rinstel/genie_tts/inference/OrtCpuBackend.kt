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

class OrtCpuBackend(
    override val backend: ExecutionBackend = ExecutionBackend.CPU,
    private val configureSessionOptions: (OrtSession.SessionOptions) -> Unit = {},
    private val traceLogger: InferenceTraceLogger = InferenceTraceLogger.None,
) : InferenceBackend, PreparedTtsBackend {

    private val environment: OrtEnvironment by lazy(LazyThreadSafetyMode.NONE) {
        OrtEnvironment.getEnvironment()
    }
    private val sessions: MutableMap<String, OrtSession> = linkedMapOf()

    @Volatile
    internal var runtimeCallbacks: BackendRuntimeCallbacks? = null

    override fun isInitialized(): Boolean = sessions.isNotEmpty()

    fun hasPromptEncoder(): Boolean =
        sessions.keys.any { it.contains("prompt_encoder", ignoreCase = true) }

    data class CpuThreadPlan(
        val intraOpThreads: Int,
        val interOpThreads: Int,
    )

    fun applySessionOptions(options: OrtSession.SessionOptions) {
        configureSessionOptions(options)
    }

    override fun initialize(modelRoot: File, characterModel: CharacterModel): SessionSummary {
        close()

        val options = OrtSession.SessionOptions()
        configureSessionOptions(options)
        val modelDirectory = File(modelRoot, characterModel.relativeModelDirectory)
        for (modelName in characterModel.modelFiles.sessionModels) {
            val modelFile = File(modelDirectory, modelName)
            require(modelFile.exists()) { "Model file not found: ${modelFile.absolutePath}" }
            sessions[modelName] = environment.createSession(modelFile.absolutePath, options)
        }

        warmUp()

        return SessionSummary(
            backend = backend,
            sessionNames = sessions.keys.toList(),
            inputNames = sessions.values.flatMap { it.inputNames.toList() }.distinct(),
            outputNames = sessions.values.flatMap { it.outputNames.toList() }.distinct(),
        )
    }

    fun warmUp() {
        for ((_, session) in sessions) {
            warmUpSession(environment, session)
        }
    }

    override fun generatePrepared(input: TtsPreparedInput, maxDecoderSteps: Int): TtsGenerationResult {
        require(sessions.isNotEmpty()) { "CPU backend is not initialized." }

        val timer = InferenceTimer(traceLogger)
        val ownedTensors = mutableListOf<OnnxTensor>()
        var encoderResult: OrtSession.Result? = null
        var previousDecoderResult: OrtSession.Result? = null
        var finalDecoderResult: OrtSession.Result? = null

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
            val stageOutputNames = stageSession.outputNames.toList()
            val decoderProgress = decoderProgressCallback(maxDecoderSteps)

            val predSemantic = if (NativeDecoderLoop.isAvailable()) {
                val fsd = requireNotNull(previousDecoderResult)
                previousDecoderResult = null

                val yTensor = fsd.get(0) as OnnxTensor
                val yData = LongArray(yTensor.info.numElements.toInt())
                yTensor.longBuffer.get(yData)
                val yShape = yTensor.info.shape

                val yEmbTensor = fsd.get(1) as OnnxTensor
                val yEmbData = FloatArray(yEmbTensor.info.numElements.toInt())
                yEmbTensor.floatBuffer.get(yEmbData)
                val yEmbShape = yEmbTensor.info.shape

                val numKv = (fsd.size() - 2) / 2 * 2
                val kvTensor0 = fsd.get(2) as OnnxTensor
                val kvShape = kvTensor0.info.shape
                val kvPerTensor = kvTensor0.info.numElements.toInt()
                val kvData = FloatArray(kvPerTensor * numKv)
                for (i in 0 until numKv) {
                    val t = fsd.get(2 + i) as OnnxTensor
                    t.floatBuffer.get(kvData, i * kvPerTensor, kvPerTensor)
                }
                fsd.close()

                val outputMapping = IntArray(stageInputNames.size) { i ->
                    if (i < 2) i else i + 1
                }

                val semantic = timer.measure("decoder_loop_ms") {
                    NativeDecoderLoop.runDecoderLoop(
                        session = stageSession,
                        inputNames = stageInputNames,
                        outputNames = stageOutputNames,
                        initialY = yData, yShape = yShape,
                        initialYEmb = yEmbData, yEmbShape = yEmbShape,
                        initialKv = kvData, kvShape = kvShape,
                        numKvTensors = numKv,
                        outputMapping = outputMapping,
                        useQnnIoBinding = backend == ExecutionBackend.QNN,
                        maxSteps = maxDecoderSteps,
                        progressCallback = decoderProgress,
                    )
                }

                val finalSemantic = if (semantic.isNotEmpty()) {
                    semantic[semantic.lastIndex] = 0L
                    semantic
                } else { longArrayOf(0L) }

                OnnxTensor.createTensor(
                    environment,
                    LongBuffer.wrap(finalSemantic),
                    longArrayOf(1L, 1L, finalSemantic.size.toLong()),
                ).also(ownedTensors::add)
            } else {
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
                            decoderProgress.onProgress(generatedSteps, maxDecoderSteps)

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
                semanticTensorFromDecoderOutput(finalDecoderResult, generatedSteps)
                    .also(ownedTensors::add)
            }

            val vocoderInputs = if (hasPromptEncoder()) {
                timer.event(InferenceTraceEvent.TensorCount(InferenceTensorMetric.SEMANTIC_TOKENS, predSemantic.info.numElements))
                timer.event(InferenceTraceEvent.TensorShape(InferenceTensorMetric.SEMANTIC_SHAPE, predSemantic.info.shape))
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
                val refAudio = createFloatTensor(
                    requireNotNull(input.refAudio32k) { "V2 vocoder requires refAudio32k in TtsPreparedInput." }
                ).also(ownedTensors::add)
                mapOf(
                    "text_seq" to textSeq,
                    "pred_semantic" to predSemantic,
                    "ref_audio" to refAudio,
                )
            }

            val vocoderResult = timer.measure("vocoder_ms") {
                session(VITS).run(vocoderInputs)
            }
            vocoderResult.use {
                val audioTensor = vocoderResult.get(0) as OnnxTensor
                val audio = FloatArray(audioTensor.info.numElements.toInt())
                audioTensor.floatBuffer.get(audio)
                return TtsGenerationResult(
                    audio = audio,
                    shape = audioTensor.info.shape,
                )
            }
        } finally {
            encoderResult?.close()
            previousDecoderResult?.close()
            finalDecoderResult?.close()
            ownedTensors.forEach(OnnxTensor::close)
        }
    }

    fun encodePrompt(refAudio32k: FloatArray, speakerEmbedding: FloatTensorData): PromptEmbeddings {
        require(sessions.isNotEmpty()) { "CPU backend is not initialized." }
        val refAudio = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(refAudio32k),
            longArrayOf(1L, refAudio32k.size.toLong()),
        )
        val svEmbedding = createFloatTensor(speakerEmbedding)
        refAudio.use {
            svEmbedding.use {
                session(PROMPT_ENCODER).run(
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

    override fun close() {
        sessions.values.forEach(OrtSession::close)
        sessions.clear()
    }

    private fun session(modelName: String): OrtSession {
        if (sessions.containsKey(modelName)) {
            return sessions.getValue(modelName)
        }
        val keyword = modelName.substringBefore("_fp32.onnx")
        val match = sessions.entries.firstOrNull { it.key.contains(keyword, ignoreCase = true) }
        return requireNotNull(match?.value) { "Session is not loaded: $modelName" }
    }

    private fun createLongTensor(data: LongTensorData): OnnxTensor =
        OnnxTensor.createTensor(environment, LongBuffer.wrap(data.values), data.shape)

    private fun createFloatTensor(data: FloatTensorData): OnnxTensor =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(data.values), data.shape)

    private fun tensorLike(result: OrtSession.Result, index: Int): OnnxTensorLike =
        result.get(index) as OnnxTensorLike

    private fun semanticTensorFromDecoderOutput(
        decoderResult: OrtSession.Result,
        generatedSteps: Int,
    ): OnnxTensor {
        val yTensor = decoderResult.get(0) as OnnxTensor
        val shape = yTensor.info.shape
        require(shape.size == 2) { "Expected decoder y shape [batch, tokens], got ${shape.contentToString()}" }
        require(shape[0] == 1L) { "Only batch size 1 is supported for prototype CPU generation." }

        val tokenCount = minOf(generatedSteps.coerceAtLeast(1), shape[1].toInt())
        val source = LongArray(yTensor.info.numElements.toInt())
        yTensor.longBuffer.get(source)
        val start = source.size - tokenCount
        var semantic = source.copyOfRange(start, source.size)

        // EOS trimming: remove tokens whose value >= 1024 (EOS/special tokens).
        // Mirrors Python Inference.py:41-44 — truncate at the first EOS occurrence.
        val firstEosIndex = semantic.indexOfFirst { it >= 1024L }
        if (firstEosIndex >= 0) {
            semantic = if (firstEosIndex > 0) {
                semantic.copyOfRange(0, firstEosIndex)
            } else {
                longArrayOf(0L)
            }
        }

        if (semantic.isNotEmpty()) {
            semantic[semantic.lastIndex] = 0L
        }

        return OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(semantic),
            longArrayOf(1L, 1L, semantic.size.toLong()),
        )
    }

    private fun booleanValue(value: OnnxValue): Boolean =
        scalarBoolean(value.value)

    private fun decoderProgressCallback(maxDecoderSteps: Int): NativeDecoderLoop.DecoderProgressCallback {
        var lastPercent = -1
        val safeMaxSteps = maxDecoderSteps.coerceAtLeast(1)
        return NativeDecoderLoop.DecoderProgressCallback { generatedSteps, _ ->
            val decoderPercent = (generatedSteps * 100 / safeMaxSteps).coerceIn(0, 100)
            val overallPercent = (
                DECODER_PROGRESS_START +
                    (DECODER_PROGRESS_END - DECODER_PROGRESS_START) * decoderPercent / 100
                ).coerceIn(DECODER_PROGRESS_START, DECODER_PROGRESS_END)
            if (overallPercent != lastPercent) {
                lastPercent = overallPercent
                runtimeCallbacks?.onStage(
                    GenerationStage.RUNNING_INFERENCE,
                    "running_inference",
                    overallPercent,
                    "Decoding $generatedSteps/$safeMaxSteps",
                )
            }
        }.also { it.onProgress(0, safeMaxSteps) }
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
        private const val VITS = "vits_fp32.onnx"
        private const val DECODER_PROGRESS_START = 82
        private const val DECODER_PROGRESS_END = 94

        fun cpuThreadPlan(availableProcessors: Int): CpuThreadPlan {
            val processors = availableProcessors.coerceAtLeast(1)
            return CpuThreadPlan(
                intraOpThreads = processors.coerceAtMost(8),
                interOpThreads = 1,
            )
        }

        fun configureCpuSessionOptions(options: OrtSession.SessionOptions) {
            val threadPlan = cpuThreadPlan(Runtime.getRuntime().availableProcessors())
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            options.setMemoryPatternOptimization(true)
            options.setCPUArenaAllocator(true)
            options.setIntraOpNumThreads(threadPlan.intraOpThreads)
            options.setInterOpNumThreads(threadPlan.interOpThreads)
        }

        // Kept for XNNPACK parity debugging; MainActivity does not expose it as a selectable backend yet.
        fun configureXnnpackSessionOptions(options: OrtSession.SessionOptions) {
            val threadPlan = cpuThreadPlan(Runtime.getRuntime().availableProcessors())
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            options.setMemoryPatternOptimization(true)
            options.setCPUArenaAllocator(true)
            options.addXnnpack(mapOf(
                "intra_op_num_threads" to threadPlan.intraOpThreads.toString(),
            ))
        }

        fun configureQnnSessionOptions(
            options: OrtSession.SessionOptions,
            providerOptions: Map<String, String>,
        ) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            options.setMemoryPatternOptimization(true)
            options.setCPUArenaAllocator(true)
            // Add QNN EP last so the shared ORT setup above stays identical to CPU/XNNPACK.
            options.addQnn(providerOptions)
        }

        fun sessionModelPaths(modelRoot: File, characterModel: CharacterModel): List<File> {
            val modelDirectory = File(modelRoot, characterModel.relativeModelDirectory)
            return characterModel.modelFiles.sessionModels.map { File(modelDirectory, it) }
        }

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
                        OnnxJavaType.INT64 -> OnnxTensor.createTensor(
                            environment, LongBuffer.wrap(LongArray(numElements)), shape,
                        )
                        else -> continue
                    }
                    dummyTensors.add(tensor)
                    feed[name] = tensor
                }

                if (feed.isNotEmpty()) {
                    session.run(feed).use { }
                }
                dummyTensors.forEach(OnnxTensor::close)
            } catch (e: Exception) {
                // Warm-up failure is non-fatal; first real inference will trigger compilation.
            }
        }
    }
}
