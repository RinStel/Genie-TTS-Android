package dev.rinstel.genie_tts.inference

class OrtSessionRuntime(
    private val modelRepository: ModelAssetRepository,
    private val runtimeRepository: RuntimeAssetRepository,
    private val backendEngine: OrtCpuBackend = OrtCpuBackend(
        backend = ExecutionBackend.CPU,
        configureSessionOptions = OrtCpuBackend::configureCpuSessionOptions,
    ),
    private val traceLogger: InferenceTraceLogger = InferenceTraceLogger.None,
    private val featureExtractorFactory: (OrtCpuBackend) -> OrtRuntimeFeatureExtractor,
) : BackendRuntime {
    override val backend: ExecutionBackend = backendEngine.backend
    override val runtimeLabel: String =
        when (backendEngine.backend) {
            ExecutionBackend.CPU -> "ORT CPU EP"
            ExecutionBackend.QNN -> "ORT QNN EP"
            ExecutionBackend.XNNPACK -> "ORT XNNPACK EP"
        }

    private var runtimeFeatureExtractor: OrtRuntimeFeatureExtractor? = null
    private var initializedModelId: String? = null
    private var runtimePrepared = false

    override fun generate(
        request: GenerationRequest,
        callbacks: BackendRuntimeCallbacks,
    ): GeneratedAudioFile {
        callbacks.onStage(
            GenerationStage.INSPECTING_RESOURCES,
            "inspecting",
        )
        val modelInspection = modelRepository.inspect(request.characterModel.modelFiles)
        require(modelInspection.isComplete) {
            "Missing model files: ${modelInspection.missingFiles.joinToString(", ")}"
        }

        val runtimeInspection = runtimeRepository.inspect()
        require(runtimeInspection.isComplete) {
            "Missing runtime files: ${runtimeInspection.missingFiles.joinToString(", ")}"
        }

        if (!backendEngine.isInitialized() || initializedModelId != request.characterModel.id) {
            callbacks.onStage(
                GenerationStage.INSTALLING_MODEL,
                "installing_model",
            )
            modelRepository.installToPrivateStorage(request.characterModel.modelFiles)

            callbacks.onStage(
                GenerationStage.INITIALIZING_BACKEND,
                "initializing_backend",
            )
            backendEngine.initialize(modelRepository.modelRootDirectory(), request.characterModel)
            initializedModelId = request.characterModel.id
        }

        if (!runtimePrepared) {
            callbacks.onStage(
                GenerationStage.PREPARING_FEATURES,
                "preparing_runtime",
            )
            runtimeRepository.prepareRuntimeWeights()
            runtimePrepared = true
        } else {
            callbacks.onStage(
                GenerationStage.PREPARING_FEATURES,
                "preparing_features",
            )
        }

        val featureExtractor = runtimeFeatureExtractor ?: featureExtractorFactory(backendEngine).also {
            runtimeFeatureExtractor = it
        }

        callbacks.onStage(
            GenerationStage.RUNNING_INFERENCE,
            "running_inference",
        )
        val result = try {
            backendEngine.runtimeCallbacks = callbacks
            GenerationPipeline(
                backend = backendEngine,
                inputPreparer = featureExtractor,
                outputDirectory = runtimeRepository.outputRoot(),
                traceLogger = traceLogger,
                executionBackend = backendEngine.backend,
                runtimeLabel = runtimeLabel,
            ).generate(request)
        } finally {
            backendEngine.runtimeCallbacks = null
        }

        callbacks.onStage(
            GenerationStage.WRITING_OUTPUT,
            "writing_output",
        )
        return result
    }

    override fun close() {
        runtimeFeatureExtractor?.close()
        runtimeFeatureExtractor = null
        backendEngine.close()
        initializedModelId = null
        runtimePrepared = false
    }
}
