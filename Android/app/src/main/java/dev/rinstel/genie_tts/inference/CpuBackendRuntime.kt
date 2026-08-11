package dev.rinstel.genie_tts.inference

class OrtSessionRuntime(
    private val modelRepository: ModelAssetRepository,
    private val runtimeRepository: RuntimeAssetRepository,
    private val backendEngine: OrtCpuBackend = OrtCpuBackend(
        backend = ExecutionBackend.CPU,
        configureSessionOptions = OrtCpuBackend::configureCpuSessionOptions,
        configureRoleSessionOptions = OrtCpuBackend::configureCpuSessionOptionsForRole,
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
        val timer = InferenceTimer(traceLogger)
        callbacks.onStage(
            GenerationStage.INSPECTING_RESOURCES,
            "inspecting",
            GenerationProgress.RESOURCE_INSPECTION,
            "Inspecting resources",
        )
        val modelInspection = timer.measure("model_inspection_ms") {
            modelRepository.inspect(request.characterModel.modelFiles)
        }
        require(modelInspection.isComplete) {
            "Missing model files: ${modelInspection.missingFiles.joinToString(", ")}"
        }

        val runtimeInspection = timer.measure("runtime_inspection_ms") {
            runtimeRepository.inspect()
        }
        require(runtimeInspection.isComplete) {
            "Missing runtime files: ${runtimeInspection.missingFiles.joinToString(", ")}"
        }

        if (!backendEngine.isInitialized() || initializedModelId != request.characterModel.id) {
            callbacks.onStage(
                GenerationStage.INSTALLING_MODEL,
                "installing_model",
                GenerationProgress.MODEL_INSTALL,
                "Preparing model",
            )
            timer.measure("model_install_ms") {
                modelRepository.installToPrivateStorage(request.characterModel.modelFiles)
            }

            callbacks.onStage(
                GenerationStage.INITIALIZING_BACKEND,
                "initializing_backend",
                GenerationProgress.BACKEND_INITIALIZATION,
                "Initializing backend",
            )
            timer.measure("backend_initialization_ms") {
                backendEngine.initialize(modelRepository.modelRootDirectory(), request.characterModel)
            }
            initializedModelId = request.characterModel.id
        }

        if (!runtimePrepared) {
            callbacks.onStage(
                GenerationStage.PREPARING_FEATURES,
                "preparing_runtime",
                GenerationProgress.FEATURE_PREPARATION_START,
                "Preparing runtime",
            )
            timer.measure("runtime_weights_ms") {
                runtimeRepository.prepareRuntimeWeights()
            }
            runtimePrepared = true
        } else {
            callbacks.onStage(
                GenerationStage.PREPARING_FEATURES,
                "preparing_features",
                GenerationProgress.FEATURE_PREPARATION_START,
                "Preparing features",
            )
        }

        val featureExtractor = runtimeFeatureExtractor ?: featureExtractorFactory(backendEngine).also {
            runtimeFeatureExtractor = it
        }

        val result = try {
            backendEngine.runtimeCallbacks = callbacks
            GenerationPipeline(
                backend = backendEngine,
                inputPreparer = featureExtractor,
                outputDirectory = runtimeRepository.outputRoot(),
                traceLogger = traceLogger,
                executionBackend = backendEngine.backend,
                runtimeLabel = runtimeLabel,
                progressCallbacks = callbacks,
            ).generate(request)
        } finally {
            backendEngine.runtimeCallbacks = null
        }

        return result
    }

    override fun warmup(request: GenerationRequest) {
        val timer = InferenceTimer(traceLogger)
        val characterModel = request.characterModel
        val modelInspection = timer.measure("model_inspection_ms") {
            modelRepository.inspect(characterModel.modelFiles)
        }
        require(modelInspection.isComplete) {
            "Missing model files: ${modelInspection.missingFiles.joinToString(", ")}"
        }
        val runtimeInspection = timer.measure("runtime_inspection_ms") {
            runtimeRepository.inspect()
        }
        require(runtimeInspection.isComplete) {
            "Missing runtime files: ${runtimeInspection.missingFiles.joinToString(", ")}"
        }

        if (!backendEngine.isInitialized() || initializedModelId != characterModel.id) {
            timer.measure("model_install_ms") {
                modelRepository.installToPrivateStorage(characterModel.modelFiles)
            }
            timer.measure("backend_initialization_ms") {
                backendEngine.initialize(modelRepository.modelRootDirectory(), characterModel)
            }
            initializedModelId = characterModel.id
        }
        if (!runtimePrepared) {
            timer.measure("runtime_weights_ms") {
                runtimeRepository.prepareRuntimeWeights()
            }
            runtimePrepared = true
        }

        val featureExtractor = runtimeFeatureExtractor ?: featureExtractorFactory(backendEngine).also {
            runtimeFeatureExtractor = it
        }
        timer.measure("prepare_total_ms") {
            featureExtractor.warmup(request)
        }
        timer.measure("t2s_session_preload_ms") {
            backendEngine.preloadT2SSessions()
        }
        // Move the largest session-construction cost into background warmup so
        // the first user request can start with a hot VITS session.
        timer.measure("vocoder_session_ms") {
            backendEngine.preloadVocoderSession()
        }
    }

    override fun close() {
        runtimeFeatureExtractor?.close()
        runtimeFeatureExtractor = null
        backendEngine.close()
        initializedModelId = null
        runtimePrepared = false
    }

    override fun trimMemory() {
        runtimeFeatureExtractor?.releaseLoadedSessions()
        backendEngine.setRetainT2SSessionsAfterGeneration(false)
        backendEngine.releaseLoadedSessions()
    }
}
