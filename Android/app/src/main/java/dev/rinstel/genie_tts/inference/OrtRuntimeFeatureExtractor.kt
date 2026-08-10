package dev.rinstel.genie_tts.inference

import android.app.ActivityManager
import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.util.LinkedHashMap

class OrtRuntimeFeatureExtractor(
    private val context: Context,
    private val runtimeAssets: RuntimeAssetRepository,
    private val backend: OrtCpuBackend,
    private val traceLogger: InferenceTraceLogger = InferenceTraceLogger.None,
    private val referenceCache: ReferenceConditioningCache = ReferenceConditioningCache(),
) : TtsInputPreparer, AutoCloseable {
    private data class AuxiliaryReferenceCacheKey(
        val audio: ReferenceAudioFingerprint,
        val modelInterfaceVersion: String,
        val preprocessingVersion: String,
        val backend: ExecutionBackend,
        val conditioningRole: InferenceModelRole,
    )

    private data class AuxiliaryReferenceConditioning(
        val audio32k: FloatArray,
        val speakerEmbedding: FloatTensorData,
    )

    private val environment: OrtEnvironment by lazy(LazyThreadSafetyMode.NONE) {
        OrtEnvironment.getEnvironment()
    }
    private var hubertSession: OrtSession? = null
    private var speakerSession: OrtSession? = null
    private val auxiliaryReferenceCache = object : LinkedHashMap<AuxiliaryReferenceCacheKey, AuxiliaryReferenceConditioning>(
        AUXILIARY_REFERENCE_CACHE_CAPACITY,
        0.75f,
        true,
    ) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<AuxiliaryReferenceCacheKey, AuxiliaryReferenceConditioning>,
        ): Boolean = size > AUXILIARY_REFERENCE_CACHE_CAPACITY
    }
    private val promptConditioningCache = PromptConditioningCache()

    private val robertaProvider: RobertaFeatureProvider? by lazy(LazyThreadSafetyMode.NONE) {
        RobertaFeatureProvider.tryCreate(
            environment = environment,
            runtimeRoot = runtimeAssets.runtimeRoot(),
            configureSessionOptions = { options ->
                backend.applySessionOptions(options, InferenceModelRole.ROBERTA)
            },
            onSessionCreated = {
                traceLogger.event(
                    InferenceTraceEvent.Model(
                        InferenceTraceRecord(
                            modelRole = InferenceModelRole.ROBERTA,
                            provider = ModelRoleSessionOptions.providerFor(
                                backend.backend,
                                InferenceModelRole.ROBERTA,
                            ),
                        ),
                    ),
                )
            },
        )
    }

    private val textFeatureExtractor: TextFeatureExtractor by lazy(LazyThreadSafetyMode.NONE) {
        DefaultTextFeatureExtractor.fromAssets(context, robertaProvider)
    }

    override fun prepare(request: GenerationRequest): TtsPreparedInput {
        val timer = InferenceTimer(traceLogger)
        var retainRobertaSession = false
        try {
            val inspection = runtimeAssets.inspect()
            require(inspection.isComplete) {
                "Missing runtime assets under ${runtimeAssets.runtimeRoot().absolutePath}: ${inspection.missingFiles.joinToString(", ")}"
            }

            val isV2ProPlus = backend.hasPromptEncoder()
            val auxiliaryReferencePaths = canonicalAuxiliaryReferencePaths(request.auxiliaryReferenceAudioPaths)
            if (auxiliaryReferencePaths.isNotEmpty()) {
                require(isV2ProPlus) { "Auxiliary references require a V2ProPlus prompt encoder." }
                val primaryReferencePath = canonicalReferencePath(request.referenceAudioPath)
                require(auxiliaryReferencePaths.none { it == primaryReferencePath }) {
                    "Auxiliary reference path duplicates the primary reference."
                }
            }

            val normalizedPromptLanguage = LanguageNormalizer.normalize(request.promptLanguage)
            val primaryCacheKey = referenceCacheKey(request, normalizedPromptLanguage, isV2ProPlus)
            val cachedPrimary = referenceCache.get(primaryCacheKey)
            val promptCacheKey = if (isV2ProPlus) {
                promptConditioningCacheKey(request, auxiliaryReferencePaths)
            } else {
                null
            }
            val cachedPrompt = promptCacheKey?.let(promptConditioningCache::get)
            val cacheHit = cachedPrimary != null && (!isV2ProPlus || cachedPrompt != null)
            val retainRoberta = canRetainRobertaSession()
            val retainT2S = canRetainT2SSessions()
            retainRobertaSession = retainRoberta

            // Character sessions overlap with the vocoder working set. Always
            // release them after a sentence; keep only the text frontend when
            // memory permits.
            backend.setRetainT2SSessionsAfterGeneration(retainT2S)
            if (cacheHit && retainT2S) {
                backend.releaseLoadedSessionsKeepingReusableT2S()
            } else {
                backend.releaseLoadedSessions()
            }
            closeTransientFeatureSessions(retainRoberta = retainRoberta)

            backend.reportProgress(
                GenerationStage.PREPARING_FEATURES,
                "preparing_features",
                GenerationProgress.TEXT_FEATURES_START,
                "Text features",
            )
            val textFeatures = timer.measure("text_features_ms") {
                textFeatureExtractor.extract(
                    request.language,
                    GenieTextConventions.prepareSynthesisText(request.language, request.synthesisText),
                )
            }
            timer.event(
                InferenceTraceEvent.Model(
                    InferenceTraceRecord(
                        modelRole = InferenceModelRole.TEXT_FRONTEND,
                        provider = InferenceProvider.CPU,
                    ),
                ),
            )
            backend.reportProgress(
                GenerationStage.PREPARING_FEATURES,
                "preparing_features",
                GenerationProgress.TEXT_FEATURES_COMPLETE,
                "Reference conditioning",
            )
            timer.event(
                InferenceTraceEvent.TensorHash(
                    InferenceTensorMetric.TEXT_SEQ_HASH,
                    InferenceTraceHash.sha256(textFeatures.phoneIds.values),
                ),
            )
            timer.traceFloatTensor(InferenceTensorMetric.TEXT_BERT_HASH, textFeatures.bert.values)

            // Keep semantic/reference features independent from the ordered auxiliary bundle.
            timer.event(
                InferenceTraceEvent.Cache(
                    if (cacheHit) InferenceCacheStatus.HIT else InferenceCacheStatus.MISS,
                ),
            )
            if (!cacheHit) {
                runtimeAssets.prepareRuntimeWeights()
            }
            backend.reportProgress(
                GenerationStage.PREPARING_FEATURES,
                "preparing_features",
                GenerationProgress.REFERENCE_CONDITIONING_START,
                "Reference audio",
            )
            val primary = if (cachedPrimary != null) {
                cachedPrimary
            } else {
                val refFeatures = timer.measure("reference_text_features_ms") {
                    textFeatureExtractor.extract(normalizedPromptLanguage, request.referenceText)
                }
                timer.event(
                    InferenceTraceEvent.TensorHash(
                        InferenceTensorMetric.REF_SEQ_HASH,
                        InferenceTraceHash.sha256(refFeatures.phoneIds.values),
                    ),
                )
                timer.traceFloatTensor(InferenceTensorMetric.REF_BERT_HASH, refFeatures.bert.values)
                // Keep RoBERTa warm only under the memory policy above. When
                // it is not retained, close it before the audio feature models.
                if (!retainRoberta) {
                    closeRobertaSession()
                }
                referenceCache.getOrPut(primaryCacheKey) {
                    computePrimaryReferenceConditioning(
                        request = request,
                        refFeatures = refFeatures,
                        isV2ProPlus = isV2ProPlus,
                        timer = timer,
                    )
                }
            }
            val reference = if (!isV2ProPlus) {
                primary
            } else {
                val prompt = cachedPrompt ?: promptConditioningCache.getOrPut(requireNotNull(promptCacheKey)) {
                    backend.reportProgress(
                        GenerationStage.PREPARING_FEATURES,
                        "preparing_features",
                        GenerationProgress.PROMPT_CONDITIONING_START,
                        "Prompt conditioning",
                    )
                    // Capability inspection opens the optional prompt graph.
                    if (!retainRoberta) {
                        closeRobertaSession()
                    }
                    requireNotNull(backend.multiReferenceCapability())
                        .validateAuxiliaryReferencePaths(auxiliaryReferencePaths)
                    backend.releasePreparationSessions()
                    computePromptConditioning(
                        primary = primary,
                        auxiliaryReferencePaths = auxiliaryReferencePaths,
                        modelInterfaceVersion = modelInterfaceVersion(request.characterModel),
                        timer = timer,
                    )
                }
                primary.copy(
                    globalEmbedding = prompt.globalEmbedding,
                    advancedGlobalEmbedding = prompt.advancedGlobalEmbedding,
                )
            }

            backend.reportProgress(
                GenerationStage.PREPARING_FEATURES,
                "preparing_features",
                GenerationProgress.FEATURES_READY,
                "Features ready",
            )

            // Feature preparation can allocate hundreds of megabytes. Recheck
            // the watermark after that work so the next T2S request cannot
            // retain decoder graphs on an already pressured device.
            val retainT2SAfterPreparation = retainT2S && canRetainT2SSessions()
            backend.setRetainT2SSessionsAfterGeneration(retainT2SAfterPreparation)
            if (!retainT2SAfterPreparation) {
                backend.releaseT2SSessions()
            }

            return reference.toPreparedInput(textFeatures)
        } finally {
            closeTransientFeatureSessions(retainRoberta = retainRobertaSession)
            backend.releasePreparationSessions()
        }
    }

    private fun canRetainRobertaSession(): Boolean {
        val manager = context.getSystemService(ActivityManager::class.java) ?: return false
        val memoryInfo = ActivityManager.MemoryInfo()
        manager.getMemoryInfo(memoryInfo)
        return !memoryInfo.lowMemory &&
            memoryInfo.totalMem >= MIN_ROBERTA_CACHE_TOTAL_MEMORY_BYTES &&
            memoryInfo.availMem >= MIN_ROBERTA_CACHE_AVAILABLE_MEMORY_BYTES
    }

    private fun canRetainT2SSessions(): Boolean {
        val manager = context.getSystemService(ActivityManager::class.java) ?: return false
        val memoryInfo = ActivityManager.MemoryInfo()
        manager.getMemoryInfo(memoryInfo)
        return !memoryInfo.lowMemory &&
            memoryInfo.totalMem >= MIN_T2S_CACHE_TOTAL_MEMORY_BYTES &&
            memoryInfo.availMem >= MIN_T2S_CACHE_AVAILABLE_MEMORY_BYTES
    }

    fun releaseLoadedSessions() {
        closeTransientFeatureSessions()
        backend.releasePreparationSessions()
    }

    private fun ensureHubertSession() {
        if (hubertSession == null) {
            runtimeAssets.prepareRuntimeWeights()
            val options = OrtSession.SessionOptions()
            try {
                backend.applySessionOptions(options, InferenceModelRole.HUBERT)
                hubertSession = environment.createSession(
                    runtimeAssets.hubertModelFile().absolutePath, options
                )
            } finally {
                options.close()
            }
            traceLogger.event(
                InferenceTraceEvent.Model(
                    InferenceTraceRecord(
                        modelRole = InferenceModelRole.HUBERT,
                        provider = ModelRoleSessionOptions.providerFor(
                            backend.backend,
                            InferenceModelRole.HUBERT,
                        ),
                    ),
                ),
            )
            OrtCpuBackend.warmUpSession(environment, hubertSession!!)
        }
    }

    private fun ensureSpeakerSession() {
        if (speakerSession == null) {
            val options = OrtSession.SessionOptions()
            try {
                backend.applySessionOptions(options, InferenceModelRole.SPEAKER_ENCODER)
                speakerSession = environment.createSession(
                    runtimeAssets.speakerEncoderFile().absolutePath, options
                )
            } finally {
                options.close()
            }
            traceLogger.event(
                InferenceTraceEvent.Model(
                    InferenceTraceRecord(
                        modelRole = InferenceModelRole.SPEAKER_ENCODER,
                        provider = ModelRoleSessionOptions.providerFor(
                            backend.backend,
                            InferenceModelRole.SPEAKER_ENCODER,
                        ),
                    ),
                ),
            )
            OrtCpuBackend.warmUpSession(environment, speakerSession!!)
        }
    }

    private fun closeHubertSession() {
        hubertSession?.close()
        hubertSession = null
    }

    private fun closeSpeakerSession() {
        speakerSession?.close()
        speakerSession = null
    }

    private fun closeRobertaSession() {
        robertaProvider?.close()
    }

    private fun closeTransientFeatureSessions(retainRoberta: Boolean = false) {
        closeHubertSession()
        closeSpeakerSession()
        if (!retainRoberta) {
            closeRobertaSession()
        }
    }

    private fun computePrimaryReferenceConditioning(
        request: GenerationRequest,
        refFeatures: TextFeatures,
        isV2ProPlus: Boolean,
        timer: InferenceTimer,
    ): ReferenceConditioning {
        val refAudio32k = timer.measure("reference_audio_load_ms") {
            loadReferenceAudio(File(request.referenceAudioPath), 32000)
        }
        timer.traceFloatTensor(InferenceTensorMetric.REFERENCE_AUDIO_HASH, refAudio32k)
        val refAudio16k = timer.measure("reference_audio_resample_16k_ms") {
            WavAudioReader.resample(refAudio32k, 32000, 16000)
        }
        val sslContent = timer.measure("hubert_ms") {
            ensureHubertSession()
            try {
                runHubert(refAudio16k)
            } finally {
                closeHubertSession()
            }
        }
        backend.reportProgress(
            GenerationStage.PREPARING_FEATURES,
            "preparing_features",
            GenerationProgress.HUBERT_COMPLETE,
            "HuBERT",
        )
        timer.traceFloatTensor(InferenceTensorMetric.SSL_CONTENT_HASH, sslContent.values)

        val zeroEmb = FloatTensorData(FloatArray(0), longArrayOf(0L))
        val refAudioTensor = if (isV2ProPlus) {
            null
        } else {
            FloatTensorData(
                values = refAudio32k,
                shape = longArrayOf(1L, refAudio32k.size.toLong()),
            )
        }
        val primarySpeakerEmbedding = if (isV2ProPlus) {
            timer.measure("speaker_encoder_ms") {
                ensureSpeakerSession()
                try {
                    runSpeakerEncoder(refAudio16k)
                } finally {
                    closeSpeakerSession()
                }
            }
        } else {
            null
        }
        backend.reportProgress(
            GenerationStage.PREPARING_FEATURES,
            "preparing_features",
            GenerationProgress.REFERENCE_CONDITIONING_COMPLETE,
            "Reference ready",
        )
        return ReferenceConditioning(
            refSeq = refFeatures.phoneIds,
            refBert = refFeatures.bert,
            sslContent = sslContent,
            globalEmbedding = zeroEmb,
            advancedGlobalEmbedding = zeroEmb,
            refAudio32k = refAudioTensor,
            primaryAudio32k = refAudio32k,
            primaryAudio16k = refAudio16k,
            primarySpeakerEmbedding = primarySpeakerEmbedding,
        )
    }

    private fun computePromptConditioning(
        primary: ReferenceConditioning,
        auxiliaryReferencePaths: List<String>,
        modelInterfaceVersion: String,
        timer: InferenceTimer,
    ): PromptEmbeddings {
        timer.traceFloatTensor(
            speakerEmbeddingMetric(0),
            requireNotNull(primary.primarySpeakerEmbedding).values,
        )
        val promptEmbeddings = if (auxiliaryReferencePaths.isEmpty()) {
            timer.measure("prompt_encoder_ms") {
                backend.encodePrompt(
                    requireNotNull(primary.primaryAudio32k),
                    requireNotNull(primary.primarySpeakerEmbedding),
                )
            }
        } else {
            val auxiliaryConditioning = run {
                ensureSpeakerSession()
                try {
                    auxiliaryReferencePaths.mapIndexed { index, path ->
                        loadAuxiliaryReference(path, modelInterfaceVersion).also { conditioning ->
                            timer.traceFloatTensor(
                                speakerEmbeddingMetric(index + 1),
                                conditioning.speakerEmbedding.values,
                            )
                        }
                    }
                } finally {
                    closeSpeakerSession()
                }
            }
            val referenceAudio32k = buildList {
                add(requireNotNull(primary.primaryAudio32k))
                addAll(auxiliaryConditioning.map(AuxiliaryReferenceConditioning::audio32k))
            }
            val speakerEmbeddings = buildList {
                add(requireNotNull(primary.primarySpeakerEmbedding))
                addAll(auxiliaryConditioning.map(AuxiliaryReferenceConditioning::speakerEmbedding))
            }
            timer.measure("prompt_encoder_ms") {
                backend.encodePromptBatch(
                    referenceAudio32k = referenceAudio32k,
                    speakerEmbeddings = speakerEmbeddings,
                )
            }
        }
        timer.traceFloatTensor(
            InferenceTensorMetric.GLOBAL_EMBEDDING_HASH,
            promptEmbeddings.globalEmbedding.values,
        )
        timer.traceFloatTensor(
            InferenceTensorMetric.ADVANCED_GLOBAL_EMBEDDING_HASH,
            promptEmbeddings.advancedGlobalEmbedding.values,
        )
        return promptEmbeddings
    }

    // Keep each reference slot distinguishable from the Python parity trace.
    private fun speakerEmbeddingMetric(referenceIndex: Int): InferenceTensorMetricName =
        NamedInferenceTensorMetric("speaker_embedding_${referenceIndex}_hash")

    private fun loadAuxiliaryReference(
        path: String,
        modelInterfaceVersion: String,
    ): AuxiliaryReferenceConditioning {
        val cacheKey = AuxiliaryReferenceCacheKey(
            audio = referenceAudioFingerprint(path),
            modelInterfaceVersion = modelInterfaceVersion,
            preprocessingVersion = REFERENCE_PREPROCESSING_VERSION,
            backend = backend.backend,
            conditioningRole = InferenceModelRole.SPEAKER_ENCODER,
        )
        auxiliaryReferenceCache[cacheKey]?.let { return it }
        val audio32k = loadReferenceAudio(File(cacheKey.audio.canonicalPath), 32000)
        val audio16k = WavAudioReader.resample(audio32k, 32000, 16000)
        return AuxiliaryReferenceConditioning(
            audio32k = audio32k,
            speakerEmbedding = runSpeakerEncoder(audio16k),
        ).also { conditioning ->
            auxiliaryReferenceCache[cacheKey] = conditioning
        }
    }

    private fun referenceCacheKey(
        request: GenerationRequest,
        normalizedPromptLanguage: String,
        isV2ProPlus: Boolean,
    ): ReferenceConditioningCacheKey =
        ReferenceConditioningCacheKey(
            characterModelId = request.characterModel.id,
            backend = backend.backend,
            language = normalizedPromptLanguage,
            referenceAudioPath = canonicalReferencePath(request.referenceAudioPath),
            referenceAudioSize = File(request.referenceAudioPath)
                .takeIf(File::exists)
                ?.length()
                ?: -1L,
            referenceAudioLastModified = File(request.referenceAudioPath)
                .takeIf(File::exists)
                ?.lastModified()
                ?: -1L,
            referenceText = request.referenceText,
            usesPromptEncoder = isV2ProPlus,
            modelInterfaceVersion = modelInterfaceVersion(request.characterModel),
            preprocessingVersion = REFERENCE_PREPROCESSING_VERSION,
            conditioningRole = if (isV2ProPlus) {
                InferenceModelRole.PROMPT_ENCODER
            } else {
                InferenceModelRole.VOCODER
            },
        )

    private fun promptConditioningCacheKey(
        request: GenerationRequest,
        auxiliaryReferencePaths: List<String>,
    ): PromptConditioningCacheKey =
        PromptConditioningCacheKey(
            characterModelId = request.characterModel.id,
            backend = backend.backend,
            modelInterfaceVersion = modelInterfaceVersion(request.characterModel),
            preprocessingVersion = REFERENCE_PREPROCESSING_VERSION,
            conditioningRole = InferenceModelRole.PROMPT_ENCODER,
            primaryReferenceAudio = referenceAudioFingerprint(request.referenceAudioPath),
            auxiliaryReferenceAudioFingerprints = auxiliaryReferencePaths.map(::referenceAudioFingerprint),
        )

    private fun modelInterfaceVersion(characterModel: CharacterModel): String =
        buildString {
            append(characterModel.modelFiles.variantName)
            append("|storage=")
            append(characterModel.modelFiles.storageVersion)
            append("|sessions=")
            append(characterModel.modelFiles.sessionModels.joinToString(","))
            append("|session_variants=")
            append(
                characterModel.modelFiles.sessionModelVariants.entries
                    .sortedBy { it.key }
                    .joinToString(",") { (base, variant) -> "$base->$variant" },
            )
            append("|optional_sessions=")
            append(characterModel.modelFiles.optionalSessionModels.joinToString(","))
        }

    private fun canonicalReferencePath(path: String): String {
        val file = File(path)
        return runCatching { file.canonicalPath }.getOrElse { file.absolutePath }
    }

    private fun canonicalAuxiliaryReferencePaths(paths: List<String>): List<String> {
        val canonical = paths.map(::canonicalReferencePath)
        require(canonical.none(String::isBlank)) { "Auxiliary reference paths cannot be blank." }
        require(canonical.distinct().size == canonical.size) {
            "Auxiliary reference paths must be unique."
        }
        canonical.forEach { path ->
            require(File(path).isFile) { "Auxiliary reference audio not found: $path" }
        }
        return canonical
    }

    private fun referenceAudioFingerprint(path: String): ReferenceAudioFingerprint {
        val file = File(canonicalReferencePath(path))
        return ReferenceAudioFingerprint(
            canonicalPath = file.absolutePath,
            size = if (file.isFile) file.length() else -1L,
            lastModified = if (file.isFile) file.lastModified() else -1L,
        )
    }

    private fun ReferenceConditioning.toPreparedInput(textFeatures: TextFeatures): TtsPreparedInput =
        TtsPreparedInput(
            refSeq = refSeq,
            textSeq = textFeatures.phoneIds,
            refBert = refBert,
            textBert = textFeatures.bert,
            sslContent = sslContent,
            globalEmbedding = globalEmbedding,
            advancedGlobalEmbedding = advancedGlobalEmbedding,
            refAudio32k = refAudio32k,
        )

    private fun loadReferenceAudio(file: File, targetRate: Int): FloatArray {
        val decoded = MediaAudioDecoder.decodeMonoPcm(file) ?: WavAudioReader.readMonoPcm(file)
        val resampled = WavAudioReader.resample(decoded.samples, decoded.sampleRate, targetRate)
        return WavAudioReader.appendSilence(resampled, targetRate, 0.3f)
    }

    private fun runHubert(audio16k: FloatArray): FloatTensorData {
        val input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(audio16k), longArrayOf(1L, audio16k.size.toLong()))
        input.use {
            requireNotNull(hubertSession).run(mapOf("input_values" to input)).use { result ->
                return tensorData(result.get(0) as OnnxTensor)
            }
        }
    }

    private fun runSpeakerEncoder(audio16k: FloatArray): FloatTensorData {
        val input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(audio16k), longArrayOf(1L, audio16k.size.toLong()))
        input.use {
            requireNotNull(speakerSession).run(mapOf("waveform" to input)).use { result ->
                return tensorData(result.get(0) as OnnxTensor)
            }
        }
    }

    private fun tensorData(tensor: OnnxTensor): FloatTensorData {
        val values = FloatArray(tensor.info.numElements.toInt())
        tensor.floatBuffer.get(values)
        return FloatTensorData(values, tensor.info.shape)
    }

    override fun close() {
        hubertSession?.close()
        speakerSession?.close()
        hubertSession = null
        speakerSession = null
        referenceCache.clear()
        promptConditioningCache.clear()
        auxiliaryReferenceCache.clear()
        robertaProvider?.close()
    }

    companion object {
        private const val REFERENCE_PREPROCESSING_VERSION =
            "reference-conditioning-v2-soxr-hq-silence-0.3"
        private const val AUXILIARY_REFERENCE_CACHE_CAPACITY = 8
        private const val MIN_ROBERTA_CACHE_TOTAL_MEMORY_BYTES = 8L * 1024L * 1024L * 1024L
        private const val MIN_ROBERTA_CACHE_AVAILABLE_MEMORY_BYTES = 2L * 1024L * 1024L * 1024L
        private const val MIN_T2S_CACHE_TOTAL_MEMORY_BYTES = 8L * 1024L * 1024L * 1024L
        // Keep a margin for VITS and Android services while allowing the two
        // decoder graphs to stay warm on high-memory devices.
        private const val MIN_T2S_CACHE_AVAILABLE_MEMORY_BYTES = 2400L * 1024L * 1024L
    }
}
