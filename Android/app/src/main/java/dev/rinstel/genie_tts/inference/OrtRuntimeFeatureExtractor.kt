package dev.rinstel.genie_tts.inference

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

class OrtRuntimeFeatureExtractor(
    private val context: Context,
    private val runtimeAssets: RuntimeAssetRepository,
    private val backend: OrtCpuBackend,
    private val traceLogger: InferenceTraceLogger = InferenceTraceLogger.None,
    private val referenceCache: ReferenceConditioningCache = ReferenceConditioningCache(),
) : TtsInputPreparer, AutoCloseable {
    private val environment: OrtEnvironment by lazy(LazyThreadSafetyMode.NONE) {
        OrtEnvironment.getEnvironment()
    }
    private var hubertSession: OrtSession? = null
    private var speakerSession: OrtSession? = null

    private val robertaProvider: RobertaFeatureProvider? by lazy(LazyThreadSafetyMode.NONE) {
        RobertaFeatureProvider.tryCreate(environment, runtimeAssets.runtimeRoot()) { options ->
            backend.applySessionOptions(options)
        }
    }

    private val textFeatureExtractor: TextFeatureExtractor by lazy(LazyThreadSafetyMode.NONE) {
        DefaultTextFeatureExtractor.fromAssets(context, robertaProvider)
    }

    override fun prepare(request: GenerationRequest): TtsPreparedInput {
        val timer = InferenceTimer(traceLogger)
        val inspection = runtimeAssets.inspect()
        require(inspection.isComplete) {
            "Missing runtime assets under ${runtimeAssets.runtimeRoot().absolutePath}: ${inspection.missingFiles.joinToString(", ")}"
        }

        val isV2ProPlus = backend.hasPromptEncoder()
        val normalizedPromptLanguage = LanguageNormalizer.normalize(request.promptLanguage)
        val textFeatures = timer.measure("text_features_ms") {
            textFeatureExtractor.extract(
                request.language,
                GenieTextConventions.prepareSynthesisText(request.language, request.synthesisText),
            )
        }
        val cacheKey = referenceCacheKey(request, normalizedPromptLanguage, isV2ProPlus)
        val cachedReference = referenceCache.get(cacheKey)
        val reference = if (cachedReference != null) {
            timer.event(InferenceTraceEvent.Cache(InferenceCacheStatus.HIT))
            cachedReference
        } else {
            timer.event(InferenceTraceEvent.Cache(InferenceCacheStatus.MISS))
            runtimeAssets.prepareRuntimeWeights()
            ensureSessions(needsSpeakerEncoder = isV2ProPlus)
            referenceCache.getOrPut(cacheKey) {
                computeReferenceConditioning(
                    request = request,
                    normalizedPromptLanguage = normalizedPromptLanguage,
                    isV2ProPlus = isV2ProPlus,
                    timer = timer,
                )
            }
        }

        return reference.toPreparedInput(textFeatures)
    }

    private fun ensureSessions(needsSpeakerEncoder: Boolean) {
        if (hubertSession == null) {
            val options = OrtSession.SessionOptions()
            backend.applySessionOptions(options)
            hubertSession = environment.createSession(
                runtimeAssets.hubertModelFile().absolutePath, options
            )
            OrtCpuBackend.warmUpSession(environment, hubertSession!!)
        }
        if (needsSpeakerEncoder && speakerSession == null) {
            val options = OrtSession.SessionOptions()
            backend.applySessionOptions(options)
            speakerSession = environment.createSession(
                runtimeAssets.speakerEncoderFile().absolutePath, options
            )
            OrtCpuBackend.warmUpSession(environment, speakerSession!!)
        }
    }

    private fun computeReferenceConditioning(
        request: GenerationRequest,
        normalizedPromptLanguage: String,
        isV2ProPlus: Boolean,
        timer: InferenceTimer,
    ): ReferenceConditioning {
        val refFeatures = timer.measure("reference_text_features_ms") {
            textFeatureExtractor.extract(normalizedPromptLanguage, request.referenceText)
        }
        val refAudio32k = timer.measure("reference_audio_load_ms") {
            loadReferenceAudio(File(request.referenceAudioPath), 32000)
        }
        val refAudio16k = timer.measure("reference_audio_resample_16k_ms") {
            WavAudioReader.resampleLinear(refAudio32k, 32000, 16000)
        }
        val sslContent = timer.measure("hubert_ms") {
            runHubert(refAudio16k)
        }

        return if (isV2ProPlus) {
            val svEmbedding = timer.measure("speaker_encoder_ms") {
                runSpeakerEncoder(refAudio16k)
            }
            val promptEmbeddings = timer.measure("prompt_encoder_ms") {
                backend.encodePrompt(refAudio32k, svEmbedding)
            }
            ReferenceConditioning(
                refSeq = refFeatures.phoneIds,
                refBert = refFeatures.bert,
                sslContent = sslContent,
                globalEmbedding = promptEmbeddings.globalEmbedding,
                advancedGlobalEmbedding = promptEmbeddings.advancedGlobalEmbedding,
                refAudio32k = null,
            )
        } else {
            val refAudioTensor = FloatTensorData(
                values = refAudio32k,
                shape = longArrayOf(1L, refAudio32k.size.toLong()),
            )
            val zeroEmb = FloatTensorData(FloatArray(0), longArrayOf(0L))
            ReferenceConditioning(
                refSeq = refFeatures.phoneIds,
                refBert = refFeatures.bert,
                sslContent = sslContent,
                globalEmbedding = zeroEmb,
                advancedGlobalEmbedding = zeroEmb,
                refAudio32k = refAudioTensor,
            )
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
            referenceText = request.referenceText,
            usesPromptEncoder = isV2ProPlus,
        )

    private fun canonicalReferencePath(path: String): String {
        val file = File(path)
        return runCatching { file.canonicalPath }.getOrElse { file.absolutePath }
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
        val resampled = WavAudioReader.resampleLinear(decoded.samples, decoded.sampleRate, targetRate)
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
        robertaProvider?.close()
    }
}
