package dev.rinstel.genie_tts.inference

class GenerationPipeline(
    private val backend: PreparedTtsBackend,
    private val inputPreparer: TtsInputPreparer,
    private val outputDirectory: java.io.File,
    private val traceLogger: InferenceTraceLogger = InferenceTraceLogger.None,
    private val executionBackend: ExecutionBackend? = null,
    private val runtimeLabel: String? = null,
) {
    fun generate(request: GenerationRequest): GeneratedAudioFile {
        val timer = InferenceTimer(traceLogger)
        return timer.measure("generation_total_ms") {
            validate(request)
            executionBackend?.let { timer.event(InferenceTraceEvent.Backend(it)) }
            runtimeLabel?.let { label ->
                timer.event(InferenceTraceEvent.Runtime(InferenceRuntimeLabel.entries.first { it.wireValue == label }))
            }

            val preparedInput = timer.measure("prepare_total_ms") {
                inputPreparer.prepare(request)
            }
            val result = timer.measure("backend_total_ms") {
                backend.generatePrepared(preparedInput, request.maxDecoderSteps)
            }
            timer.event(InferenceTraceEvent.TensorCount(InferenceTensorMetric.AUDIO_SAMPLES, result.audio.size.toLong()))
            timer.event(InferenceTraceEvent.TensorShape(InferenceTensorMetric.AUDIO_SHAPE, result.shape))
            timer.traceFloatTensor(InferenceTensorMetric.AUDIO_HASH, result.audio)
            val outputFile = java.io.File(
                outputDirectory,
                "${request.characterModel.id}-${System.currentTimeMillis()}.wav",
            )
            timer.measure("write_wav_ms") {
                WavFileWriter.writeMonoPcm16(outputFile, result.audio, OUTPUT_SAMPLE_RATE)
            }
            GeneratedAudioFile(
                file = outputFile,
                audioSamples = result.audio.size,
                tensorShape = result.shape,
            )
        }
    }

    private fun validate(request: GenerationRequest) {
        require(request.synthesisText.isNotBlank()) { "Synthesis text is required." }
        require(request.referenceAudioPath.isNotBlank()) { "Reference audio path is required." }
        require(request.referenceText.isNotBlank()) { "Reference text is required." }
        require(request.language.isNotBlank()) { "Language is required." }
        require(backend.isInitialized()) { "CPU backend is not initialized." }
    }

    companion object {
        const val OUTPUT_SAMPLE_RATE = 32000
    }
}
