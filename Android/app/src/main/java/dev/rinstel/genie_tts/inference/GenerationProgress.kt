package dev.rinstel.genie_tts.inference

/** Progress ranges shared by the service, CPU pipeline, and decoder loop. */
object GenerationProgress {
    const val RESOURCE_INSPECTION = 0
    const val MODEL_INSTALL = 5
    const val BACKEND_INITIALIZATION = 15
    const val FEATURE_PREPARATION_START = 20
    const val TEXT_FEATURES_START = 23
    const val TEXT_FEATURES_COMPLETE = 29
    const val REFERENCE_CONDITIONING_START = 30
    const val HUBERT_COMPLETE = 36
    const val REFERENCE_CONDITIONING_COMPLETE = 40
    const val PROMPT_CONDITIONING_START = 42
    const val FEATURES_READY = 45
    const val T2S_ENCODER = 46
    const val FIRST_DECODER = 54
    const val DECODER_START = 58
    const val DECODER_END = 80
    const val VOCODER_COMPLETE = 97
    const val WRITING_OUTPUT = 98
    const val OUTPUT_READY = 99
    const val COMPLETED = 100

    fun decoderPercent(generatedSteps: Int, maxDecoderSteps: Int): Int {
        val safeMaxSteps = maxDecoderSteps.coerceAtLeast(1)
        val decoderPercent = (generatedSteps.toLong() * 100L / safeMaxSteps)
            .toInt()
            .coerceIn(0, 100)
        return (
            DECODER_START +
                (DECODER_END - DECODER_START) * decoderPercent / 100
            ).coerceIn(DECODER_START, DECODER_END)
    }
}
