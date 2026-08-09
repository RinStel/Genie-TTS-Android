package dev.rinstel.genie_tts.inference

interface PreparedTtsBackend {
    fun isInitialized(): Boolean

    fun generatePrepared(input: TtsPreparedInput, maxDecoderSteps: Int = 500): TtsGenerationResult
}
