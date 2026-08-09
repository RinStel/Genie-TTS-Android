package dev.rinstel.genie_tts.inference

interface TtsInputPreparer {
    fun prepare(request: GenerationRequest): TtsPreparedInput
}
