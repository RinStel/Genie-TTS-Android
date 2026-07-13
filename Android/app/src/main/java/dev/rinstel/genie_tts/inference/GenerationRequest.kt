package dev.rinstel.genie_tts.inference

data class GenerationRequest(
    val characterModel: CharacterModel,
    val language: String,
    val promptLanguage: String = language,
    val synthesisText: String,
    val referenceAudioPath: String,
    val referenceText: String,
    val maxDecoderSteps: Int = 500,
)
