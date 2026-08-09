package dev.rinstel.genie_tts.inference

interface InferenceBackend : AutoCloseable {
    val backend: ExecutionBackend

    fun initialize(modelRoot: java.io.File, characterModel: CharacterModel): SessionSummary
}
