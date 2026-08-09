package dev.rinstel.genie_tts.inference

interface BackendRuntime : AutoCloseable {
    val backend: ExecutionBackend
    val runtimeLabel: String
        get() = backend.label

    fun generate(
        request: GenerationRequest,
        callbacks: BackendRuntimeCallbacks,
    ): GeneratedAudioFile
}
