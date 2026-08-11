package dev.rinstel.genie_tts.inference

interface BackendRuntime : AutoCloseable {
    val backend: ExecutionBackend
    /** False when an optional provider's native libraries are not loadable. */
    val isAvailable: Boolean
        get() = true

    val runtimeLabel: String
        get() = backend.label

    fun generate(
        request: GenerationRequest,
        callbacks: BackendRuntimeCallbacks,
    ): GeneratedAudioFile

    /** Prepare the active reference and synthesis sessions while the UI is idle. */
    fun warmup(request: GenerationRequest) = Unit

    fun trimMemory() = Unit
}
