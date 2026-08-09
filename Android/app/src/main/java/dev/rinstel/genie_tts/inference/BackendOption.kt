package dev.rinstel.genie_tts.inference

data class BackendOption(
    val backend: ExecutionBackend,
    val enabled: Boolean,
    val status: String,
) {
    val label: String
        get() = if (enabled) backend.label else "${backend.label} (不可用)"
}
