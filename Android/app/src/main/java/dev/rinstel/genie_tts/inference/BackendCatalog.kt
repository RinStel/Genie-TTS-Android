package dev.rinstel.genie_tts.inference

object BackendCatalog {
    fun cpuOnly(): List<BackendOption> = listOf(
        BackendOption(
            backend = ExecutionBackend.CPU,
            enabled = true,
            status = "Ready",
        ),
    )

    /**
     * Compatibility entry point for older callers. Accelerator probing is no
     * longer part of the production app, so every catalog is CPU-only.
     */
    fun build(
        qnnAvailable: Boolean = false,
        qnnStatus: String = "Archived",
        xnnpackAvailable: Boolean = false,
        xnnpackStatus: String = "Archived",
    ): List<BackendOption> = cpuOnly()

    fun defaultOption(options: List<BackendOption>): BackendOption =
        options.firstOrNull { it.backend == ExecutionBackend.CPU && it.enabled }
            ?: cpuOnly().first()
}
