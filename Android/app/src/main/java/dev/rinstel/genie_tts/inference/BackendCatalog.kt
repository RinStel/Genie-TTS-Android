package dev.rinstel.genie_tts.inference

object BackendCatalog {
    fun cpuOnly(): List<BackendOption> = listOf(
        BackendOption(
            backend = ExecutionBackend.CPU,
            enabled = true,
            status = "Ready",
        ),
    )

    /** Build the options supported by the selected APK flavor. */
    fun build(
        qnnAvailable: Boolean = false,
        qnnStatus: String = "Archived",
        xnnpackAvailable: Boolean = false,
        xnnpackStatus: String = "Archived",
    ): List<BackendOption> = buildList {
        addAll(cpuOnly())
        if (qnnAvailable) {
            add(
                BackendOption(
                    backend = ExecutionBackend.QNN,
                    enabled = true,
                    status = qnnStatus,
                ),
            )
        }
    }

    fun defaultOption(options: List<BackendOption>): BackendOption =
        options.firstOrNull { it.backend == ExecutionBackend.CPU && it.enabled }
            ?: cpuOnly().first()
}
