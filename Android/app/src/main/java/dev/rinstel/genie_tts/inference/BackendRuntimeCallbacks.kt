package dev.rinstel.genie_tts.inference

interface BackendRuntimeCallbacks {
    fun onStage(stage: GenerationStage, message: String) {
        onStage(stage, message, progressPercent = null, progressLabel = null)
    }

    fun onStage(
        stage: GenerationStage,
        message: String,
        progressPercent: Int?,
        progressLabel: String?,
    )
}
