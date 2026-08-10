package dev.rinstel.genie_tts.inference

enum class GenerationStage(
    val progressPercent: Int,
) {
    IDLE(GenerationProgress.RESOURCE_INSPECTION),
    INSPECTING_RESOURCES(GenerationProgress.RESOURCE_INSPECTION),
    INSTALLING_MODEL(GenerationProgress.MODEL_INSTALL),
    INITIALIZING_BACKEND(GenerationProgress.BACKEND_INITIALIZATION),
    PREPARING_FEATURES(GenerationProgress.FEATURE_PREPARATION_START),
    RUNNING_INFERENCE(GenerationProgress.T2S_ENCODER),
    WRITING_OUTPUT(GenerationProgress.WRITING_OUTPUT),
    COMPLETED(GenerationProgress.COMPLETED),
    ERROR(0),
}
