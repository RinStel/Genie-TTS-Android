package dev.rinstel.genie_tts.inference

enum class GenerationStage(
    val progressPercent: Int,
) {
    IDLE(0),
    INSPECTING_RESOURCES(10),
    INSTALLING_MODEL(25),
    INITIALIZING_BACKEND(45),
    PREPARING_FEATURES(65),
    RUNNING_INFERENCE(82),
    WRITING_OUTPUT(95),
    COMPLETED(100),
    ERROR(0),
}
