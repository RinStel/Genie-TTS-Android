package dev.rinstel.genie_tts.inference

data class ModelInspectionResult(
    val presentFiles: Set<String>,
    val missingFiles: List<String>,
) {
    val isComplete: Boolean
        get() = missingFiles.isEmpty()
}
