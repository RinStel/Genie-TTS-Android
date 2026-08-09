package dev.rinstel.genie_tts.inference

data class SessionSummary(
    val backend: ExecutionBackend,
    val sessionNames: List<String>,
    val inputNames: List<String>,
    val outputNames: List<String>,
)
