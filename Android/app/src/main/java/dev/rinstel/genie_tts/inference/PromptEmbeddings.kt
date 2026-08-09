package dev.rinstel.genie_tts.inference

data class PromptEmbeddings(
    val globalEmbedding: FloatTensorData,
    val advancedGlobalEmbedding: FloatTensorData,
)
