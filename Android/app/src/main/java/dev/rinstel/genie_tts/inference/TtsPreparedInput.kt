package dev.rinstel.genie_tts.inference

data class LongTensorData(
    val values: LongArray,
    val shape: LongArray,
) {
    override fun equals(other: Any?): Boolean =
        other is LongTensorData && values.contentEquals(other.values) && shape.contentEquals(other.shape)

    override fun hashCode(): Int = 31 * values.contentHashCode() + shape.contentHashCode()
}

data class FloatTensorData(
    val values: FloatArray,
    val shape: LongArray,
) {
    override fun equals(other: Any?): Boolean =
        other is FloatTensorData && values.contentEquals(other.values) && shape.contentEquals(other.shape)

    override fun hashCode(): Int = 31 * values.contentHashCode() + shape.contentHashCode()
}

data class TtsPreparedInput(
    val refSeq: LongTensorData,
    val textSeq: LongTensorData,
    val refBert: FloatTensorData,
    val textBert: FloatTensorData,
    val sslContent: FloatTensorData,
    val globalEmbedding: FloatTensorData,
    val advancedGlobalEmbedding: FloatTensorData,
    val refAudio32k: FloatTensorData? = null,
)
