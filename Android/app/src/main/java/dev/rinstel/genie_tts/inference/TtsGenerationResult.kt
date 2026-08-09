package dev.rinstel.genie_tts.inference

data class TtsGenerationResult(
    val audio: FloatArray,
    val shape: LongArray,
) {
    override fun equals(other: Any?): Boolean =
        other is TtsGenerationResult && audio.contentEquals(other.audio) && shape.contentEquals(other.shape)

    override fun hashCode(): Int = 31 * audio.contentHashCode() + shape.contentHashCode()
}
