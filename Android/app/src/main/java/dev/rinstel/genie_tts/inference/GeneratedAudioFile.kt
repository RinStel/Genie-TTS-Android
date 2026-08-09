package dev.rinstel.genie_tts.inference

import java.io.File

data class GeneratedAudioFile(
    val file: File,
    val audioSamples: Int,
    val tensorShape: LongArray,
) {
    override fun equals(other: Any?): Boolean =
        other is GeneratedAudioFile &&
            file == other.file &&
            audioSamples == other.audioSamples &&
            tensorShape.contentEquals(other.tensorShape)

    override fun hashCode(): Int = 31 * (31 * file.hashCode() + audioSamples) + tensorShape.contentHashCode()
}
