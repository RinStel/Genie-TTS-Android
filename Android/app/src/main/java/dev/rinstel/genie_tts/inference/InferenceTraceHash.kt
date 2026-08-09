package dev.rinstel.genie_tts.inference

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** Hashes numeric boundary tensors without logging text, audio, or tensor values. */
object InferenceTraceHash {
    fun floatSummary(values: FloatArray): FloatTensorSummary {
        val messageDigest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteBuffer.allocate(BUFFER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        var finiteCount = 0L
        var minimum = Float.POSITIVE_INFINITY
        var maximum = Float.NEGATIVE_INFINITY
        var sum = 0.0

        values.forEach { value ->
            flushIfNeeded(buffer, Float.SIZE_BYTES, messageDigest)
            buffer.putInt(java.lang.Float.floatToRawIntBits(value))
            if (value.isFinite()) {
                finiteCount += 1L
                minimum = minOf(minimum, value)
                maximum = maxOf(maximum, value)
                sum += value.toDouble()
            }
        }
        if (buffer.position() > 0) {
            messageDigest.update(buffer.array(), 0, buffer.position())
        }

        return FloatTensorSummary(
            hash = messageDigest.digest().joinToString("") { byte -> "%02x".format(byte) },
            finiteCount = finiteCount,
            nonFiniteCount = values.size.toLong() - finiteCount,
            min = minimum.takeUnless { finiteCount == 0L },
            max = maximum.takeUnless { finiteCount == 0L },
            mean = if (finiteCount == 0L) null else sum / finiteCount,
        )
    }

    fun sha256(values: LongArray): String = digest { buffer, digest ->
        values.forEach { value ->
            flushIfNeeded(buffer, Long.SIZE_BYTES, digest)
            buffer.putLong(value)
        }
    }

    fun sha256(values: FloatArray): String = digest { buffer, digest ->
        values.forEach { value ->
            flushIfNeeded(buffer, Float.SIZE_BYTES, digest)
            buffer.putInt(java.lang.Float.floatToRawIntBits(value))
        }
    }

    private fun digest(
        write: (ByteBuffer, MessageDigest) -> Unit,
    ): String {
        val messageDigest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteBuffer.allocate(BUFFER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        write(buffer, messageDigest)
        if (buffer.position() > 0) {
            messageDigest.update(buffer.array(), 0, buffer.position())
        }
        return messageDigest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private fun flushIfNeeded(
        buffer: ByteBuffer,
        requiredBytes: Int,
        digest: MessageDigest,
    ) {
        if (buffer.remaining() < requiredBytes) {
            digest.update(buffer.array(), 0, buffer.position())
            buffer.clear()
        }
    }

    private const val BUFFER_SIZE = 8192
}

data class FloatTensorSummary(
    val hash: String,
    val finiteCount: Long,
    val nonFiniteCount: Long,
    val min: Float?,
    val max: Float?,
    val mean: Double?,
)
