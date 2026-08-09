package dev.rinstel.genie_tts.inference

import java.io.File
import java.io.RandomAccessFile
import kotlin.math.min

object WavAudioReader {
    fun readMonoPcm(file: File): AudioBuffer {
        RandomAccessFile(file, "r").use { input ->
            require(input.length() >= RIFF_HEADER_SIZE) {
                "WAV file is too small: ${file.absolutePath}"
            }
            require(readAscii(input, 4) == "RIFF") {
                "Unsupported audio file: RIFF header missing."
            }
            input.skipBytes(4)
            require(readAscii(input, 4) == "WAVE") {
                "Unsupported audio file: WAVE header missing."
            }

            var audioFormat = 0
            var channels = 0
            var sampleRate = 0
            var bitsPerSample = 0
            var dataOffset = -1L
            var dataSize = 0L

            // Parse chunk headers without loading the audio payload into a ByteArray.
            while (input.filePointer + CHUNK_HEADER_SIZE <= input.length()) {
                val chunkId = readAscii(input, 4)
                val chunkSize = readUnsignedIntLe(input)
                val chunkDataOffset = input.filePointer
                require(chunkSize <= input.length() - chunkDataOffset) {
                    "WAV chunk exceeds file bounds."
                }
                when (chunkId) {
                    "fmt " -> {
                        require(chunkSize >= FMT_CHUNK_SIZE) { "WAV fmt chunk is too small." }
                        val format = ByteArray(FMT_CHUNK_SIZE)
                        input.readFully(format)
                        audioFormat = readUnsignedShortLe(format, 0)
                        channels = readUnsignedShortLe(format, 2)
                        sampleRate = readIntLe(format, 4)
                        bitsPerSample = readUnsignedShortLe(format, 14)
                    }
                    "data" -> {
                        dataOffset = chunkDataOffset
                        dataSize = chunkSize
                    }
                }
                input.seek(chunkDataOffset + chunkSize + (chunkSize and 1L))
            }

            require(dataOffset >= 0L) { "WAV data chunk not found." }
            require(channels > 0) { "Invalid WAV channel count." }
            require(sampleRate > 0) { "Invalid WAV sample rate." }
            val samples = when {
                audioFormat == 1 && bitsPerSample == 16 ->
                    readPcm16(input, dataOffset, dataSize, channels)
                audioFormat == 3 && bitsPerSample == 32 ->
                    readFloat32(input, dataOffset, dataSize, channels)
                else -> error("Unsupported WAV format: format=$audioFormat bits=$bitsPerSample")
            }
            return AudioBuffer(samples = samples, sampleRate = sampleRate)
        }
    }

    fun resample(input: FloatArray, sourceRate: Int, targetRate: Int): FloatArray =
        ReferenceAudioResampler.resample(input, sourceRate, targetRate)

    fun appendSilence(input: FloatArray, sampleRate: Int, seconds: Float): FloatArray =
        input + FloatArray((sampleRate * seconds).toInt())

    private fun readPcm16(
        input: RandomAccessFile,
        offset: Long,
        size: Long,
        channels: Int,
    ): FloatArray {
        val bytesPerFrame = 2 * channels
        require(size % bytesPerFrame == 0L) { "WAV PCM16 data is not frame-aligned." }
        val frameCount = requireFrameCount(size / bytesPerFrame)
        val output = FloatArray(frameCount)
        val framesPerChunk = maxOf(1, CHUNK_BYTES / bytesPerFrame)
        val chunk = ByteArray(framesPerChunk * bytesPerFrame)
        input.seek(offset)
        var frame = 0
        while (frame < frameCount) {
            val framesToRead = min(framesPerChunk, frameCount - frame)
            input.readFully(chunk, 0, framesToRead * bytesPerFrame)
            for (chunkFrame in 0 until framesToRead) {
                var mixed = 0f
                for (channel in 0 until channels) {
                    val sampleOffset = (chunkFrame * channels + channel) * 2
                    mixed += readShortSignedLe(chunk, sampleOffset) / 32768f
                }
                output[frame + chunkFrame] = mixed / channels
            }
            frame += framesToRead
        }
        return output
    }

    private fun readFloat32(
        input: RandomAccessFile,
        offset: Long,
        size: Long,
        channels: Int,
    ): FloatArray {
        val bytesPerFrame = 4 * channels
        require(size % bytesPerFrame == 0L) { "WAV float32 data is not frame-aligned." }
        val frameCount = requireFrameCount(size / bytesPerFrame)
        val output = FloatArray(frameCount)
        val framesPerChunk = maxOf(1, CHUNK_BYTES / bytesPerFrame)
        val chunk = ByteArray(framesPerChunk * bytesPerFrame)
        input.seek(offset)
        var frame = 0
        while (frame < frameCount) {
            val framesToRead = min(framesPerChunk, frameCount - frame)
            input.readFully(chunk, 0, framesToRead * bytesPerFrame)
            for (chunkFrame in 0 until framesToRead) {
                var mixed = 0f
                for (channel in 0 until channels) {
                    val sampleOffset = (chunkFrame * channels + channel) * 4
                    mixed += readFloatLe(chunk, sampleOffset)
                }
                output[frame + chunkFrame] = mixed / channels
            }
            frame += framesToRead
        }
        return output
    }

    private fun readAscii(input: RandomAccessFile, size: Int): String {
        val bytes = ByteArray(size)
        input.readFully(bytes)
        return String(bytes, Charsets.US_ASCII)
    }

    private fun readUnsignedIntLe(input: RandomAccessFile): Long {
        val bytes = ByteArray(4)
        input.readFully(bytes)
        return readIntLe(bytes, 0).toLong() and 0xffff_ffffL
    }

    private fun readUnsignedShortLe(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or ((bytes[offset + 1].toInt() and 0xff) shl 8)

    private fun readIntLe(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or
            ((bytes[offset + 1].toInt() and 0xff) shl 8) or
            ((bytes[offset + 2].toInt() and 0xff) shl 16) or
            ((bytes[offset + 3].toInt() and 0xff) shl 24)

    private fun readShortSignedLe(bytes: ByteArray, offset: Int): Short =
        readUnsignedShortLe(bytes, offset).toShort()

    private fun readFloatLe(bytes: ByteArray, offset: Int): Float =
        Float.fromBits(readIntLe(bytes, offset))

    private fun requireFrameCount(value: Long): Int {
        require(value in 0..Int.MAX_VALUE) { "WAV audio payload is too large." }
        return value.toInt()
    }

    private const val RIFF_HEADER_SIZE = 12L
    private const val CHUNK_HEADER_SIZE = 8L
    private const val FMT_CHUNK_SIZE = 16
    private const val CHUNK_BYTES = 64 * 1024
}

data class AudioBuffer(
    val samples: FloatArray,
    val sampleRate: Int,
) {
    override fun equals(other: Any?): Boolean =
        other is AudioBuffer && samples.contentEquals(other.samples) && sampleRate == other.sampleRate

    override fun hashCode(): Int = 31 * samples.contentHashCode() + sampleRate
}
