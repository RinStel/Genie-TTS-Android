package dev.rinstel.genie_tts.inference

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File
import java.nio.ByteBuffer

internal enum class PcmSampleEncoding(val bytesPerSample: Int) {
    PCM_8(1),
    PCM_16(2),
    PCM_24_PACKED(3),
    PCM_32(4),
    PCM_FLOAT(4),
    ;

    companion object {
        fun fromMediaFormat(format: MediaFormat): PcmSampleEncoding {
            val value = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                format.getInteger(MediaFormat.KEY_PCM_ENCODING)
            } else {
                AudioFormat.ENCODING_PCM_16BIT
            }
            return when (value) {
                AudioFormat.ENCODING_PCM_8BIT -> PCM_8
                AudioFormat.ENCODING_PCM_16BIT -> PCM_16
                AudioFormat.ENCODING_PCM_FLOAT -> PCM_FLOAT
                // These constants are not available on every Android SDK stub.
                21 -> PCM_24_PACKED // AudioFormat.ENCODING_PCM_24BIT_PACKED
                22 -> PCM_32 // AudioFormat.ENCODING_PCM_32BIT
                else -> error("Unsupported MediaCodec PCM encoding: $value")
            }
        }
    }
}

object MediaAudioDecoder {

    fun decodeMonoPcm(file: File): AudioBuffer? {
        if (!file.exists()) return null

        val extension = file.extension.lowercase()
        if (extension == "wav") return null

        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        return try {
            extractor.setDataSource(file.absolutePath)
            val audioTrackIndex = findAudioTrack(extractor) ?: return null
            extractor.selectTrack(audioTrackIndex)
            val format = extractor.getTrackFormat(audioTrackIndex)
            val sampleRate = readIntOrDefault(format, MediaFormat.KEY_SAMPLE_RATE, 0)
            var channels = readIntOrDefault(format, MediaFormat.KEY_CHANNEL_COUNT, 0)
            require(sampleRate > 0 && channels > 0) { "Invalid decoded audio format." }
            var encoding = PcmSampleEncoding.fromMediaFormat(format)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: return null

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            val samples = decodeToMonoFloat(
                extractor = extractor,
                codec = codec,
                initialChannels = channels,
                initialEncoding = encoding,
                onOutputFormatChanged = { outputFormat ->
                    channels = readIntOrDefault(outputFormat, MediaFormat.KEY_CHANNEL_COUNT, channels)
                    encoding = PcmSampleEncoding.fromMediaFormat(outputFormat)
                },
            )
            AudioBuffer(samples = samples, sampleRate = sampleRate)
        } catch (_: Exception) {
            null
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { extractor.release() }
        }
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int? {
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return i
        }
        return null
    }

    private fun decodeToMonoFloat(
        extractor: MediaExtractor,
        codec: MediaCodec,
        initialChannels: Int,
        initialEncoding: PcmSampleEncoding,
        onOutputFormatChanged: (MediaFormat) -> Unit,
    ): FloatArray {
        var samples = FloatArray(0)
        var sampleCount = 0
        var channels = initialChannels
        var encoding = initialEncoding
        var inputEosQueued = false
        val bufferInfo = MediaCodec.BufferInfo()
        val timeoutUs = 10_000L

        while (true) {
            val inputBufferIndex = codec.dequeueInputBuffer(timeoutUs)
            if (inputBufferIndex >= 0 && !inputEosQueued) {
                val inputBuffer = codec.getInputBuffer(inputBufferIndex)
                    ?: continue
                val sampleSize = extractor.readSampleData(inputBuffer, 0)
                if (sampleSize < 0) {
                    codec.queueInputBuffer(
                        inputBufferIndex,
                        0,
                        0,
                        0,
                        MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                    )
                    inputEosQueued = true
                } else {
                    codec.queueInputBuffer(inputBufferIndex, 0, sampleSize, extractor.sampleTime, 0)
                    extractor.advance()
                }
            }

            when (val outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, timeoutUs)) {
                in 0..Int.MAX_VALUE -> {
                    val outputBuffer = codec.getOutputBuffer(outputBufferIndex)
                    if (outputBuffer != null && bufferInfo.size > 0) {
                        // MediaCodec may expose a sliced buffer whose limit does not
                        // include BufferInfo.offset; use the full duplicate for bounds-safe reads.
                        val pcmBuffer = outputBuffer.duplicate().apply { clear() }
                        val chunk = decodePcmChunk(
                            buffer = pcmBuffer,
                            offset = bufferInfo.offset,
                            size = bufferInfo.size,
                            channels = channels,
                            encoding = encoding,
                        )
                        val requiredSize = sampleCount + chunk.size
                        if (requiredSize > samples.size) {
                            val expandedSize = maxOf(requiredSize, maxOf(1024, samples.size * 2))
                            samples = samples.copyOf(expandedSize)
                        }
                        chunk.copyInto(samples, destinationOffset = sampleCount)
                        sampleCount = requiredSize
                    }
                    codec.releaseOutputBuffer(outputBufferIndex, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val outputFormat = codec.outputFormat
                    onOutputFormatChanged(outputFormat)
                    channels = readIntOrDefault(outputFormat, MediaFormat.KEY_CHANNEL_COUNT, channels)
                    encoding = PcmSampleEncoding.fromMediaFormat(outputFormat)
                }
            }
        }

        return samples.copyOf(sampleCount)
    }

    /** Decode one interleaved PCM chunk and average channels without applying gain or clipping. */
    internal fun decodePcmChunk(
        buffer: ByteBuffer,
        offset: Int,
        size: Int,
        channels: Int,
        encoding: PcmSampleEncoding,
    ): FloatArray {
        require(offset >= 0 && size >= 0 && offset <= buffer.limit()) { "Invalid PCM buffer range." }
        require(offset + size <= buffer.limit()) { "PCM buffer range exceeds output buffer." }
        require(channels > 0) { "PCM channel count must be positive." }

        val bytesPerFrame = encoding.bytesPerSample * channels
        require(size % bytesPerFrame == 0) { "PCM output is not frame-aligned." }
        val frameCount = size / bytesPerFrame
        val output = FloatArray(frameCount)
        for (frame in 0 until frameCount) {
            var mixed = 0f
            for (channel in 0 until channels) {
                val sampleOffset = offset + (frame * channels + channel) * encoding.bytesPerSample
                mixed += readSample(buffer, sampleOffset, encoding)
            }
            output[frame] = mixed / channels
        }
        return output
    }

    private fun readSample(
        buffer: ByteBuffer,
        offset: Int,
        encoding: PcmSampleEncoding,
    ): Float = when (encoding) {
        PcmSampleEncoding.PCM_8 -> ((buffer.get(offset).toInt() and 0xff) - 128) / 128f
        PcmSampleEncoding.PCM_16 -> readSigned16(buffer, offset) / 32768f
        PcmSampleEncoding.PCM_24_PACKED -> readSigned24(buffer, offset) / 8_388_608f
        PcmSampleEncoding.PCM_32 -> readSigned32(buffer, offset).toFloat() / 2_147_483_648f
        PcmSampleEncoding.PCM_FLOAT -> Float.fromBits(readSigned32(buffer, offset))
    }

    private fun readSigned16(buffer: ByteBuffer, offset: Int): Int =
        ((buffer.get(offset).toInt() and 0xff) or
            ((buffer.get(offset + 1).toInt() and 0xff) shl 8)).toShort().toInt()

    private fun readSigned24(buffer: ByteBuffer, offset: Int): Int {
        val value = (buffer.get(offset).toInt() and 0xff) or
            ((buffer.get(offset + 1).toInt() and 0xff) shl 8) or
            ((buffer.get(offset + 2).toInt() and 0xff) shl 16)
        return if (value and 0x800000 != 0) value or -0x1000000 else value
    }

    private fun readSigned32(buffer: ByteBuffer, offset: Int): Int =
        (buffer.get(offset).toInt() and 0xff) or
            ((buffer.get(offset + 1).toInt() and 0xff) shl 8) or
            ((buffer.get(offset + 2).toInt() and 0xff) shl 16) or
            ((buffer.get(offset + 3).toInt() and 0xff) shl 24)

    private fun readIntOrDefault(format: MediaFormat, key: String, default: Int): Int =
        if (format.containsKey(key)) format.getInteger(key) else default
}
