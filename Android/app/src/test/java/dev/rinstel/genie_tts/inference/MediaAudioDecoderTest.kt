package dev.rinstel.genie_tts.inference

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Test

class MediaAudioDecoderTest {
    @Test
    fun decodesPcm16AndAveragesChannelsWithoutClipping() {
        val bytes = byteArrayOf(
            0x00, 0x80.toByte(), 0xFF.toByte(), 0x7F,
            0x00, 0x00, 0x00, 0x00,
        )

        val decoded = MediaAudioDecoder.decodePcmChunk(
            buffer = ByteBuffer.wrap(bytes),
            offset = 0,
            size = bytes.size,
            channels = 2,
            encoding = PcmSampleEncoding.PCM_16,
        )

        assertArrayEquals(floatArrayOf(-1.5258789e-5f, 0f), decoded, 1e-7f)
    }

    @Test
    fun decodesFloat32WithAByteBufferOffset() {
        val payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            .putFloat(-0.25f)
            .putFloat(0.75f)
            .array()
        val bytes = byteArrayOf(0x55) + payload + byteArrayOf(0x66)

        val decoded = MediaAudioDecoder.decodePcmChunk(
            buffer = ByteBuffer.wrap(bytes),
            offset = 1,
            size = payload.size,
            channels = 1,
            encoding = PcmSampleEncoding.PCM_FLOAT,
        )

        assertArrayEquals(floatArrayOf(-0.25f, 0.75f), decoded, 0f)
    }

    @Test
    fun decodesPackedPcm24WithSignedExtension() {
        val bytes = byteArrayOf(
            0x00, 0x00, 0x80.toByte(),
            0xFF.toByte(), 0xFF.toByte(), 0x7F,
        )

        val decoded = MediaAudioDecoder.decodePcmChunk(
            buffer = ByteBuffer.wrap(bytes),
            offset = 0,
            size = bytes.size,
            channels = 1,
            encoding = PcmSampleEncoding.PCM_24_PACKED,
        )

        assertArrayEquals(floatArrayOf(-1f, 0.9999999f), decoded, 1e-7f)
    }
}
