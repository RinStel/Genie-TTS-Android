package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class WavFileWriterTest {
    @Test
    fun writesPcm16WavFile() {
        val output = File(createTempDir(), "out.wav")

        WavFileWriter.writeMonoPcm16(output, floatArrayOf(-1f, 0f, 1f), 32000)

        val bytes = output.readBytes()
        assertEquals("RIFF", String(bytes.copyOfRange(0, 4), Charsets.US_ASCII))
        assertEquals("WAVE", String(bytes.copyOfRange(8, 12), Charsets.US_ASCII))
        assertTrue(output.length() > 44)
    }

    @Test
    fun convertsSamplesToLittleEndianPcm16Buffer() {
        val bytes = WavFileWriter.toPcm16LeBytes(floatArrayOf(-2f, -1f, 0f, 1f, 2f))

        assertEquals(
            listOf(
                0x01, 0x80,
                0x01, 0x80,
                0x00, 0x00,
                0xff, 0x7f,
                0xff, 0x7f,
            ),
            bytes.map { it.toInt() and 0xff },
        )
    }
}
