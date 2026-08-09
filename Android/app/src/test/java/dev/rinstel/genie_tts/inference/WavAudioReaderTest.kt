package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class WavAudioReaderTest {
    @Test
    fun readsPcm16AcrossMultipleStreamingChunks() {
        val directory = Files.createTempDirectory("genie-wav-reader").toFile()
        val file = File(directory, "input.wav")
        val source = FloatArray(70_000) { index ->
            ((index % 101) - 50) / 50f
        }
        WavFileWriter.writeMonoPcm16(file, source, 32_000)

        val decoded = WavAudioReader.readMonoPcm(file)

        assertEquals(32_000, decoded.sampleRate)
        assertEquals(source.size, decoded.samples.size)
        assertTrue(decoded.samples.zip(source).all { (actual, expected) ->
            kotlin.math.abs(actual - expected) <= 2f / 32768f
        })
    }
}
