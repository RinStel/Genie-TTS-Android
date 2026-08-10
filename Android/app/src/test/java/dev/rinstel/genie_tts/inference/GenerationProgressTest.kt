package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class GenerationProgressTest {
    @Test
    fun decoderProgressUsesTheFullAutoregressiveRange() {
        assertEquals(GenerationProgress.DECODER_START, GenerationProgress.decoderPercent(0, 100))
        assertEquals(GenerationProgress.DECODER_END, GenerationProgress.decoderPercent(100, 100))
        assertEquals(69, GenerationProgress.decoderPercent(50, 100))
    }

    @Test
    fun decoderProgressClampsInvalidStepCounts() {
        assertEquals(GenerationProgress.DECODER_START, GenerationProgress.decoderPercent(-1, 0))
        assertEquals(GenerationProgress.DECODER_END, GenerationProgress.decoderPercent(200, 100))
    }
}
