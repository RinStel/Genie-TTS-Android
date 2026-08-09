package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ReferenceAudioResamplerTest {
    @Test
    fun sameRateDoesNotRequireNativeLibraryOrCopySamples() {
        val input = floatArrayOf(0.0f, 0.5f, -0.25f)

        assertArrayEquals(input, ReferenceAudioResampler.resample(input, 16_000, 16_000), 0.0f)
    }

    @Test
    fun emptyInputPreservesShapeWithoutNativeLibrary() {
        assertArrayEquals(
            FloatArray(0),
            ReferenceAudioResampler.resample(FloatArray(0), 32_000, 16_000),
            0.0f,
        )
    }

    @Test
    fun invalidRatesAreRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            ReferenceAudioResampler.resample(floatArrayOf(1.0f), 0, 16_000)
        }
    }
}
