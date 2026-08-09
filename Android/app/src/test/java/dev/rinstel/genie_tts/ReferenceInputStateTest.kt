package dev.rinstel.genie_tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReferenceInputStateTest {
    @Test
    fun defaultSampleIsReadyWhenAudioAndTextExist() {
        val state = ReferenceInputState.defaultSample(
            backendAudioPath = "/tmp/sample.wav",
            displayAudioName = "sample.wav",
            referenceText = "hello",
            hint = "/tmp/sample.wav",
        )

        assertEquals(ReferenceInputState.Source.DEFAULT_SAMPLE, state.source)
        assertTrue(state.isReady)
    }

    @Test
    fun emptyStateIsNotReady() {
        assertFalse(ReferenceInputState.empty().isReady)
    }

    @Test
    fun withReferenceTextKeepsSourceAndAudio() {
        val state = ReferenceInputState.manualOverride(
            backendAudioPath = "/tmp/manual.wav",
            displayAudioName = "manual.wav",
            referenceText = "before",
            hint = "/tmp/manual.wav",
        )

        val updated = state.withReferenceText("after")

        assertEquals(ReferenceInputState.Source.MANUAL_OVERRIDE, updated.source)
        assertEquals("/tmp/manual.wav", updated.backendAudioPath)
        assertEquals("after", updated.referenceText)
    }

    @Test
    fun auxiliaryReferencesPreserveSelectionOrder() {
        val state = ReferenceInputState.empty().withAuxiliaryReferences(
            listOf(
                AuxiliaryReferenceAudio("/tmp/first.wav", "first.wav"),
                AuxiliaryReferenceAudio("/tmp/second.wav", "second.wav"),
            ),
        )

        assertEquals(
            listOf("first.wav", "second.wav"),
            state.auxiliaryReferences.map(AuxiliaryReferenceAudio::displayAudioName),
        )
    }
}
