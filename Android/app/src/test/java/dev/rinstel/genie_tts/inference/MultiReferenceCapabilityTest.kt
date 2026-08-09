package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MultiReferenceCapabilityTest {
    @Test
    fun recognizesVersionedPromptEncoderContract() {
        val capability = MultiReferenceCapability.fromMetadata(
            metadata = mapOf(
                "genie.interface" to "multi_reference_v1",
                "genie.multi_reference.max_count" to "3",
                "genie.multi_reference.placeholder_audio_samples" to "2048",
            ),
            inputNames = setOf(
                "reference_count",
                "ref_audio_0", "sv_emb_0",
                "ref_audio_1", "sv_emb_1",
                "ref_audio_2", "sv_emb_2",
            ),
            outputNames = setOf("ge", "ge_advanced"),
        )

        assertTrue(capability.supported)
        assertEquals(3, capability.maxReferenceCount)
        assertEquals(2048, capability.placeholderAudioSamples)
        capability.validate(listOf("one.wav", "two.wav"))
        var tooManyFailed = false
        try {
            capability.validate(listOf("one.wav", "two.wav", "three.wav"))
        } catch (_: IllegalArgumentException) {
            tooManyFailed = true
        }
        assertTrue(tooManyFailed)
    }

    @Test
    fun validatesAnonymousTensorSlotsByCount() {
        val capability = MultiReferenceCapability(
            supported = true,
            maxReferenceCount = 3,
        )

        capability.validateReferenceCount(3)
        assertEquals(2048, capability.placeholderAudioSamples)
    }

    @Test
    fun legacyOrIncompleteContractIsRejected() {
        val capability = MultiReferenceCapability.fromMetadata(
            metadata = emptyMap(),
            inputNames = setOf("reference_count", "ref_audio_0", "sv_emb_0"),
            outputNames = setOf("ge", "ge_advanced"),
        )

        assertFalse(capability.supported)
        var failed = false
        try {
            capability.validate(listOf("aux.wav"))
        } catch (_: IllegalArgumentException) {
            failed = true
        }
        assertTrue(failed)
    }

    @Test
    fun rejectsBlankAndDuplicatePathsBeforeInference() {
        val capability = MultiReferenceCapability(
            supported = true,
            maxReferenceCount = 3,
        )

        var blankFailed = false
        try {
            capability.validate(listOf(""))
        } catch (_: IllegalArgumentException) {
            blankFailed = true
        }
        assertTrue(blankFailed)

        var duplicateFailed = false
        try {
            capability.validate(listOf("same.wav", "same.wav"))
        } catch (_: IllegalArgumentException) {
            duplicateFailed = true
        }
        assertTrue(duplicateFailed)
    }
}
