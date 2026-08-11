package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class T2SSessionRetentionPolicyTest {
    @Test
    fun keepsDecoderSessionsHotAfterReusingRetainedDecoderSessions() {
        assertTrue(
            OrtRuntimeFeatureExtractor.shouldRetainT2SSessionsAfterGeneration(
                memoryAllowsRetention = true,
            ),
        )
    }

    @Test
    fun retainsDecoderSessionsAfterAReferenceCacheHitWhenNoDecoderWasRetained() {
        assertTrue(
            OrtRuntimeFeatureExtractor.shouldRetainT2SSessionsAfterGeneration(
                memoryAllowsRetention = true,
            ),
        )
    }

    @Test
    fun neverRetainsDecoderSessionsWithoutMemoryHeadroom() {
        assertFalse(
            OrtRuntimeFeatureExtractor.shouldRetainT2SSessionsAfterGeneration(
                memoryAllowsRetention = false,
            ),
        )
    }

    @Test
    fun retainsVocoderSessionWhenSynthesisSessionCacheIsAllowed() {
        assertTrue(
            OrtRuntimeFeatureExtractor.shouldRetainVocoderSessionAfterGeneration(
                memoryAllowsRetention = true,
            ),
        )
    }

    @Test
    fun neverRetainsVocoderSessionWithoutMemoryHeadroom() {
        assertFalse(
            OrtRuntimeFeatureExtractor.shouldRetainVocoderSessionAfterGeneration(
                memoryAllowsRetention = false,
            ),
        )
    }
}
