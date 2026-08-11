package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SemanticTokenStabilityTest {
    @Test
    fun derivesAConservativeMinimumFromTargetTextLength() {
        assertEquals(6, SemanticTokenStability.minimumAcceptedTokens(0))
        assertEquals(6, SemanticTokenStability.minimumAcceptedTokens(4))
        assertEquals(12, SemanticTokenStability.minimumAcceptedTokens(12))
        assertEquals(16, SemanticTokenStability.minimumAcceptedTokens(32))
    }

    @Test
    fun retriesASequenceThatIsTooShortForTheTarget() {
        assertTrue(SemanticTokenStability.shouldRetry(2, textTokenCount = 12, retryCount = 0))
        assertFalse(SemanticTokenStability.shouldRetry(12, textTokenCount = 12, retryCount = 0))
    }

    @Test
    fun stopsRetryingAfterTheBoundedRetryBudget() {
        assertFalse(SemanticTokenStability.shouldRetry(2, textTokenCount = 12, retryCount = 2))
    }
}
