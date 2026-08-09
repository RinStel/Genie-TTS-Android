package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class SemanticTokenSlicingTest {
    @Test
    fun oneStepEosProducesNoSemanticTokens() {
        assertArrayEquals(
            longArrayOf(),
            SemanticTokenSlicing.completedTokens(longArrayOf(99L, 1024L), completedCount = 1),
        )
    }

    @Test
    fun maxStepsKeepsEveryGeneratedSemanticToken() {
        assertArrayEquals(
            longArrayOf(11L, 12L, 13L),
            SemanticTokenSlicing.completedTokens(longArrayOf(99L, 11L, 12L, 13L), completedCount = 3),
        )
    }

    @Test
    fun normalEosExcludesTheSpecialToken() {
        assertArrayEquals(
            longArrayOf(21L, 22L),
            SemanticTokenSlicing.completedTokens(longArrayOf(99L, 21L, 22L, 1024L), completedCount = 3),
        )
    }

    @Test
    fun terminalReplacementIsExcluded() {
        assertArrayEquals(
            longArrayOf(31L, 32L),
            SemanticTokenSlicing.completedTokens(longArrayOf(99L, 31L, 32L, 0L), completedCount = 3),
        )
    }

    @Test
    fun specialTokenBeforeBoundaryIsTruncated() {
        assertArrayEquals(
            longArrayOf(41L),
            SemanticTokenSlicing.completedTokens(longArrayOf(99L, 41L, 1024L, 0L), completedCount = 3),
        )
    }

    @Test
    fun zeroCompletedOutputsProducesNoSemanticTokens() {
        assertArrayEquals(
            longArrayOf(),
            SemanticTokenSlicing.completedTokens(longArrayOf(99L, 31L, 32L), completedCount = 0),
        )
    }
}
