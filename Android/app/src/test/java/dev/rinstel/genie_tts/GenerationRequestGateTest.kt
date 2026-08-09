package dev.rinstel.genie_tts

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationRequestGateTest {
    @Test
    fun onlyOneRequestCanBeClaimedUntilTheCurrentRequestReleases() {
        val gate = GenerationRequestGate()

        assertTrue(gate.tryAcquire())
        assertTrue(gate.isClaimed())
        assertFalse(gate.tryAcquire())

        gate.release()

        assertFalse(gate.isClaimed())
        assertTrue(gate.tryAcquire())
    }
}
