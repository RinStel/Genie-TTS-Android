package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class OrtCpuBackendThreadPlanTest {
    @Test
    fun usesAllAvailableCoresUpToEightForIntraOpWork() {
        assertEquals(
            OrtCpuBackend.CpuThreadPlan(intraOpThreads = 8, interOpThreads = 1),
            OrtCpuBackend.cpuThreadPlan(12),
        )
    }

    @Test
    fun keepsSmallDevicesFromRequestingZeroThreads() {
        assertEquals(
            OrtCpuBackend.CpuThreadPlan(intraOpThreads = 1, interOpThreads = 1),
            OrtCpuBackend.cpuThreadPlan(0),
        )
    }
}
