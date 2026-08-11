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

    @Test
    fun usesFewerIntraOpThreadsForAutoregressiveT2s() {
        assertEquals(
            OrtCpuBackend.CpuThreadPlan(intraOpThreads = 4, interOpThreads = 1),
            OrtCpuBackend.cpuThreadPlan(12, InferenceModelRole.T2S),
        )
    }

    @Test
    fun capsVocoderAutomaticModeForMobileThermalHeadroom() {
        assertEquals(
            OrtCpuBackend.CpuThreadPlan(intraOpThreads = 6, interOpThreads = 1),
            OrtCpuBackend.cpuThreadPlan(12, InferenceModelRole.VOCODER),
        )
    }

    @Test
    fun appliesIndependentFixedLimitsToT2sAndVocoder() {
        assertEquals(
            OrtCpuBackend.CpuThreadPlan(intraOpThreads = 3, interOpThreads = 1),
            OrtCpuBackend.cpuThreadPlan(
                availableProcessors = 8,
                role = InferenceModelRole.T2S,
                t2sThreadLimit = 3,
                vocoderThreadLimit = 0,
            ),
        )
        assertEquals(
            OrtCpuBackend.CpuThreadPlan(intraOpThreads = 5, interOpThreads = 1),
            OrtCpuBackend.cpuThreadPlan(
                availableProcessors = 8,
                role = InferenceModelRole.VOCODER,
                t2sThreadLimit = 0,
                vocoderThreadLimit = 5,
            ),
        )
    }

    @Test
    fun clampsFixedLimitsToAvailableProcessors() {
        assertEquals(
            OrtCpuBackend.CpuThreadPlan(intraOpThreads = 8, interOpThreads = 1),
            OrtCpuBackend.cpuThreadPlan(
                availableProcessors = 8,
                role = InferenceModelRole.VOCODER,
                t2sThreadLimit = 0,
                vocoderThreadLimit = 12,
            ),
        )
    }

    @Test
    fun avoidsNativeQnnLoopWhenDeviceMemoryIsUnavailable() {
        assertEquals(
            false,
            OrtCpuBackend.shouldUseNativeQnnDecoder(
                backend = ExecutionBackend.QNN,
                nativeDecoderAvailable = true,
                qnnDeviceMemoryAvailable = false,
            ),
        )
    }

    @Test
    fun usesNativeQnnLoopOnlyWithDeviceMemory() {
        assertEquals(
            true,
            OrtCpuBackend.shouldUseNativeQnnDecoder(
                backend = ExecutionBackend.QNN,
                nativeDecoderAvailable = true,
                qnnDeviceMemoryAvailable = true,
            ),
        )
    }

    @Test
    fun allowsNativeCpuArrayLoopWithoutQnnDeviceMemory() {
        assertEquals(
            true,
            OrtCpuBackend.shouldUseNativeDecoder(
                backend = ExecutionBackend.CPU,
                nativeDecoderAvailable = true,
                qnnDeviceMemoryAvailable = false,
            ),
        )
    }

    @Test
    fun keepsQnnNativeLoopGatedByDeviceMemory() {
        assertEquals(
            false,
            OrtCpuBackend.shouldUseNativeDecoder(
                backend = ExecutionBackend.QNN,
                nativeDecoderAvailable = true,
                qnnDeviceMemoryAvailable = false,
            ),
        )
    }

    @Test
    fun disablesMemoryPatternForVariableLengthVocoderInputs() {
        assertEquals(false, OrtCpuBackend.shouldUseMemoryPatternOptimization(InferenceModelRole.VOCODER))
        assertEquals(true, OrtCpuBackend.shouldUseMemoryPatternOptimization(InferenceModelRole.T2S))
    }

}
