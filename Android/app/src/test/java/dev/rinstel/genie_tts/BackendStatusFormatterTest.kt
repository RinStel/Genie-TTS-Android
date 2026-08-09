package dev.rinstel.genie_tts

import dev.rinstel.genie_tts.inference.ExecutionBackend
import dev.rinstel.genie_tts.inference.GenerationStage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BackendStatusFormatterTest {
    @Test
    fun formatsBusyStateWithCompactBackendSummary() {
        val state = BackendServiceState(
            stage = GenerationStage.RUNNING_INFERENCE,
            message = "Running inference",
            busy = true,
            requestedBackend = ExecutionBackend.QNN,
            resolvedBackend = ExecutionBackend.QNN,
            runtimeLabel = "ORT QNN EP",
        )

        val summary = BackendStatusFormatter.format(state, isPlaying = false)

        assertTrue(summary.contains("QNN"))
        assertTrue(summary.contains("ORT QNN EP"))
        assertTrue(summary.contains("Running inference"))
    }

    @Test
    fun showsBackendFallbackWhenRequestedAndResolvedDiffer() {
        val state = BackendServiceState(
            message = "Fallback active",
            requestedBackend = ExecutionBackend.XNNPACK,
            resolvedBackend = ExecutionBackend.CPU,
            runtimeLabel = "ORT CPU EP",
        )

        val summary = BackendStatusFormatter.format(state, isPlaying = false)

        assertTrue(summary.contains("XNNPACK -> CPU"))
        assertTrue(summary.contains("ORT CPU EP"))
        assertTrue(summary.contains("Fallback active"))
    }

    @Test
    fun progressModelUsesIndeterminateForBusyZeroPercentStages() {
        val progress = BackendStatusFormatter.progressModel(
            BackendServiceState(
                stage = GenerationStage.IDLE,
                busy = true,
            ),
        )

        assertEquals(true, progress.indeterminate)
        assertEquals(0, progress.percent)
    }

    @Test
    fun progressModelPrefersExplicitProgressOverStaticStageProgress() {
        val progress = BackendStatusFormatter.progressModel(
            BackendServiceState(
                stage = GenerationStage.RUNNING_INFERENCE,
                busy = true,
                progressPercent = 47,
                progressLabel = "Decoding 47%",
            ),
        )

        assertEquals(false, progress.indeterminate)
        assertEquals(47, progress.percent)
    }

    @Test
    fun formatIncludesCompletedGenerationDuration() {
        val summary = BackendStatusFormatter.format(
            BackendServiceState(
                stage = GenerationStage.COMPLETED,
                message = "Completed",
                generationDurationMs = 1_234L,
            ),
            isPlaying = false,
        )

        assertTrue(summary.contains("1.2s"))
    }
}
