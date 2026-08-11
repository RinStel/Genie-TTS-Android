package dev.rinstel.genie_tts

import org.junit.Assert.assertEquals
import org.junit.Test

class InferenceSettingsReloadActionTest {
    @Test
    fun usesADedicatedRuntimeReloadAction() {
        assertEquals(
            "dev.rinstel.genie_tts.action.RELOAD_INFERENCE_SETTINGS",
            GenieBackendService.ACTION_RELOAD_INFERENCE_SETTINGS,
        )
    }
}
