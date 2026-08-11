package dev.rinstel.genie_tts

import org.junit.Assert.assertNotEquals
import org.junit.Test

class SettingsThreadControlsTest {
    @Test
    fun declaresIndependentThreadControls() {
        assertNotEquals(0, R.id.t2sCpuThreadSelector)
        assertNotEquals(0, R.id.vocoderCpuThreadSelector)
        assertNotEquals(0, R.string.prototype_settings_inference_section)
        assertNotEquals(0, R.string.prototype_settings_t2s_threads_label)
        assertNotEquals(0, R.string.prototype_settings_vocoder_threads_label)
    }
}
