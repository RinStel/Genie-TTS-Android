package dev.rinstel.genie_tts.inference

import org.junit.Test

class InferenceTimingStageTest {
    @Test
    fun allowsVocoderSessionCreationTiming() {
        InferenceTraceLogger.requireAllowedStage("vocoder_session_ms", 0L)
    }

    @Test
    fun allowsPreparationAndTensorBoundaryTimings() {
        listOf(
            "runtime_inspection_ms",
            "runtime_weights_ms",
            "reference_conditioning_ms",
            "prompt_conditioning_ms",
            "t2s_input_tensors_ms",
            "semantic_tensor_ms",
            "vocoder_input_tensors_ms",
            "vocoder_output_copy_ms",
            "session_cleanup_ms",
        ).forEach { stage ->
            InferenceTraceLogger.requireAllowedStage(stage, 0L)
        }
    }
}
