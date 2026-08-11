package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class ModelRoleSessionOptionsTest {
    @Test
    fun referenceModelsStayOnCpuWhenQnnIsRequested() {
        assertEquals(
            InferenceModelRole.ROBERTA,
            ModelRoleSessionOptions.roleForModelFile("roberta/model.onnx"),
        )
        assertEquals(
            InferenceModelRole.HUBERT,
            ModelRoleSessionOptions.roleForModelFile("hubert_fp32.onnx"),
        )
        assertEquals(
            InferenceModelRole.SPEAKER_ENCODER,
            ModelRoleSessionOptions.roleForModelFile("speaker_encoder_fp32.onnx"),
        )
        assertEquals(
            InferenceModelRole.PROMPT_ENCODER,
            ModelRoleSessionOptions.roleForModelFile("prompt_encoder_fp32.onnx"),
        )

        listOf(
            InferenceModelRole.ROBERTA,
            InferenceModelRole.HUBERT,
            InferenceModelRole.SPEAKER_ENCODER,
            InferenceModelRole.PROMPT_ENCODER,
        ).forEach { role ->
            assertEquals(
                InferenceProvider.CPU,
                ModelRoleSessionOptions.providerFor(ExecutionBackend.QNN, role),
            )
        }
    }

    @Test
    fun synthesisRolesUseQnnWhenQnnIsRequested() {
        assertEquals(
            InferenceModelRole.T2S,
            ModelRoleSessionOptions.roleForModelFile("t2s_stage_decoder_fp32.onnx"),
        )
        assertEquals(
            InferenceModelRole.VOCODER,
            ModelRoleSessionOptions.roleForModelFile("vits_fp32.onnx"),
        )
        assertEquals(
            InferenceProvider.QNN,
            ModelRoleSessionOptions.providerFor(ExecutionBackend.QNN, InferenceModelRole.T2S),
        )
        assertEquals(
            InferenceProvider.QNN,
            ModelRoleSessionOptions.providerFor(ExecutionBackend.QNN, InferenceModelRole.VOCODER),
        )
        assertEquals(
            InferenceProvider.CPU,
            ModelRoleSessionOptions.providerFor(ExecutionBackend.XNNPACK, InferenceModelRole.T2S),
        )
    }
}
