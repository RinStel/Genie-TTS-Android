package dev.rinstel.genie_tts.inference

import ai.onnxruntime.OrtSession
import java.util.Locale

/** CPU-only session policy used by the production Android runtime. */
object ModelRoleSessionOptions {
    fun roleForModelFile(modelFileName: String): InferenceModelRole {
        val name = modelFileName.lowercase(Locale.ROOT)
        return when {
            "roberta" in name -> InferenceModelRole.ROBERTA
            "hubert" in name -> InferenceModelRole.HUBERT
            "speaker" in name || "sv_encoder" in name -> InferenceModelRole.SPEAKER_ENCODER
            "prompt_encoder" in name -> InferenceModelRole.PROMPT_ENCODER
            "vits" in name || "vocoder" in name -> InferenceModelRole.VOCODER
            "t2s" in name || "decoder" in name -> InferenceModelRole.T2S
            else -> InferenceModelRole.TEXT_FRONTEND
        }
    }

    fun providerFor(
        backend: ExecutionBackend,
        role: InferenceModelRole,
    ): InferenceProvider = InferenceProvider.CPU

    fun configure(
        options: OrtSession.SessionOptions,
        backend: ExecutionBackend,
        role: InferenceModelRole,
        qnnProviderOptions: Map<String, String> = emptyMap(),
    ) {
        OrtCpuBackend.configureCpuSessionOptions(options)
    }
}
