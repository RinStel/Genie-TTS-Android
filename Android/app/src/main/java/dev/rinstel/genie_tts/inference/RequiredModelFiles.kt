package dev.rinstel.genie_tts.inference

import java.io.File

data class RequiredModelFiles(
    val variantName: String,
    val assetSubdirectory: String,
    val primarySessionModel: String,
    val files: List<String>,
    val sessionModels: List<String> = listOf(primarySessionModel),
    val optionalSessionModels: List<String> = emptyList(),
    val sessionModelVariants: Map<String, String> = emptyMap(),
    val derivedFiles: List<DerivedModelFile> = emptyList(),
    val storageVersion: Int = 3,
    val maxAuxiliaryReferenceCount: Int = 0,
) {

    /** Select a graph variant without making the optimized graph mandatory. */
    fun sessionModelNames(
        modelDirectory: File,
        preferFp16: Boolean = false,
    ): List<String> =
        (sessionModels.map { modelName ->
            val variant = sessionModelVariants[modelName]
            when {
                variant == null -> modelName
                preferFp16 && File(modelDirectory, variant).isFile -> variant
                File(modelDirectory, modelName).isFile -> modelName
                File(modelDirectory, variant).isFile -> variant
                else -> modelName
            }
        } + optionalSessionModels.filter { File(modelDirectory, it).isFile }).distinct()

    fun fallbackSessionModelName(modelName: String): String? =
        sessionModelVariants.entries.firstOrNull { it.value == modelName }?.key

    companion object {
        private val fp16SynthesisVariants = mapOf(
            "t2s_first_stage_decoder_fp32.onnx" to "t2s_first_stage_decoder_fp16.onnx",
            "t2s_stage_decoder_fp32.onnx" to "t2s_stage_decoder_fp16.onnx",
            "vits_fp32.onnx" to "vits_fp16.onnx",
        )

        fun v2ProPlusCharacter(characterId: String): RequiredModelFiles = RequiredModelFiles(
            variantName = "${characterId.replaceFirstChar { it.uppercase() }} (V2ProPlus)",
            assetSubdirectory = "v2ProPlus/$characterId/tts_models",
            primarySessionModel = "t2s_encoder_fp32.onnx",
            files = listOf(
                "prompt_encoder_fp32.bin",
                "t2s_encoder_fp32.bin",
                "t2s_shared_fp32.bin",
                "vits_fp32.bin",
            ),
            sessionModels = listOf(
                "t2s_encoder_fp32.onnx",
                "t2s_first_stage_decoder_fp32.onnx",
                "t2s_stage_decoder_fp32.onnx",
                "prompt_encoder_fp32.onnx",
                "vits_fp32.onnx",
            ),
            optionalSessionModels = listOf("prompt_encoder_multi_fp32.onnx"),
            sessionModelVariants = fp16SynthesisVariants,
            derivedFiles = listOf(
                DerivedModelFile("prompt_encoder_fp16.bin", "prompt_encoder_fp32.bin"),
                DerivedModelFile("t2s_shared_fp16.bin", "t2s_shared_fp32.bin"),
                DerivedModelFile("vits_fp16.bin", "vits_fp32.bin"),
            ),
            // The graph reserves one slot for the primary reference.
            maxAuxiliaryReferenceCount = 7,
        )

        val mansui = v2ProPlusCharacter("mansui")

        fun v2Character(characterId: String): RequiredModelFiles = RequiredModelFiles(
            variantName = "${characterId.replaceFirstChar { it.uppercase() }} (V2)",
            assetSubdirectory = "v2/$characterId/tts_models",
            primarySessionModel = "t2s_encoder_fp32.onnx",
            files = listOf(
                "t2s_encoder_fp32.bin",
                "t2s_shared_fp32.bin",
                "vits_fp32.bin",
            ),
            sessionModels = listOf(
                "t2s_encoder_fp32.onnx",
                "t2s_first_stage_decoder_fp32.onnx",
                "t2s_stage_decoder_fp32.onnx",
                "vits_fp32.onnx",
            ),
            sessionModelVariants = fp16SynthesisVariants,
            derivedFiles = listOf(
                DerivedModelFile("t2s_shared_fp16.bin", "t2s_shared_fp32.bin"),
                DerivedModelFile("vits_fp16.bin", "vits_fp32.bin"),
            ),
        )

        val v2 = RequiredModelFiles(
            variantName = "V2",
            assetSubdirectory = "models/v2",
            primarySessionModel = "t2s_encoder_fp32.onnx",
            files = listOf(
                "t2s_encoder_fp32.bin",
                "t2s_shared_fp32.bin",
                "vits_fp32.bin",
            ),
            sessionModels = listOf(
                "t2s_encoder_fp32.onnx",
                "t2s_first_stage_decoder_fp32.onnx",
                "t2s_stage_decoder_fp32.onnx",
                "vits_fp32.onnx",
            ),
            sessionModelVariants = fp16SynthesisVariants,
            derivedFiles = listOf(
                DerivedModelFile("t2s_shared_fp16.bin", "t2s_shared_fp32.bin"),
                DerivedModelFile("vits_fp16.bin", "vits_fp32.bin"),
            ),
        )

        val v2ProPlus = RequiredModelFiles(
            variantName = "V2ProPlus",
            assetSubdirectory = "models/v2ProPlus",
            primarySessionModel = "vits_fp32.onnx",
            files = listOf(
                "prompt_encoder_fp32.onnx",
                "vits_fp32.onnx",
            ),
        )

        fun all(): List<RequiredModelFiles> = listOf(mansui, v2, v2ProPlus)

        fun fromVariantName(variantName: String): RequiredModelFiles =
            all().firstOrNull { it.variantName == variantName } ?: mansui
    }
}
