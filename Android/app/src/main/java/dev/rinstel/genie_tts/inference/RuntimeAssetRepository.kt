package dev.rinstel.genie_tts.inference

import android.content.Context
import java.io.File

class RuntimeAssetRepository {
    private val filesRoot: File

    constructor(context: Context) {
        filesRoot = requireNotNull(context.getExternalFilesDir(null))
    }

    constructor(filesRoot: File) {
        this.filesRoot = filesRoot
    }

    fun runtimeRoot(): File = File(filesRoot, RUNTIME_ASSET_DIRECTORY)

    fun outputRoot(): File = File(filesRoot, OUTPUT_DIRECTORY).also { it.mkdirs() }

    fun inspect(): ModelInspectionResult {
        val root = runtimeRoot()
        val allPresent = if (root.isDirectory) {
            root.walkTopDown()
                .filter(File::isFile)
                .map { it.relativeTo(root).path.replace(File.separatorChar, '/') }
                .toSet()
        } else {
            emptySet()
        }
        val missing = requiredFiles.filterNot(allPresent::contains)

        return ModelInspectionResult(
            presentFiles = allPresent.toSet(),
            missingFiles = missing,
        )
    }

    fun prepareRuntimeWeights() {
        // FP16 HuBERT weights are expanded to FP32 initializers in memory when
        // the session is created. Do not create a second disk copy here.
    }

    fun hubertModelFile(): File =
        File(File(runtimeRoot(), "chinese-hubert-base"), "chinese-hubert-base.onnx")

    fun speakerEncoderFile(): File =
        File(runtimeRoot(), "speaker_encoder.onnx")

    companion object {
        const val RUNTIME_ASSET_DIRECTORY = "RuntimeAssets"
        const val OUTPUT_DIRECTORY = "GeneratedAudio"

        private val requiredFiles = listOf(
            "chinese-hubert-base/chinese-hubert-base.onnx",
            "chinese-hubert-base/chinese-hubert-base_weights_fp16.bin",
            "chinese-hubert-base/chinese-hubert-base_weights_fp16_manifest.json",
            "speaker_encoder.onnx",
            "roberta-wwm-ext-large-onnx/model.onnx",
            "roberta-wwm-ext-large-onnx/roberta_tokenizer/tokenizer.json",
        )
    }
}
