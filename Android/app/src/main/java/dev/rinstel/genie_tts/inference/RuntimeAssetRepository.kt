package dev.rinstel.genie_tts.inference

import android.content.Context
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

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

    @Synchronized
    fun prepareRuntimeWeights() {
        val source = hubertFp16WeightsFile()
        val target = hubertFp32WeightsFile()
        require(source.isFile) { "Missing FP16 HuBERT weights: ${source.absolutePath}" }

        // ORT can load the model's regular FP32 external data directly from
        // disk. Materialize it with bounded memory instead of retaining large
        // Kotlin FloatArrays and OnnxTensors on the 256 MiB app heap.
        if (target.isFile && target.length() == source.length() * 2L) return

        val temporary = File(target.parentFile, ".${target.name}.${System.nanoTime()}.tmp")
        try {
            HalfPrecisionFileConverter.convertFp16FileToFp32File(source, temporary)
            require(temporary.length() == source.length() * 2L) {
                "Invalid converted HuBERT weight length."
            }
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
        } finally {
            temporary.delete()
        }
    }

    /**
     * Remove only derived runtime data and generated WAV files. Source runtime
     * assets remain intact and can be used to rebuild the derived weight file.
     */
    @Synchronized
    fun clearCaches(): CacheClearResult {
        val generatedAudioFiles = outputRoot()
            .listFiles()
            .orEmpty()
            .count { file ->
                file.isFile &&
                    file.extension.equals("wav", ignoreCase = true) &&
                    file.delete()
            }
        val derivedRuntimeFiles = listOf(hubertFp32WeightsFile())
            .count { file -> file.isFile && file.delete() }
        return CacheClearResult(
            generatedAudioFiles = generatedAudioFiles,
            derivedRuntimeFiles = derivedRuntimeFiles,
        )
    }

    fun hubertModelFile(): File =
        File(File(runtimeRoot(), "chinese-hubert-base"), "chinese-hubert-base.onnx")

    fun speakerEncoderFile(): File =
        File(runtimeRoot(), "speaker_encoder.onnx")

    private fun hubertFp16WeightsFile(): File =
        File(File(runtimeRoot(), "chinese-hubert-base"), "chinese-hubert-base_weights_fp16.bin")

    private fun hubertFp32WeightsFile(): File =
        File(File(runtimeRoot(), "chinese-hubert-base"), "chinese-hubert-base_weights.bin")

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

data class CacheClearResult(
    val generatedAudioFiles: Int,
    val derivedRuntimeFiles: Int,
)
