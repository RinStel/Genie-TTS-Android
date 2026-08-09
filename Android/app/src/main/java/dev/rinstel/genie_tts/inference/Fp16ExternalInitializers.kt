package dev.rinstel.genie_tts.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OrtEnvironment
import java.io.File
import java.nio.FloatBuffer
import java.util.LinkedHashMap
import org.json.JSONObject

/** Loads FP16 external weights as FP32 initializers without a disk copy. */
object Fp16ExternalInitializers {
    const val MANIFEST_FILE = "prompt_encoder_fp16_manifest.json"
    const val HUBERT_MANIFEST_FILE = "chinese-hubert-base_weights_fp16_manifest.json"
    private const val DEFAULT_WEIGHT_FILE = "prompt_encoder_fp16.bin"
    private const val MANIFEST_VERSION = 1
    private const val FLOAT32_ONNX_TYPE = 1

    data class Loaded(
        val values: Map<String, OnnxTensorLike>,
        val tensors: List<OnnxTensor>,
    ) : AutoCloseable {
        override fun close() {
            tensors.forEach(OnnxTensor::close)
        }
    }

    fun loadOrNull(
        environment: OrtEnvironment,
        modelDirectory: File,
        manifestFileName: String = MANIFEST_FILE,
        defaultWeightFile: String = DEFAULT_WEIGHT_FILE,
    ): Loaded? {
        val manifestFile = File(modelDirectory, manifestFileName)
        if (!manifestFile.isFile) return null
        // A present manifest is an explicit opt-in. Surface malformed data
        // instead of silently falling back to a missing FP32 external file.
        return load(
            environment,
            modelDirectory,
            JSONObject(manifestFile.readText()),
            defaultWeightFile,
        )
    }

    private fun load(
        environment: OrtEnvironment,
        modelDirectory: File,
        manifest: JSONObject,
        defaultWeightFile: String,
    ): Loaded {
        require(manifest.optInt("version", 0) == MANIFEST_VERSION) {
            "Unsupported FP16 external manifest version."
        }
        require(manifest.optString("logical_data_type") == "float32") {
            "FP16 external manifest must describe FP32 logical tensors."
        }
        val weightFile = File(
            modelDirectory,
            manifest.optString("weight_file", defaultWeightFile),
        )
        require(weightFile.isFile) { "FP16 external weights are missing." }
        require(weightFile.length() % 2L == 0L) { "Invalid FP16 external weight length." }

        val entries = requireNotNull(manifest.optJSONArray("initializers")) {
            "FP16 external manifest has no initializers."
        }
        val tensors = mutableListOf<OnnxTensor>()
        val initializerValues = LinkedHashMap<String, OnnxTensorLike>()
        return try {
            weightFile.inputStream().use { rawInput ->
                val input = java.io.BufferedInputStream(rawInput)
                var currentFp16Offset = 0L
                for (index in 0 until entries.length()) {
                    val entry = entries.getJSONObject(index)
                    val name = entry.getString("name")
                    require(entry.optInt("data_type", 0) == FLOAT32_ONNX_TYPE) {
                        "Unsupported FP16 external initializer data type."
                    }
                    val offset = entry.getLong("offset")
                    val length = entry.getLong("length")
                    require(offset >= 0L && length > 0L && offset % 4L == 0L && length % 4L == 0L) {
                        "Invalid FP16 external initializer byte range."
                    }
                    require(offset / 2L + length / 2L <= weightFile.length()) {
                        "FP16 external initializer exceeds weight file."
                    }
                    val shape = entry.getJSONArray("shape").let { dimensions ->
                        LongArray(dimensions.length()) { dimensions.getLong(it) }
                    }
                    val elementCount = shape.fold(1L) { total, dimension ->
                        require(dimension >= 0L) { "Invalid FP16 external initializer shape." }
                        Math.multiplyExact(total, dimension)
                    }
                    require(elementCount == length / 4L && elementCount <= Int.MAX_VALUE) {
                        "FP16 external initializer shape does not match its byte range."
                    }

                    val fp16Offset = offset / 2L
                    require(fp16Offset >= currentFp16Offset) {
                        "FP16 external initializers are not ordered by offset."
                    }
                    skipFully(input, fp16Offset - currentFp16Offset)
                    val floatValues = FloatArray(elementCount.toInt())
                    for (valueIndex in floatValues.indices) {
                        val low = input.read()
                        val high = input.read()
                        require(low >= 0 && high >= 0) { "Unexpected end of FP16 external weight file." }
                        floatValues[valueIndex] = HalfPrecision.fp16BitsToFloat(low or (high shl 8))
                    }
                    currentFp16Offset = fp16Offset + length / 2L
                    val tensor = OnnxTensor.createTensor(
                        environment,
                        FloatBuffer.wrap(floatValues),
                        shape,
                    )
                    tensors += tensor
                    initializerValues[name] = tensor
                }
            }
            require(tensors.isNotEmpty()) { "FP16 external manifest has no usable initializers." }
            Loaded(initializerValues, tensors)
        } catch (error: Throwable) {
            tensors.forEach(OnnxTensor::close)
            throw error
        }
    }

    private fun skipFully(input: java.io.InputStream, byteCount: Long) {
        var remaining = byteCount
        while (remaining > 0L) {
            val skipped = input.skip(remaining)
            if (skipped > 0L) {
                remaining -= skipped
                continue
            }
            require(input.read() >= 0) { "Unexpected end of FP16 external weight file." }
            remaining -= 1L
        }
    }
}
