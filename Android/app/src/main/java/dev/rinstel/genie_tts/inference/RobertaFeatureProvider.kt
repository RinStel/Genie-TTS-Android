package dev.rinstel.genie_tts.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

interface BertFeatureProvider {
    fun computeBertFeatures(
        normalizedText: String,
        word2ph: List<Int>,
        numPhones: Int,
    ): FloatTensorData
}

class RobertaFeatureProvider(
    private val environment: OrtEnvironment,
    private val modelFile: File,
    private val tokenizerFile: File,
    private val configureSessionOptions: (OrtSession.SessionOptions) -> Unit = {},
    private val onSessionCreated: () -> Unit = {},
) : BertFeatureProvider, AutoCloseable {

    private val tokenizer: ChineseBertTokenizer by lazy { ChineseBertTokenizer(tokenizerFile) }
    private var session: OrtSession? = null

    private fun ensureSession(): OrtSession {
        session?.let { return it }
        val options = OrtSession.SessionOptions()
        configureSessionOptions(options)
        return try {
            environment.createSession(modelFile.absolutePath, options).also {
                session = it
                onSessionCreated()
            }
        } finally {
            options.close()
        }
    }

    /** Open the ORT graph without running text so the first request is cheap. */
    fun warmupSession() {
        ensureSession()
    }

    override fun computeBertFeatures(
        normalizedText: String,
        word2ph: List<Int>,
        numPhones: Int,
    ): FloatTensorData {
        return try {
            val sess = ensureSession()
            val encoded = tokenizer.encode(normalizedText)
            val inputNames = sess.inputNames.toList()
            require(inputNames.any { it in setOf("input_ids", "attention_mask", "token_type_ids", "repeats") }) {
                "RoBERTa model exposes no supported inputs."
            }

            val inputs = mutableMapOf<String, OnnxTensor>()
            if ("input_ids" in inputNames) {
                inputs["input_ids"] = OnnxTensor.createTensor(
                    environment, LongBuffer.wrap(encoded.inputIds), longArrayOf(1L, encoded.inputIds.size.toLong())
                )
            }
            if ("attention_mask" in inputNames) {
                inputs["attention_mask"] = OnnxTensor.createTensor(
                    environment, LongBuffer.wrap(encoded.attentionMask), longArrayOf(1L, encoded.attentionMask.size.toLong())
                )
            }
            if ("token_type_ids" in inputNames) {
                inputs["token_type_ids"] = OnnxTensor.createTensor(
                    environment, LongBuffer.wrap(LongArray(encoded.inputIds.size)), longArrayOf(1L, encoded.inputIds.size.toLong())
                )
            }
            if ("repeats" in inputNames) {
                inputs["repeats"] = OnnxTensor.createTensor(
                    environment, LongBuffer.wrap(word2ph.map(Int::toLong).toLongArray()), longArrayOf(word2ph.size.toLong())
                )
            }

            val ownedTensors = inputs.values.toList()
            try {
                sess.run(inputs).use { result ->
                    val outputTensor = result.get(0) as OnnxTensor
                    val outputValues = FloatArray(outputTensor.info.numElements.toInt())
                    outputTensor.floatBuffer.get(outputValues)
                    val outputShape = outputTensor.info.shape

                    val expanded = expandRobertaOutput(
                        outputValues, outputShape, word2ph, numPhones
                    ) ?: throw RobertaFeatureException(
                        "output_shape_mismatch",
                        "RoBERTa output cannot be expanded to $numPhones phones.",
                    )
                    FloatTensorData(expanded, longArrayOf(numPhones.toLong(), BERT_FEATURE_DIM.toLong()))
                }
            } finally {
                ownedTensors.forEach(OnnxTensor::close)
            }
        } catch (e: RobertaFeatureException) {
            throw e
        } catch (e: Exception) {
            throw RobertaFeatureException("execution_failed", "RoBERTa inference failed.", e)
        }
    }

    private fun expandRobertaOutput(
        values: FloatArray,
        shape: LongArray,
        word2ph: List<Int>,
        numPhones: Int,
    ): FloatArray? {
        val rows = if (shape.size >= 2) shape[shape.lastIndex - 1].toInt() else {
            if (shape.size == 1) shape[0].toInt() else return null
        }
        val cols = if (shape.size >= 2) shape[shape.lastIndex].toInt() else {
            values.size / rows
        }
        if (cols != BERT_FEATURE_DIM || rows <= 0) return null
        val totalElements = rows * cols
        if (values.size < totalElements) return null

        if (rows == numPhones) {
            return values.copyOf(totalElements)
        }

        val tokenRows = if (rows == word2ph.size + 2) {
            val start = cols
            val end = start + word2ph.size * cols
            if (end > values.size) return null
            val stripped = FloatArray(word2ph.size * cols)
            System.arraycopy(values, start, stripped, 0, stripped.size)
            stripped
        } else if (rows == word2ph.size) {
            values.copyOf(totalElements)
        } else {
            return null
        }

        val expanded = FloatArray(numPhones * cols)
        var dstPos = 0
        for (i in word2ph.indices) {
            val repeatCount = word2ph[i]
            val srcPos = i * cols
            for (r in 0 until repeatCount) {
                if (dstPos + cols > expanded.size) break
                System.arraycopy(tokenRows, srcPos, expanded, dstPos, cols)
                dstPos += cols
            }
        }
        return if (dstPos == numPhones * cols) expanded else expanded.copyOf(dstPos)
    }

    override fun close() {
        session?.close()
        session = null
    }

    companion object {
        const val BERT_FEATURE_DIM = 1024

        internal data class AssetFiles(
            val modelFile: File,
            val tokenizerFile: File,
        )

        /** Accept both the Python download layout and the legacy Android layout. */
        internal fun resolveAssetFiles(runtimeRoot: File): AssetFiles? {
            val directories = listOf(
                File(runtimeRoot, "roberta"),
                File(runtimeRoot, "RoBERTa"),
                File(runtimeRoot, "roberta-wwm-ext-large-onnx"),
            )
            val modelNames = listOf("RoBERTa.onnx", "model.onnx", "model_fp16.onnx")
            val tokenizerPaths = listOf(
                "roberta_tokenizer/tokenizer.json",
                "tokenizer.json",
            )

            for (directory in directories) {
                val model = modelNames
                    .map { File(directory, it) }
                    .firstOrNull(File::isFile)
                    ?: directory.listFiles()
                        ?.filter { it.isFile && it.extension.equals("onnx", ignoreCase = true) }
                        ?.sortedBy(File::getName)
                        ?.firstOrNull()
                val tokenizer = tokenizerPaths
                    .map { File(directory, it) }
                    .firstOrNull(File::isFile)
                if (model != null && tokenizer != null) {
                    return AssetFiles(model, tokenizer)
                }
            }
            return null
        }

        fun tryCreate(
            environment: OrtEnvironment,
            runtimeRoot: File,
            configureSessionOptions: (OrtSession.SessionOptions) -> Unit = {},
            onSessionCreated: () -> Unit = {},
        ): RobertaFeatureProvider? {
            val assets = resolveAssetFiles(runtimeRoot) ?: return null
            return RobertaFeatureProvider(
                environment,
                assets.modelFile,
                assets.tokenizerFile,
                configureSessionOptions,
                onSessionCreated,
            )
        }
    }
}

class RobertaFeatureException(
    val code: String,
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

private data class TokenizerOutput(
    val inputIds: LongArray,
    val attentionMask: LongArray,
)

private class ChineseBertTokenizer(vocabFile: File) {
    private val vocab: Map<String, Int>
    private val clsId: Int
    private val sepId: Int
    private val unkId: Int
    private val modelType: String
    private val continuingSubwordPrefix: String
    private val lowercase: Boolean

    init {
        val jsonText = vocabFile.readText(Charsets.UTF_8)
        val jsonObject = JSONObject(jsonText)
        val model = jsonObject.optJSONObject("model")
        val vocabMap = model?.optJSONObject("vocab")
        modelType = model?.optString("type", "WordPiece") ?: "WordPiece"
        continuingSubwordPrefix = model?.optString("continuing_subword_prefix", "##") ?: "##"
        val normalizer = jsonObject.optJSONObject("normalizer")
        lowercase = normalizer?.optBoolean("lowercase", false) ?: false

        val tempVocab = mutableMapOf<String, Int>()
        if (vocabMap != null) {
            for (key in vocabMap.keys()) {
                tempVocab[key] = vocabMap.getInt(key)
            }
        }

        if (tempVocab.isEmpty()) {
            vocabFile.bufferedReader(Charsets.UTF_8).useLines { lines ->
                lines.forEachIndexed { index, line ->
                    val token = line.trim()
                    if (token.isNotEmpty()) {
                        tempVocab[token] = index
                    }
                }
            }
        }

        vocab = tempVocab
        clsId = vocab["[CLS]"] ?: 101
        sepId = vocab["[SEP]"] ?: 102
        unkId = vocab["[UNK]"] ?: 100
        if (modelType != "WordPiece") {
            throw RobertaFeatureException("unsupported_tokenizer", "Unsupported tokenizer model: $modelType")
        }
    }

    fun encode(text: String): TokenizerOutput {
        val ids = mutableListOf<Int>()
        ids.add(clsId)
        basicTokenize(text).forEach { token -> ids += wordPieceTokenize(token) }
        ids.add(sepId)

        val inputIds = ids.map(Int::toLong).toLongArray()
        val attentionMask = LongArray(ids.size) { 1L }
        return TokenizerOutput(inputIds, attentionMask)
    }

    private fun basicTokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val current = StringBuilder()
        fun flush() {
            if (current.isNotEmpty()) {
                tokens += current.toString()
                current.setLength(0)
            }
        }
        for (rawChar in text) {
            val char = if (lowercase) rawChar.lowercaseChar() else rawChar
            when {
                char.isWhitespace() -> flush()
                isChineseCharacter(char) || isPunctuation(char) -> {
                    flush()
                    tokens += char.toString()
                }
                else -> current.append(char)
            }
        }
        flush()
        return tokens
    }

    private fun wordPieceTokenize(token: String): List<Int> {
        vocab[token]?.let { return listOf(it) }
        if (token.isEmpty()) return emptyList()

        val pieces = mutableListOf<Int>()
        var start = 0
        while (start < token.length) {
            var end = token.length
            var matched: Int? = null
            while (end > start) {
                val piece = token.substring(start, end)
                val candidate = if (start == 0) piece else continuingSubwordPrefix + piece
                val id = vocab[candidate]
                if (id != null) {
                    matched = id
                    break
                }
                end -= 1
            }
            if (matched == null) return listOf(unkId)
            pieces += matched
            start = end
        }
        return pieces
    }

    private fun isChineseCharacter(char: Char): Boolean =
        char.code in 0x3400..0x4dbf || char.code in 0x4e00..0x9fff

    private fun isPunctuation(char: Char): Boolean =
        !char.isLetterOrDigit() && !char.isWhitespace()
}
