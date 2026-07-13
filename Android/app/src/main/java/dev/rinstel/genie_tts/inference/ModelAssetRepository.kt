package dev.rinstel.genie_tts.inference

import android.content.Context
import java.io.File
import org.json.JSONObject

class ModelAssetRepository private constructor(
    private val filesRoot: File,
) {
    constructor(context: Context) : this(requireNotNull(context.getExternalFilesDir(null)))

    constructor(filesRoot: File, marker: Unit = Unit) : this(filesRoot)

    private val modelRoot: File
        get() = File(filesRoot, "CharacterModels")

    fun modelRootPath(): String = modelRoot.absolutePath

    fun modelRootDirectory(): File = modelRoot

    fun discoverCharacterModels(): List<CharacterModel> {
        val result = mutableListOf<CharacterModel>()

        val v2ProPlusRoot = File(modelRoot, "v2ProPlus")
        val v2ProPlusIds = v2ProPlusRoot
            .listFiles()
            ?.filter { File(it, "tts_models").isDirectory }
            ?.map { it.name }
            .orEmpty()
        result += CharacterModelCatalog.v2ProPlusFromCharacterIds(v2ProPlusIds)

        val v2Root = File(modelRoot, "v2")
        val v2Ids = v2Root
            .listFiles()
            ?.filter { File(it, "tts_models").isDirectory }
            ?.map { it.name }
            .orEmpty()
        result += CharacterModelCatalog.v2FromCharacterIds(v2Ids)

        return result
    }

    fun inspect(requiredModelFiles: RequiredModelFiles): ModelInspectionResult {
        val modelDirectory = File(modelRoot, requiredModelFiles.assetSubdirectory)
        val present = modelDirectory.list()?.toSet().orEmpty()
        val missingFiles = requiredModelFiles.files.filterNot { fileName ->
            present.contains(fileName) ||
                requiredModelFiles.derivedFiles.any { d ->
                    d.outputFile == fileName && present.contains(d.sourceFile)
                }
        }
        val missingSessionModels = missingSessionModels(modelDirectory, requiredModelFiles)
        return ModelInspectionResult(
            presentFiles = present,
            missingFiles = missingFiles + missingSessionModels,
        )
    }

    fun loadPrimaryModel(requiredModelFiles: RequiredModelFiles): ByteArray {
        return File(modelRoot, "${requiredModelFiles.assetSubdirectory}/${requiredModelFiles.primarySessionModel}")
            .readBytes()
    }

    fun installToPrivateStorage(requiredModelFiles: RequiredModelFiles): File {
        val targetDirectory = File(modelRoot, requiredModelFiles.assetSubdirectory)
        if (!targetDirectory.exists()) {
            targetDirectory.mkdirs()
        }
        val versionFile = File(targetDirectory, ".genie-model-version")
        val expectedVersion = requiredModelFiles.storageVersion.toString()
        if (versionFile.exists() && versionFile.readText() == expectedVersion) {
            val present = targetDirectory.list()?.toSet().orEmpty()
            if (requiredModelFiles.files.all(present::contains) &&
                missingSessionModels(targetDirectory, requiredModelFiles).isEmpty()
            ) {
                return targetDirectory
            }
        }

        val present = targetDirectory.list()?.toSet().orEmpty()

        requiredModelFiles.files.forEach { fileName ->
            val targetFile = File(targetDirectory, fileName)
            if (targetFile.exists()) return@forEach

            val derivedFile = requiredModelFiles.derivedFiles.firstOrNull { it.outputFile == fileName }
            if (derivedFile != null) {
                val sourceFile = File(targetDirectory, derivedFile.sourceFile)
                if (sourceFile.exists()) {
                    HalfPrecisionFileConverter.convertFp16FileToFp32File(sourceFile, targetFile)
                    return@forEach
                }
            }

            require(false) { "Missing model file: ${targetDirectory.absolutePath}/$fileName" }
        }

        val missingSessionModels = missingSessionModels(targetDirectory, requiredModelFiles)
        require(missingSessionModels.isEmpty()) {
            "Missing model file: ${targetDirectory.absolutePath}/${missingSessionModels.first()}"
        }

        versionFile.writeText(expectedVersion)
        return targetDirectory
    }

    fun findReferenceExample(characterModel: CharacterModel): ReferenceExample? {
        // Reference assets live next to each character's tts_models directory for both v2 and v2ProPlus.
        val characterRoot = File(modelRoot, characterModel.relativeModelDirectory).parentFile
            ?: return null
        val promptJson = File(characterRoot, "prompt_wav.json")
        val promptDirectory = File(characterRoot, "prompt_wav")

        if (promptJson.exists()) {
            val (wav, text) = readNormalPromptFields(promptJson.readText()) ?: ("" to null)
            if (wav.isNotBlank() && text != null) {
                val audioFile = File(promptDirectory, wav)
                if (audioFile.exists()) {
                    return ReferenceExample(audioFile = audioFile, referenceText = text)
                }
            }
        }

        val wavFile = promptDirectory
            .listFiles()
            ?.filter { it.isFile && it.extension.equals("wav", ignoreCase = true) }
            ?.sortedBy { it.name }
            ?.firstOrNull()
            ?: return null
        return ReferenceExample(
            audioFile = wavFile,
            referenceText = referenceTextFromFilename(wavFile.nameWithoutExtension),
        )
    }

    private fun readNormalPromptFields(jsonText: String): Pair<String, String?>? {
        val normal = runCatching { JSONObject(jsonText).optJSONObject("Normal") }.getOrNull()
        if (normal != null) {
            val wav = normal.optString("wav").trim()
            val text = normal.takeIf { it.has("text") && !it.isNull("text") }?.optString("text")
            return wav to text
        }

        // Android's JSONObject is a throwing stub in local JVM unit tests.
        return runCatching { PromptJsonCursor(jsonText).readNormalFields() }.getOrNull()
    }

    private fun referenceTextFromFilename(filename: String): String {
        val trimmed = filename.trim()
        val stripped = trimmed.substringAfter('-', trimmed)
        return stripped.ifBlank { trimmed }
    }

    private fun missingSessionModels(
        modelDirectory: File,
        requiredModelFiles: RequiredModelFiles,
    ): List<String> =
        requiredModelFiles.sessionModels.filterNot { fileName ->
            File(modelDirectory, fileName).exists()
        }

    private class PromptJsonCursor(
        private val source: String,
    ) {
        private var index = 0

        fun readNormalFields(): Pair<String, String?>? {
            skipWhitespace()
            expect('{')
            while (true) {
                skipWhitespace()
                if (consume('}')) return null
                val key = readString()
                skipWhitespace()
                expect(':')
                skipWhitespace()
                val fields = if (key == "Normal" && peek() == '{') readFlatObject() else null
                if (fields == null) skipValue()
                skipWhitespace()
                if (consume('}')) return fields?.let { it["wav"].orEmpty() to it["text"] }
                expect(',')
                if (fields != null) {
                    skipRemainingObject()
                    return fields["wav"].orEmpty() to fields["text"]
                }
            }
        }

        private fun readFlatObject(): Map<String, String?> {
            val values = linkedMapOf<String, String?>()
            expect('{')
            while (true) {
                skipWhitespace()
                if (consume('}')) return values
                val key = readString()
                skipWhitespace()
                expect(':')
                skipWhitespace()
                values[key] = if (peek() == '"') readString() else {
                    skipValue()
                    null
                }
                skipWhitespace()
                if (consume('}')) return values
                expect(',')
            }
        }

        private fun skipRemainingObject() {
            while (true) {
                skipWhitespace()
                if (consume('}')) return
                readString()
                skipWhitespace()
                expect(':')
                skipWhitespace()
                skipValue()
                skipWhitespace()
                if (consume('}')) return
                expect(',')
            }
        }

        private fun skipValue() {
            when (peek()) {
                '"' -> readString()
                '{' -> {
                    expect('{')
                    while (true) {
                        skipWhitespace()
                        if (consume('}')) return
                        readString()
                        skipWhitespace()
                        expect(':')
                        skipWhitespace()
                        skipValue()
                        skipWhitespace()
                        if (consume('}')) return
                        expect(',')
                    }
                }
                '[' -> {
                    expect('[')
                    while (true) {
                        skipWhitespace()
                        if (consume(']')) return
                        skipValue()
                        skipWhitespace()
                        if (consume(']')) return
                        expect(',')
                    }
                }
                else -> while (peek()?.let { it !in ",}]" && !it.isWhitespace() } == true) index += 1
            }
        }

        private fun readString(): String {
            expect('"')
            return buildString {
                while (true) {
                    val char = source.getOrNull(index++) ?: throw IllegalArgumentException("Unterminated string.")
                    when (char) {
                        '"' -> return@buildString
                        '\\' -> append(readEscape())
                        else -> append(char)
                    }
                }
            }
        }

        private fun readEscape(): Char {
            return when (val escaped = source.getOrNull(index++) ?: throw IllegalArgumentException("Incomplete escape.")) {
                '"', '\\', '/' -> escaped
                'b' -> '\b'
                'f' -> '\u000C'
                'n' -> '\n'
                'r' -> '\r'
                't' -> '\t'
                'u' -> source.substring(index, index + 4).also { index += 4 }.toInt(16).toChar()
                else -> throw IllegalArgumentException("Invalid escape.")
            }
        }

        private fun skipWhitespace() {
            while (peek()?.isWhitespace() == true) index += 1
        }

        private fun expect(expected: Char) {
            require(consume(expected)) { "Expected '$expected'." }
        }

        private fun consume(expected: Char): Boolean =
            (source.getOrNull(index) == expected).also { if (it) index += 1 }

        private fun peek(): Char? = source.getOrNull(index)
    }
}
