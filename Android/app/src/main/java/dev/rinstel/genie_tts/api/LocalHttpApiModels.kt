package dev.rinstel.genie_tts.api

import dev.rinstel.genie_tts.BackendServiceState
import dev.rinstel.genie_tts.inference.CharacterModel
import dev.rinstel.genie_tts.inference.ExecutionBackend
import java.io.File

data class ApiServerStatus(
    val running: Boolean,
    val port: Int,
    val address: String = "127.0.0.1",
    val lastError: String? = null,
)

data class ActiveDefaults(
    val backend: ExecutionBackend,
    val modelId: String,
    val language: String,
    val referenceAudioPath: String,
    val referenceText: String,
    val maxDecoderSteps: Int,
)

sealed class InferResult {
    data class Success(val outputFile: File) : InferResult()
    data class Error(val message: String) : InferResult()
    data object Busy : InferResult()
    data object Timeout : InferResult()
}

internal data class HttpResponse(
    val statusCode: Int,
    val body: String,
    val contentType: String = "application/json; charset=utf-8",
    val binaryBody: ByteArray? = null,
)

internal data class InferPayload(
    val backendName: String,
    val modelId: String,
    val language: String,
    val promptLanguage: String = language,
    val text: String,
    val referenceAudioPath: String,
    val referenceText: String,
    val maxDecoderSteps: Int,
)

internal fun BackendServiceState.toJsonString(): String =
    jsonObject(
        "stage" to stage.name,
        "message" to message,
        "busy" to busy,
        "requestedBackend" to requestedBackend.name,
        "resolvedBackend" to resolvedBackend.name,
        "runtimeLabel" to runtimeLabel,
        "initializedModelId" to initializedModelId,
        "latestOutputFilePath" to latestOutputFilePath,
        "progressPercent" to progressPercent,
        "progressLabel" to progressLabel,
        "generationDurationMs" to generationDurationMs,
        "lastCompletedAtMs" to lastCompletedAtMs,
    )

internal fun List<CharacterModel>.toJsonString(): String =
    jsonArray(
        map { model ->
            jsonObject(
                "id" to model.id,
                "displayName" to model.displayName,
                "relativeModelDirectory" to model.relativeModelDirectory,
            )
        },
    )

internal fun parseInferPayload(body: String): InferPayload {
    val values = parseJsonObject(body.ifBlank { "{}" })
    return InferPayload(
        backendName = values["backend"].orEmpty(),
        modelId = values["modelId"].orEmpty().trim(),
        language = values["language"].orEmpty().trim(),
        promptLanguage = values["promptLanguage"]?.trim().takeUnless { it.isNullOrEmpty() }
            ?: values["language"].orEmpty().trim(),
        text = values["text"].orEmpty().trim(),
        referenceAudioPath = values["referenceAudioPath"].orEmpty().trim(),
        referenceText = values["referenceText"].orEmpty().trim(),
        maxDecoderSteps = values["maxDecoderSteps"]?.trim()?.toIntOrNull() ?: 500,
    )
}

internal fun jsonObject(vararg entries: Pair<String, Any?>): String =
    entries.joinToString(
        prefix = "{",
        postfix = "}",
        separator = ",",
    ) { (key, value) ->
        "${quoteJson(key)}:${jsonValue(value)}"
    }

internal fun jsonArray(items: List<String>): String =
    items.joinToString(prefix = "[", postfix = "]", separator = ",")

private fun jsonValue(value: Any?): String =
    when (value) {
        null -> "null"
        is Number, is Boolean -> value.toString()
        is RawJson -> value.value
        else -> quoteJson(value.toString())
    }

private fun quoteJson(value: String): String {
    val escaped = buildString(value.length + 8) {
        value.forEach { char ->
            when (char) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (char.code < 0x20) {
                        append("\\u%04x".format(char.code))
                    } else {
                        append(char)
                    }
                }
            }
        }
    }
    return "\"$escaped\""
}

// The localhost API only needs a tiny flat-object parser, so keep it local and dependency-free.
private fun parseJsonObject(source: String): Map<String, String?> {
    val cursor = JsonCursor(source)
    val result = linkedMapOf<String, String?>()
    cursor.skipWhitespace()
    cursor.expect('{')
    cursor.skipWhitespace()
    if (cursor.peek() == '}') {
        cursor.expect('}')
        return result
    }
    while (true) {
        cursor.skipWhitespace()
        val key = cursor.readString()
        cursor.skipWhitespace()
        cursor.expect(':')
        cursor.skipWhitespace()
        result[key] = cursor.readPrimitiveOrString()
        cursor.skipWhitespace()
        when (cursor.readDelimiter()) {
            ',' -> continue
            '}' -> return result
            else -> throw IllegalArgumentException("Malformed JSON object.")
        }
    }
}

private class JsonCursor(
    private val source: String,
) {
    private var index = 0

    fun skipWhitespace() {
        while (index < source.length && source[index].isWhitespace()) {
            index += 1
        }
    }

    fun peek(): Char? = source.getOrNull(index)

    fun expect(expected: Char) {
        val actual = source.getOrNull(index)
        require(actual == expected) { "Expected '$expected'." }
        index += 1
    }

    fun readDelimiter(): Char {
        val value = source.getOrNull(index) ?: throw IllegalArgumentException("Unexpected end of JSON.")
        index += 1
        return value
    }

    fun readString(): String {
        expect('"')
        val output = StringBuilder()
        while (true) {
            val current = source.getOrNull(index) ?: throw IllegalArgumentException("Unterminated JSON string.")
            index += 1
            when (current) {
                '"' -> return output.toString()
                '\\' -> output.append(readEscape())
                else -> output.append(current)
            }
        }
    }

    fun readPrimitiveOrString(): String? {
        if (peek() == '"') {
            return readString()
        }
        val start = index
        while (index < source.length) {
            val current = source[index]
            if (current == ',' || current == '}' || current.isWhitespace()) {
                break
            }
            index += 1
        }
        val token = source.substring(start, index).trim()
        return when (token) {
            "", "null" -> null
            else -> token
        }
    }

    private fun readEscape(): Char {
        val escaped = source.getOrNull(index) ?: throw IllegalArgumentException("Incomplete JSON escape.")
        index += 1
        return when (escaped) {
            '"', '\\', '/' -> escaped
            'b' -> '\b'
            'f' -> '\u000C'
            'n' -> '\n'
            'r' -> '\r'
            't' -> '\t'
            'u' -> {
                val hex = source.substring(index, (index + 4).coerceAtMost(source.length))
                require(hex.length == 4) { "Incomplete unicode escape." }
                require(hex.all { it.isDigit() || it.lowercaseChar() in 'a'..'f' }) { "Invalid unicode escape." }
                index += 4
                hex.toInt(16).toChar()
            }
            else -> throw IllegalArgumentException("Unsupported JSON escape.")
        }
    }
}

@JvmInline
internal value class RawJson(
    val value: String,
)
