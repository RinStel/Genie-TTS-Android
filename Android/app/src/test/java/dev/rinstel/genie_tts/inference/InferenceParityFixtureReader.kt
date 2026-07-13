package dev.rinstel.genie_tts.inference

internal object InferenceParityFixtureReader {
    fun read(resourceName: String): Map<String, Any?> {
        val stream = requireNotNull(javaClass.classLoader?.getResourceAsStream(resourceName)) {
            "Parity fixture resource was not found."
        }
        return stream.bufferedReader(Charsets.UTF_8).use { decode(it.readText()) }
    }

    fun decode(json: String): Map<String, Any?> {
        val root = JsonParser(json).parse().asObject("fixture root")
        require(root["schema_version"] == 1L) { "schema_version must be integer 1." }
        val fixtureType = root["fixture_type"]
        require(
            fixtureType is String &&
                fixtureType in setOf("chinese_frontend", "genie_tts_parity_contract"),
        ) {
            "fixture_type is unsupported."
        }
        require(root["cases"].asArray("cases").isNotEmpty()) { "cases must not be empty." }
        return root
    }
}

internal object InferenceParityFixtureAssertions {
    fun singleExpectedCase(fixture: Map<String, Any?>): ExpectedParityCase {
        val cases = fixture["cases"].asArray("cases")
        require(cases.size == 1) { "Expected exactly one parity case." }
        val case = cases.single().asObject("case")
        return ExpectedParityCase(case["expected"].asObject("expected"))
    }
}

internal class ExpectedParityCase(
    private val expected: Map<String, Any?>,
) {
    fun tensorValues(name: String): List<Long> = values(expected[name], name)

    fun namedTensorValues(name: String): List<Long> =
        values(expected["tensors"].asObject("tensors")[name], name)

    fun firstTrace(): Map<String, Any?> =
        expected["trace"].asArray("trace").first().asObject("trace record")

    private fun values(record: Any?, label: String): List<Long> =
        record.asObject(label)["values"].asArray("$label.values").map { value ->
            require(value is Long) { "$label values must be integers." }
            value
        }
}

private fun Any?.asObject(label: String): Map<String, Any?> {
    require(this is Map<*, *>) { "$label must be an object." }
    return entries.associate { (key, value) ->
        require(key is String) { "$label keys must be strings." }
        key to value
    }
}

private fun Any?.asArray(label: String): List<Any?> {
    require(this is List<*>) { "$label must be an array." }
    return this
}

private class JsonParser(private val source: String) {
    private var index = 0

    fun parse(): Any? {
        val value = value()
        whitespace()
        require(index == source.length) { "Unexpected trailing JSON content." }
        return value
    }

    private fun value(): Any? {
        whitespace()
        require(index < source.length) { "Unexpected end of JSON." }
        return when (source[index]) {
            '{' -> objectValue()
            '[' -> arrayValue()
            '"' -> stringValue()
            't' -> literal("true", true)
            'f' -> literal("false", false)
            'n' -> literal("null", null)
            else -> numberValue()
        }
    }

    private fun objectValue(): Map<String, Any?> {
        expect('{')
        val result = linkedMapOf<String, Any?>()
        whitespace()
        if (take('}')) return result
        while (true) {
            whitespace()
            val key = stringValue()
            whitespace()
            expect(':')
            result[key] = value()
            whitespace()
            if (take('}')) return result
            expect(',')
        }
    }

    private fun arrayValue(): List<Any?> {
        expect('[')
        val result = mutableListOf<Any?>()
        whitespace()
        if (take(']')) return result
        while (true) {
            result += value()
            whitespace()
            if (take(']')) return result
            expect(',')
        }
    }

    private fun stringValue(): String {
        expect('"')
        val result = StringBuilder()
        while (index < source.length) {
            val char = source[index++]
            if (char == '"') return result.toString()
            if (char != '\\') {
                result.append(char)
                continue
            }
            require(index < source.length) { "Invalid JSON escape." }
            result.append(
                when (val escaped = source[index++]) {
                    '"', '\\', '/' -> escaped
                    'b' -> '\b'
                    'f' -> '\u000c'
                    'n' -> '\n'
                    'r' -> '\r'
                    't' -> '\t'
                    'u' -> {
                        require(index + 4 <= source.length) { "Invalid unicode escape." }
                        source.substring(index, index + 4).toInt(16).toChar().also { index += 4 }
                    }
                    else -> error("Invalid JSON escape.")
                },
            )
        }
        error("Unterminated JSON string.")
    }

    private fun numberValue(): Number {
        val start = index
        while (index < source.length && source[index] in "-+0123456789.eE") index++
        val token = source.substring(start, index)
        return if (token.any { it == '.' || it == 'e' || it == 'E' }) {
            token.toDouble()
        } else {
            token.toLong()
        }
    }

    private fun literal(token: String, value: Any?): Any? {
        require(source.startsWith(token, index)) { "Invalid JSON literal." }
        index += token.length
        return value
    }

    private fun whitespace() {
        while (index < source.length && source[index].isWhitespace()) index++
    }

    private fun take(char: Char): Boolean {
        if (index < source.length && source[index] == char) {
            index++
            return true
        }
        return false
    }

    private fun expect(char: Char) {
        require(take(char)) { "Expected JSON delimiter." }
    }
}
