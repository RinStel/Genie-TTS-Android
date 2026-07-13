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
        if (fixtureType == "genie_tts_parity_contract") validateParityContract(root)
        return root
    }

    private fun validateParityContract(root: Map<String, Any?>) {
        root["cases"].asArray("cases").forEach { rawCase ->
            val expected = rawCase.asObject("case")["expected"].asObject("expected")
            require(expected["normalized_text"] is String) { "expected.normalized_text must be a string." }
            require(expected["phones"].asArray("expected.phones").isNotEmpty()) { "expected.phones must not be empty." }
            expected["phones"].asArray("expected.phones").forEach { require(it is String) { "expected.phones must contain strings." } }
            requireIntegerTensor(expected["phone_ids"], "expected.phone_ids")
            requireIntegerTensor(expected["word2ph"], "expected.word2ph")
            requireIntegerTensor(expected["semantic_tokens"], "expected.semantic_tokens")

            val tensors = expected["tensors"].asObject("expected.tensors")
            listOf("reference_pcm", "hubert_features", "speaker_embedding", "prompt_conditioning", "t2s_output", "vocoder_output").forEach { name ->
                requireFloatTensor(tensors[name], "expected.tensors.$name")
            }

            val stages = expected["timing"].asObject("expected.timing")["stages"].asArray("expected.timing.stages")
            require(stages.size == 6) { "expected.timing.stages must contain six records." }
            stages.forEach { stage ->
                val record = stage.asObject("timing stage")
                require(record["name"] is String) { "timing stage name must be a string." }
                requireNonNegativeInteger(record["elapsed_ms"], "timing stage elapsed_ms")
            }

            val trace = expected["trace"].asArray("expected.trace")
            require(trace.size == 6) { "expected.trace must contain six records." }
            trace.forEach { rawRecord ->
                val record = rawRecord.asObject("trace record")
                listOf("model_role", "provider", "cache_status").forEach { key ->
                    require(record[key] is String) { "trace $key must be a string." }
                }
                record["tensor_shape"].asArray("trace tensor_shape").forEach { requireNonNegativeInteger(it, "trace tensor_shape") }
                requireNonNegativeInteger(record["elapsed_ms"], "trace elapsed_ms")
            }
        }
    }

    private fun requireIntegerTensor(value: Any?, label: String) {
        val tensor = value.asObject(label)
        require(tensor["dtype"] == "int64") { "$label dtype must be int64." }
        requireShape(tensor["shape"], "$label.shape")
        tensor["values"].asArray("$label.values").forEach { require(it is Long) { "$label values must be integers." } }
    }

    private fun requireFloatTensor(value: Any?, label: String) {
        val tensor = value.asObject(label)
        require(tensor["dtype"] == "float32") { "$label dtype must be float32." }
        requireShape(tensor["shape"], "$label.shape")
        require(tensor["sha256"] is String && (tensor["sha256"] as String).matches(Regex("[0-9a-f]{64}"))) { "$label sha256 must be a 64-character lowercase hex string." }
        val statistics = tensor["statistics"].asObject("$label.statistics")
        requireNonNegativeInteger(statistics["finite_count"], "$label.statistics.finite_count")
        requireNonNegativeInteger(statistics["non_finite_count"], "$label.statistics.non_finite_count")
        listOf("min", "max", "mean").forEach { key ->
            require(statistics[key] is Number && (statistics[key] as Number).toDouble().isFinite()) { "$label.statistics.$key must be finite." }
        }
    }

    private fun requireShape(value: Any?, label: String) {
        value.asArray(label).forEach { requireNonNegativeInteger(it, label) }
    }

    private fun requireNonNegativeInteger(value: Any?, label: String) {
        require(value is Long && value >= 0L) { "$label must be a non-negative integer." }
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
