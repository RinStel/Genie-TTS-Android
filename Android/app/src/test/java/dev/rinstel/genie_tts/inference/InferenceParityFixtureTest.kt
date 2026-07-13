package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class InferenceParityFixtureTest {
    @Test
    fun structuredTraceRecordIncludesParityMetadataWithoutUserContent() {
        val record = InferenceTraceRecord(
            modelRole = InferenceModelRole.TEXT_FRONTEND,
            provider = InferenceProvider.CPU,
            tensorShape = longArrayOf(1L, 4L, 1024L),
            cacheStatus = InferenceCacheStatus.HIT,
            elapsedMs = 12L,
        )

        val line = record.toLogLine()

        assertEquals(
            "role=frontend provider=cpu shape=[1,4,1024] cache=hit elapsed_ms=12",
            line,
        )
        assertFalse(line.contains("text", ignoreCase = true))
        assertFalse(line.contains("audio", ignoreCase = true))
    }

    @Test
    fun loadsAndValidatesVersionedFullContractFixture() {
        val fixture = InferenceParityFixtureReader.read("parity/full_contract_case.json")
        val expected = InferenceParityFixtureAssertions.singleExpectedCase(fixture)

        assertEquals(1L, fixture["schema_version"])
        assertEquals(
            listOf(0L, -2147483648L, 42L, 2147483647L, 1L),
            expected.tensorValues("phone_ids"),
        )
        assertEquals(listOf(-2147483648L, 2147483647L), expected.tensorValues("semantic_tokens"))
        assertEquals("cpu", expected.firstTrace()["provider"])
    }

    @Test
    fun fixtureReaderRejectsMissingAndUnsupportedSchemaVersions() {
        assertThrows(IllegalArgumentException::class.java) {
            InferenceParityFixtureReader.decode("{\"fixture_type\":\"chinese_frontend\",\"cases\":[]}")
        }
        assertThrows(IllegalArgumentException::class.java) {
            InferenceParityFixtureReader.decode(
                "{\"schema_version\":2,\"fixture_type\":\"chinese_frontend\",\"cases\":[]}",
            )
        }
    }

    @Test
    fun fullContractReaderRejectsEmptyExpectedAndMissingBoundaryRecords() {
        val valid = javaClass.classLoader
            ?.getResourceAsStream("parity/full_contract_case.json")
            ?.bufferedReader(Charsets.UTF_8)
            ?.use { it.readText() }
            ?: error("full parity fixture resource was not found")

        assertThrows(IllegalArgumentException::class.java) {
            InferenceParityFixtureReader.decode(valid.replace("\"expected\": {", "\"expected\": {}"))
        }
        assertThrows(IllegalArgumentException::class.java) {
            InferenceParityFixtureReader.decode(valid.replace("\"reference_pcm\": {", "\"missing_pcm\": {"))
        }
        assertThrows(IllegalArgumentException::class.java) {
            InferenceParityFixtureReader.decode(valid.replace("\"sha256\": \"", "\"sha256\": \"not-a-hash"))
        }
        assertThrows(IllegalArgumentException::class.java) {
            InferenceParityFixtureReader.decode(valid.replace("\"elapsed_ms\": 12", "\"elapsed_ms\": -1"))
        }
    }

    @Test
    fun fullContractReaderRejectsMissingOrInvalidParitySections() {
        val valid = javaClass.classLoader
            ?.getResourceAsStream("parity/full_contract_case.json")
            ?.bufferedReader(Charsets.UTF_8)
            ?.use { it.readText() }
            ?: error("full parity fixture resource was not found")

        listOf(
            valid.replace("\"normalized_text\":", "\"missing_normalized_text\":"),
            valid.replace("\"reference_pcm\":", "\"missing_reference_pcm\":"),
            valid.replace("\"vocoder_output\":", "\"missing_vocoder_output\":"),
            valid.replace("\"semantic_tokens\":", "\"missing_semantic_tokens\":"),
            valid.replace("\"timing\":", "\"missing_timing\":"),
            valid.replace("\"trace\":", "\"missing_trace\":"),
            valid.replace("\"dtype\": \"float32\"", "\"dtype\": \"int64\""),
            valid.replace("\"dtype\": \"int64\"", "\"dtype\": \"float32\""),
            valid.replace("\"elapsed_ms\": 12", "\"elapsed_ms\": -1"),
        ).forEach { invalid ->
            assertThrows(IllegalArgumentException::class.java) {
                InferenceParityFixtureReader.decode(invalid)
            }
        }
    }

    @Test
    fun traceLoggerDoesNotExposeRawStringEventInterface() {
        assertFalse(
            InferenceTraceLogger::class.java.methods.any { method ->
                method.name == "event" &&
                    method.parameterTypes.contentEquals(arrayOf(String::class.java, String::class.java))
            },
        )
    }

    @Test
    fun legacyTraceEventsRejectRawUserMessages() {
        assertFalse(InferenceTraceLogger.isAllowedLegacyEvent("runtime_label", "用户输入文本"))
        assertFalse(InferenceTraceLogger.isAllowedLegacyEvent("runtime_label", "C:/private/model.onnx"))
        assertFalse(InferenceTraceLogger.isAllowedLegacyEvent("unknown", "42"))
        assertEquals(true, InferenceTraceLogger.isAllowedLegacyEvent("semantic_tokens", "42"))
        assertEquals(true, InferenceTraceLogger.isAllowedLegacyEvent("reference_cache", "hit"))
        assertEquals(true, InferenceTraceLogger.isAllowedLegacyEvent("runtime_label", "ORT QNN EP"))
    }

    @Test
    fun inferenceTimerRejectsUnsafeEventBeforeLoggerBoundary() {
        var captured: Pair<String, String>? = null
        val logger = object : InferenceTraceLogger {
            override fun stage(name: String, elapsedMs: Long) = Unit

            override fun event(name: String, message: String) {
                captured = name to message
            }
        }

        assertThrows(IllegalArgumentException::class.java) {
            InferenceTimer(logger).event("runtime_label", "C:/private/model.onnx")
        }
        assertNull(captured)
    }

    @Test
    fun structuredTraceRejectsNegativeTensorDimensions() {
        assertThrows(IllegalArgumentException::class.java) {
            InferenceTraceRecord(
                modelRole = InferenceModelRole.T2S,
                provider = InferenceProvider.CPU,
                tensorShape = longArrayOf(1L, -1L),
            )
        }
    }
}
