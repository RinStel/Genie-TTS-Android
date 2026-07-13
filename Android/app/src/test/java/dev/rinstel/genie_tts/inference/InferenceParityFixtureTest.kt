package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
}
