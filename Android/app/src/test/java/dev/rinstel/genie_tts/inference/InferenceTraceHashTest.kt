package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class InferenceTraceHashTest {
    @Test
    fun hashesAreStableAndTypeSpecific() {
        assertEquals(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            InferenceTraceHash.sha256(FloatArray(0)),
        )
        assertEquals(
            InferenceTraceHash.sha256(longArrayOf(1L, 2L)),
            InferenceTraceHash.sha256(longArrayOf(1L, 2L)),
        )
    }

    @Test
    fun floatSummaryUsesFiniteValuesAndKeepsNonFiniteCount() {
        val summary = InferenceTraceHash.floatSummary(
            floatArrayOf(-1.0f, 1.0f, Float.NaN, Float.POSITIVE_INFINITY),
        )

        assertEquals(2L, summary.finiteCount)
        assertEquals(2L, summary.nonFiniteCount)
        assertEquals(-1.0f, summary.min)
        assertEquals(1.0f, summary.max)
        assertEquals(0.0, requireNotNull(summary.mean), 0.0)
    }

    @Test
    fun tensorHashEventOnlyAcceptsSha256OutputHashes() {
        val event = InferenceTraceEvent.TensorHash(
            InferenceTensorMetric.AUDIO_HASH,
            InferenceTraceHash.sha256(floatArrayOf(0.25f)),
        )

        assertEquals("audio_hash: ${event.value}", event.toLogLine())
    }

    @Test
    fun tensorStatisticsEventContainsOnlyNumericMetadata() {
        val event = InferenceTraceEvent.TensorStatistics(
            metric = InferenceTensorMetric.AUDIO_HASH,
            finiteCount = 2L,
            nonFiniteCount = 2L,
            min = -1.0f,
            max = 1.0f,
            mean = 0.0,
        )

        assertEquals(
            "audio_stats: finite_count=2 non_finite_count=2 min=-1 max=1 mean=0",
            event.toLogLine(),
        )
    }

    @Test
    fun namedTensorMetricsSupportPerReferenceSpeakerEmbeddings() {
        val metric = NamedInferenceTensorMetric("speaker_embedding_3_hash")
        val hash = InferenceTraceHash.sha256(floatArrayOf(0.5f))

        assertEquals(
            "speaker_embedding_3_hash: $hash",
            InferenceTraceEvent.TensorHash(metric, hash).toLogLine(),
        )
        assertEquals(
            "speaker_embedding_3_stats: finite_count=1 non_finite_count=0 " +
                "min=0.5 max=0.5 mean=0.5",
            InferenceTraceEvent.TensorStatistics(
                metric = metric,
                finiteCount = 1L,
                nonFiniteCount = 0L,
                min = 0.5f,
                max = 0.5f,
                mean = 0.5,
            ).toLogLine(),
        )
    }
}
