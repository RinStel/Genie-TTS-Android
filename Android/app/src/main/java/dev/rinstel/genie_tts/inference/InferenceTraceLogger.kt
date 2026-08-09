package dev.rinstel.genie_tts.inference

import android.util.Log
import java.math.BigDecimal

enum class InferenceModelRole(val wireValue: String) {
    TEXT_FRONTEND("frontend"),
    ROBERTA("roberta"),
    HUBERT("hubert"),
    SPEAKER_ENCODER("speaker_encoder"),
    PROMPT_ENCODER("prompt_encoder"),
    T2S("t2s"),
    VOCODER("vocoder"),
}

enum class InferenceProvider(val wireValue: String) {
    CPU("cpu"),
    QNN("qnn"),
    XNNPACK("xnnpack"),
}

enum class InferenceCacheStatus(val wireValue: String) {
    HIT("hit"),
    MISS("miss"),
    NOT_APPLICABLE("n/a"),
}

/** Metadata-only record for comparing inference boundaries across runtimes. */
data class InferenceTraceRecord(
    val modelRole: InferenceModelRole,
    val provider: InferenceProvider,
    val tensorShape: LongArray? = null,
    val cacheStatus: InferenceCacheStatus = InferenceCacheStatus.NOT_APPLICABLE,
    val elapsedMs: Long? = null,
) {
    init {
        require(tensorShape?.all { it >= 0L } != false) {
            "Trace tensor dimensions must be non-negative."
        }
        require(elapsedMs == null || elapsedMs >= 0L) {
            "Trace elapsed time must be non-negative."
        }
    }

    fun toLogLine(): String = buildList {
        add("role=${modelRole.wireValue}")
        add("provider=${provider.wireValue}")
        tensorShape?.let { add("shape=[${it.joinToString(",")}]") }
        add("cache=${cacheStatus.wireValue}")
        elapsedMs?.let { add("elapsed_ms=$it") }
    }.joinToString(" ")
}

interface InferenceTraceLogger {
    fun stage(name: String, elapsedMs: Long)

    fun event(event: InferenceTraceEvent)

    object None : InferenceTraceLogger {
        override fun stage(name: String, elapsedMs: Long) {
            requireAllowedStage(name, elapsedMs)
        }

        override fun event(event: InferenceTraceEvent) = Unit
    }

    companion object {
        private val allowedStages = setOf(
            "generation_total_ms",
            "prepare_total_ms",
            "backend_total_ms",
            "write_wav_ms",
            "text_features_ms",
            "reference_text_features_ms",
            "reference_audio_load_ms",
            "reference_audio_resample_16k_ms",
            "hubert_ms",
            "speaker_encoder_ms",
            "prompt_encoder_ms",
            "t2s_encoder_ms",
            "t2s_first_decoder_ms",
            "decoder_loop_ms",
            "vocoder_ms",
        )
        fun requireAllowedStage(name: String, elapsedMs: Long) {
            require(name in allowedStages && elapsedMs >= 0L) {
                "Trace stage metadata is not allowlisted."
            }
        }

    }
}

enum class InferenceRuntimeLabel(val wireValue: String) {
    CPU("ORT CPU EP"),
    QNN("ORT QNN EP"),
    XNNPACK("ORT XNNPACK EP"),
    QNN_HTP("ORT QNN HTP (NPU)"),
    QNN_CPU("ORT QNN CPU"),
}

interface InferenceTensorMetricName {
    val wireValue: String
}

data class NamedInferenceTensorMetric(
    override val wireValue: String,
) : InferenceTensorMetricName {
    init {
        require(wireValue.matches(Regex("[a-z][a-z0-9_]*_hash"))) {
            "Trace tensor metric names must be lowercase hash fields."
        }
    }
}

enum class InferenceTensorMetric(override val wireValue: String) : InferenceTensorMetricName {
    AUDIO_SAMPLES("audio_samples"),
    AUDIO_SHAPE("audio_shape"),
    AUDIO_HASH("audio_hash"),
    TEXT_SEQ_HASH("text_seq_hash"),
    TEXT_BERT_HASH("text_bert_hash"),
    REF_SEQ_HASH("ref_seq_hash"),
    REF_BERT_HASH("ref_bert_hash"),
    REFERENCE_AUDIO_HASH("reference_audio_hash"),
    SSL_CONTENT_HASH("ssl_content_hash"),
    SPEAKER_EMBEDDING_HASH("speaker_embedding_hash"),
    GLOBAL_EMBEDDING_HASH("global_embedding_hash"),
    ADVANCED_GLOBAL_EMBEDDING_HASH("advanced_global_embedding_hash"),
    SEMANTIC_TOKENS("semantic_tokens"),
    SEMANTIC_SHAPE("semantic_shape"),
    SEMANTIC_HASH("semantic_hash"),
}

sealed interface InferenceTraceEvent {
    fun toLogLine(): String

    data class Backend(val value: ExecutionBackend) : InferenceTraceEvent {
        override fun toLogLine(): String = "resolved_backend: ${value.name}"
    }

    data class Runtime(val value: InferenceRuntimeLabel) : InferenceTraceEvent {
        override fun toLogLine(): String = "runtime_label: ${value.wireValue}"
    }

    data class TensorCount(val metric: InferenceTensorMetric, val value: Long) : InferenceTraceEvent {
        init { require(value >= 0L) { "Trace tensor count must be non-negative." } }
        override fun toLogLine(): String = "${metric.wireValue}: $value"
    }

    data class TensorShape(val metric: InferenceTensorMetric, val value: LongArray) : InferenceTraceEvent {
        init { require(value.all { it >= 0L }) { "Trace tensor dimensions must be non-negative." } }
        override fun toLogLine(): String = "${metric.wireValue}: ${value.contentToString()}"
    }

    data class TensorHash(val metric: InferenceTensorMetricName, val value: String) : InferenceTraceEvent {
        init {
            require(metric.wireValue.endsWith("_hash")) {
                "Trace hashes are only allowed for tensor hash metrics."
            }
            require(value.matches(Regex("[0-9a-f]{64}"))) {
                "Trace tensor hash must be a SHA-256 hexadecimal value."
            }
        }

        override fun toLogLine(): String = "${metric.wireValue}: $value"
    }

    data class TensorStatistics(
        val metric: InferenceTensorMetricName,
        val finiteCount: Long,
        val nonFiniteCount: Long,
        val min: Float?,
        val max: Float?,
        val mean: Double?,
    ) : InferenceTraceEvent {
        init {
            require(metric.wireValue.endsWith("_hash")) {
                "Trace statistics require a float tensor hash metric."
            }
            require(finiteCount >= 0L && nonFiniteCount >= 0L) {
                "Trace statistics counts must be non-negative."
            }
            require((min == null) == (max == null) && (min == null) == (mean == null)) {
                "Trace statistics summaries must be present together."
            }
            require(min == null || min!!.isFinite())
            require(max == null || max!!.isFinite())
            require(mean == null || mean.isFinite())
        }

        override fun toLogLine(): String {
            val name = metric.wireValue.removeSuffix("_hash")
            return "${name}_stats: finite_count=$finiteCount " +
                "non_finite_count=$nonFiniteCount " +
                "min=${formatStatistic(min?.toDouble())} " +
                "max=${formatStatistic(max?.toDouble())} " +
                "mean=${formatStatistic(mean)}"
        }

        private fun formatStatistic(value: Double?): String = when {
            value == null -> "null"
            value == 0.0 -> "0"
            else -> BigDecimal.valueOf(value).stripTrailingZeros().toPlainString()
        }
    }

    data class Cache(val status: InferenceCacheStatus) : InferenceTraceEvent {
        override fun toLogLine(): String = "reference_cache: ${status.wireValue}"
    }

    data class Model(val value: InferenceTraceRecord) : InferenceTraceEvent {
        override fun toLogLine(): String = value.toLogLine()
    }
}

class LogcatInferenceTraceLogger(
    private val tag: String = TAG,
) : InferenceTraceLogger {
    override fun stage(name: String, elapsedMs: Long) {
        InferenceTraceLogger.requireAllowedStage(name, elapsedMs)
        Log.i(tag, "$name=$elapsedMs ms")
    }

    override fun event(event: InferenceTraceEvent) {
        Log.i(tag, event.toLogLine())
    }

    companion object {
        private const val TAG = "GenieTtsTiming"
    }
}

class InferenceTimer(
    private val logger: InferenceTraceLogger,
    private val clockNanos: () -> Long = System::nanoTime,
) {
    fun <T> measure(name: String, block: () -> T): T {
        InferenceTraceLogger.requireAllowedStage(name, 0L)
        val start = clockNanos()
        try {
            return block()
        } finally {
            logger.stage(name, nanosToMillis(clockNanos() - start))
        }
    }

    fun event(event: InferenceTraceEvent) {
        logger.event(event)
    }

    private fun nanosToMillis(nanos: Long): Long = nanos / NANOS_PER_MILLISECOND

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}

fun InferenceTimer.traceFloatTensor(
    metric: InferenceTensorMetricName,
    values: FloatArray,
) {
    val summary = InferenceTraceHash.floatSummary(values)
    event(InferenceTraceEvent.TensorHash(metric, summary.hash))
    event(
        InferenceTraceEvent.TensorStatistics(
            metric = metric,
            finiteCount = summary.finiteCount,
            nonFiniteCount = summary.nonFiniteCount,
            min = summary.min,
            max = summary.max,
            mean = summary.mean,
        ),
    )
}
