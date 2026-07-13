package dev.rinstel.genie_tts.inference

import android.util.Log

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

    /** Legacy compatibility boundary accepting only allowlisted numeric or enum metadata. */
    fun event(name: String, message: String)

    /** Emits typed metadata only; callers must never attach user text or audio samples. */
    fun trace(record: InferenceTraceRecord) {
        event("trace", record.toLogLine())
    }

    object None : InferenceTraceLogger {
        override fun stage(name: String, elapsedMs: Long) {
            requireAllowedStage(name, elapsedMs)
        }

        override fun event(name: String, message: String) {
            requireAllowedLegacyEvent(name, message)
        }
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
        private val runtimeLabels = setOf(
            "ORT CPU EP",
            "ORT QNN EP",
            "ORT XNNPACK EP",
            "ORT QNN HTP (NPU)",
            "ORT QNN CPU",
        )
        private val traceLinePattern = Regex(
            "role=(frontend|roberta|hubert|speaker_encoder|prompt_encoder|t2s|vocoder) " +
                "provider=(cpu|qnn|xnnpack)( shape=\\[(\\d+(,\\d+)*)?])? " +
                "cache=(hit|miss|n/a)( elapsed_ms=\\d+)?",
        )

        fun isAllowedLegacyEvent(name: String, message: String): Boolean =
            when (name) {
                "resolved_backend" -> message in setOf("CPU", "QNN", "XNNPACK")
                "runtime_label" -> message in runtimeLabels
                "audio_samples", "semantic_tokens" -> message.isNonNegativeInteger()
                "audio_shape", "semantic_shape" -> message.isTensorShape()
                "reference_cache" -> message == "hit" || message == "miss"
                "trace" -> traceLinePattern.matches(message)
                else -> false
            }

        fun requireAllowedLegacyEvent(name: String, message: String) {
            require(isAllowedLegacyEvent(name, message)) {
                "Trace event metadata is not allowlisted."
            }
        }

        fun requireAllowedStage(name: String, elapsedMs: Long) {
            require(name in allowedStages && elapsedMs >= 0L) {
                "Trace stage metadata is not allowlisted."
            }
        }

        private fun String.isNonNegativeInteger(): Boolean =
            toLongOrNull()?.let { it >= 0L } == true

        private fun String.isTensorShape(): Boolean {
            if (length < 2 || first() != '[' || last() != ']') return false
            val dimensions = substring(1, lastIndex).trim()
            if (dimensions.isEmpty()) return true
            return dimensions.split(',').all { part ->
                part.trim().toLongOrNull()?.let { it >= 0L } == true
            }
        }
    }
}

class LogcatInferenceTraceLogger(
    private val tag: String = TAG,
) : InferenceTraceLogger {
    override fun stage(name: String, elapsedMs: Long) {
        InferenceTraceLogger.requireAllowedStage(name, elapsedMs)
        Log.i(tag, "$name=$elapsedMs ms")
    }

    override fun event(name: String, message: String) {
        InferenceTraceLogger.requireAllowedLegacyEvent(name, message)
        Log.i(tag, "$name: $message")
    }

    override fun trace(record: InferenceTraceRecord) {
        Log.i(tag, record.toLogLine())
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

    fun event(name: String, message: String) {
        InferenceTraceLogger.requireAllowedLegacyEvent(name, message)
        logger.event(name, message)
    }

    private fun nanosToMillis(nanos: Long): Long = nanos / NANOS_PER_MILLISECOND

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}
