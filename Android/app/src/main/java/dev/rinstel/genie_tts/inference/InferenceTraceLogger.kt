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

    fun event(name: String, message: String)

    /** Emits typed metadata only; callers must never attach user text or audio samples. */
    fun trace(record: InferenceTraceRecord) {
        event("trace", record.toLogLine())
    }

    object None : InferenceTraceLogger {
        override fun stage(name: String, elapsedMs: Long) = Unit

        override fun event(name: String, message: String) = Unit
    }
}

class LogcatInferenceTraceLogger(
    private val tag: String = TAG,
) : InferenceTraceLogger {
    override fun stage(name: String, elapsedMs: Long) {
        Log.i(tag, "$name=$elapsedMs ms")
    }

    override fun event(name: String, message: String) {
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
        val start = clockNanos()
        try {
            return block()
        } finally {
            logger.stage(name, nanosToMillis(clockNanos() - start))
        }
    }

    fun event(name: String, message: String) {
        logger.event(name, message)
    }

    private fun nanosToMillis(nanos: Long): Long = nanos / NANOS_PER_MILLISECOND

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000L
    }
}
