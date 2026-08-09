package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class GenerationPipelineTest {
    @Test
    fun rejectsBlankSynthesisText() {
        val request = GenerationRequest(
            characterModel = CharacterModelCatalog.v2ProPlusFromCharacterIds(listOf("mansui")).first(),
            language = "zh",
            synthesisText = "",
            referenceAudioPath = "/tmp/ref.wav",
            referenceText = "reference",
        )

        val error = runCatching {
            GenerationPipeline(
                backend = initializedBackend(),
                inputPreparer = neverCalledInputPreparer(),
                outputDirectory = createTempDir(),
            ).generate(request)
        }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
        assertTrue(error?.message?.contains("Synthesis text") == true)
    }

    @Test
    fun writesGeneratedAudioToOutputFile() {
        val outputDirectory = createTempDir()
        var capturedMaxDecoderSteps = -1
        val traceLogger = RecordingTraceLogger()
        val request = GenerationRequest(
            characterModel = CharacterModelCatalog.v2ProPlusFromCharacterIds(listOf("mansui")).first(),
            language = "zh",
            synthesisText = "test",
            referenceAudioPath = "/tmp/ref.wav",
            referenceText = "reference",
            maxDecoderSteps = 123,
        )
        val backend = object : PreparedTtsBackend {
            override fun isInitialized(): Boolean = true

            override fun generatePrepared(input: TtsPreparedInput, maxDecoderSteps: Int): TtsGenerationResult {
                capturedMaxDecoderSteps = maxDecoderSteps
                return TtsGenerationResult(floatArrayOf(0f, 0.25f, -0.25f), longArrayOf(1L, 1L, 3L))
            }
        }
        val inputPreparer = object : TtsInputPreparer {
            override fun prepare(request: GenerationRequest): TtsPreparedInput =
                TtsPreparedInput(
                    refSeq = LongTensorData(longArrayOf(1L), longArrayOf(1L, 1L)),
                    textSeq = LongTensorData(longArrayOf(2L), longArrayOf(1L, 1L)),
                    refBert = FloatTensorData(FloatArray(1024), longArrayOf(1L, 1024L)),
                    textBert = FloatTensorData(FloatArray(1024), longArrayOf(1L, 1024L)),
                    sslContent = FloatTensorData(floatArrayOf(0f), longArrayOf(1L, 1L, 1L)),
                    globalEmbedding = FloatTensorData(floatArrayOf(0f), longArrayOf(1L, 1L)),
                    advancedGlobalEmbedding = FloatTensorData(floatArrayOf(0f), longArrayOf(1L, 1L)),
                )
        }

        val output = GenerationPipeline(
            backend = backend,
            inputPreparer = inputPreparer,
            outputDirectory = outputDirectory,
            traceLogger = traceLogger,
            executionBackend = ExecutionBackend.QNN,
            runtimeLabel = "ORT QNN EP",
        ).generate(request)

        assertTrue(output.file.exists())
        assertEquals(File(outputDirectory, output.file.name).absolutePath, output.file.absolutePath)
        assertEquals(3, output.audioSamples)
        assertEquals(123, capturedMaxDecoderSteps)
        assertEquals(
            listOf(
                "prepare_total_ms",
                "backend_total_ms",
                "write_wav_ms",
                "generation_total_ms",
            ),
            traceLogger.stageNames,
        )
        assertEquals(
            listOf(
                "resolved_backend" to "QNN",
                "runtime_label" to "ORT QNN EP",
                "audio_samples" to "3",
                "audio_shape" to "[1, 1, 3]",
                "audio_hash" to "831abcaa2f5e86e35e2cae1287ec4920e504cc71c88311711907790974cb2340",
                "audio_stats" to "finite_count=3 non_finite_count=0 min=-0.25 max=0.25 mean=0",
            ),
            traceLogger.events.map { event ->
                val parts = event.toLogLine().split(": ", limit = 2)
                parts[0] to parts[1]
            },
        )
    }

    private fun initializedBackend(): PreparedTtsBackend =
        object : PreparedTtsBackend {
            override fun isInitialized(): Boolean = true

            override fun generatePrepared(input: TtsPreparedInput, maxDecoderSteps: Int): TtsGenerationResult =
                error("Backend should not be called.")
        }

    private fun neverCalledInputPreparer(): TtsInputPreparer =
        object : TtsInputPreparer {
            override fun prepare(request: GenerationRequest): TtsPreparedInput =
                error("Input preparer should not be called.")
        }

    private class RecordingTraceLogger : InferenceTraceLogger {
        val stageNames = mutableListOf<String>()
        val events = mutableListOf<InferenceTraceEvent>()

        override fun stage(name: String, elapsedMs: Long) {
            stageNames += name
        }

        override fun event(event: InferenceTraceEvent) {
            events += event
        }
    }
}
