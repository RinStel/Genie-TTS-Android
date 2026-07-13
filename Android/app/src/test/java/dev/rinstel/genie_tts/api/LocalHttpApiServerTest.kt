package dev.rinstel.genie_tts.api

import dev.rinstel.genie_tts.BackendServiceState
import dev.rinstel.genie_tts.inference.CharacterModel
import dev.rinstel.genie_tts.inference.CharacterModelCatalog
import dev.rinstel.genie_tts.inference.ExecutionBackend
import dev.rinstel.genie_tts.inference.GenerationRequest
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalHttpApiServerTest {
    @Test
    fun inferMapsTextFieldIntoGenerationRequest() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "QNN",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "/tmp/ref.wav",
                  "referenceText": "sample",
                  "maxDecoderSteps": 321
                }
            """.trimIndent(),
        )

        assertEquals(200, response.statusCode)
        assertEquals("audio/wav", response.contentType)
        assertTrue(response.binaryBody?.isNotEmpty() == true)
        assertEquals(ExecutionBackend.QNN, bridge.capturedBackend)
        assertEquals("hello", bridge.capturedRequest?.synthesisText)
        assertEquals(321, bridge.capturedRequest?.maxDecoderSteps)
    }

    @Test
    fun inferMapsPromptLanguageIndependentlyFromSynthesisLanguage() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "en",
                  "promptLanguage": "zh",
                  "text": "hello",
                  "referenceAudioPath": "/tmp/ref.wav",
                  "referenceText": "你好"
                }
            """.trimIndent(),
        )

        assertEquals(200, response.statusCode)
        assertEquals("en", bridge.capturedRequest?.language)
        assertEquals("zh", bridge.capturedRequest?.promptLanguage)
    }

    @Test
    fun inferDefaultsMissingPromptLanguageToSynthesisLanguage() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "ja",
                  "text": "こんにちは",
                  "referenceAudioPath": "/tmp/ref.wav",
                  "referenceText": "参考テキスト"
                }
            """.trimIndent(),
        )

        assertEquals(200, response.statusCode)
        assertEquals("ja", bridge.capturedRequest?.promptLanguage)
    }

    @Test
    fun modelsEndpointReturnsCharacterList() {
        val server = LocalHttpApiServer(FakeBridge())

        val response = server.handleRequestForTest(method = "GET", path = "/models")

        assertEquals(200, response.statusCode)
        assertTrue(response.body.contains("\"models\":["))
        assertTrue(response.body.contains("\"id\":\"mansui\""))
    }

    @Test
    fun inferRejectsBackendThatIsNotEnabled() {
        val bridge = FakeBridge(
            enabledBackends = setOf(ExecutionBackend.CPU, ExecutionBackend.QNN),
        )
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "XNNPACK",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "/tmp/ref.wav",
                  "referenceText": "sample"
                }
            """.trimIndent(),
        )

        assertEquals(400, response.statusCode)
        assertTrue(response.body.contains("Backend is not enabled"))
    }

    @Test
    fun getInferUsesActiveDefaultsForMissingParams() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "GET",
            path = "/infer",
            queryParams = mapOf("text" to "hello world"),
        )

        assertEquals(200, response.statusCode)
        assertEquals("audio/wav", response.contentType)
        assertTrue(response.binaryBody?.isNotEmpty() == true)
        assertEquals(ExecutionBackend.CPU, bridge.capturedBackend)
        assertEquals("hello world", bridge.capturedRequest?.synthesisText)
        assertEquals("mansui", bridge.capturedRequest?.characterModel?.id)
        assertEquals("/tmp/ref.wav", bridge.capturedRequest?.referenceAudioPath)
        assertEquals("sample", bridge.capturedRequest?.referenceText)
        assertEquals("zh", bridge.capturedRequest?.language)
    }

    private class FakeBridge(
        private val enabledBackends: Set<ExecutionBackend> = ExecutionBackend.entries.toSet(),
    ) : LocalHttpApiServer.Bridge {
        private val models = CharacterModelCatalog.v2ProPlusFromCharacterIds(listOf("mansui"))

        var capturedBackend: ExecutionBackend? = null
        var capturedRequest: GenerationRequest? = null

        override fun currentState(): BackendServiceState = BackendServiceState(message = "Idle")

        override fun availableModels(): List<CharacterModel> = models

        override fun supportedBackends(): Set<ExecutionBackend> = enabledBackends

        override fun activeDefaults(): ActiveDefaults? = ActiveDefaults(
            backend = ExecutionBackend.CPU,
            modelId = "mansui",
            language = "zh",
            referenceAudioPath = "/tmp/ref.wav",
            referenceText = "sample",
            maxDecoderSteps = 500,
        )

        override fun infer(backend: ExecutionBackend, request: GenerationRequest): Boolean {
            capturedBackend = backend
            capturedRequest = request
            return true
        }

        override fun inferBlocking(backend: ExecutionBackend, request: GenerationRequest, timeoutMs: Long): InferResult {
            capturedBackend = backend
            capturedRequest = request
            val tempFile = File.createTempFile("tts_test", ".wav")
            tempFile.writeBytes(byteArrayOf(0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00))
            return InferResult.Success(tempFile)
        }
    }
}
