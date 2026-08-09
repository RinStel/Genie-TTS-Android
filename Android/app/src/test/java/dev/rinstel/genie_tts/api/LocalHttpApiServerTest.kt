package dev.rinstel.genie_tts.api

import dev.rinstel.genie_tts.BackendServiceState
import dev.rinstel.genie_tts.inference.CharacterModel
import dev.rinstel.genie_tts.inference.CharacterModelCatalog
import dev.rinstel.genie_tts.inference.ExecutionBackend
import dev.rinstel.genie_tts.inference.GenerationRequest
import java.io.File
import java.net.ServerSocket
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalHttpApiServerTest {
    @Test
    fun startsOnConfiguredValidPort() {
        val probe = ServerSocket(0)
        val port = probe.localPort
        probe.close()
        val server = LocalHttpApiServer(FakeBridge())

        try {
            server.start(port)

            val status = LocalHttpApiServer.currentStatus()
            assertTrue(status.running)
            assertEquals(port, status.port)
        } finally {
            server.stop()
        }
    }

    @Test
    fun rejectsPrivilegedPortBeforeAttemptingToBind() {
        val server = LocalHttpApiServer(FakeBridge())

        server.start(80)

        val status = LocalHttpApiServer.currentStatus()
        assertFalse(status.running)
        assertTrue(status.lastError?.contains("between 1024 and 65535") == true)
        server.stop()
    }

    @Test
    fun inferMapsTextFieldIntoGenerationRequest() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
                  "referenceText": "sample",
                  "maxDecoderSteps": 321
                }
            """.trimIndent(),
        )

        assertEquals(200, response.statusCode)
        assertEquals("audio/wav", response.contentType)
        assertTrue(response.binaryBody?.isNotEmpty() == true)
        assertEquals(ExecutionBackend.CPU, bridge.capturedBackend)
        assertEquals("hello", bridge.capturedRequest?.synthesisText)
        assertEquals(321, bridge.capturedRequest?.maxDecoderSteps)
    }

    @Test
    fun inferPreservesAuxiliaryReferenceOrder() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)
        val first = File.createTempFile("aux-first", ".wav")
        val second = File.createTempFile("aux-second", ".wav")

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
                  "referenceText": "sample",
                  "auxReferenceAudioPaths": ["${first.absolutePath.replace("\\", "\\\\")}", "${second.absolutePath.replace("\\", "\\\\")}"]
                }
            """.trimIndent(),
        )

        assertEquals(200, response.statusCode)
        assertEquals(
            listOf(first.absolutePath, second.absolutePath),
            bridge.capturedRequest?.auxiliaryReferenceAudioPaths,
        )
        first.delete()
        second.delete()
    }

    @Test
    fun inferRejectsMissingAuxiliaryReferenceBeforeBackendCall() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
                  "referenceText": "sample",
                  "auxReferenceAudioPaths": ["/path/that/does/not/exist.wav"]
                }
            """.trimIndent(),
        )

        assertEquals(400, response.statusCode)
        assertTrue(response.body.contains("not found"))
        assertEquals(null, bridge.capturedRequest)
    }

    @Test
    fun inferRejectsAuxiliaryReferenceThatDuplicatesPrimaryBeforeBackendCall() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
                  "referenceText": "sample",
                  "auxReferenceAudioPaths": ["${FakeBridge.primaryReferenceJsonPath}"]
                }
            """.trimIndent(),
        )

        assertEquals(400, response.statusCode)
        assertTrue(response.body.contains("primary"))
        assertEquals(null, bridge.capturedRequest)
    }

    @Test
    fun inferRejectsMissingPrimaryReferenceBeforeBackendCall() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}.missing",
                  "referenceText": "sample"
                }
            """.trimIndent(),
        )

        assertEquals(400, response.statusCode)
        assertTrue(response.body.contains("Reference audio not found"))
        assertEquals(null, bridge.capturedRequest)
    }

    @Test
    fun inferRejectsNonPositiveDecoderStepsBeforeBackendCall() {
        val bridge = FakeBridge()
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "mansui",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
                  "referenceText": "sample",
                  "maxDecoderSteps": 0
                }
            """.trimIndent(),
        )

        assertEquals(400, response.statusCode)
        assertTrue(response.body.contains("maxDecoderSteps must be positive"))
        assertEquals(null, bridge.capturedRequest)
    }

    @Test
    fun inferRejectsAuxiliaryReferencesForLegacyModelBeforeBackendCall() {
        val bridge = FakeBridge(
            models = CharacterModelCatalog.v2FromCharacterIds(listOf("legacy")),
        )
        val server = LocalHttpApiServer(bridge)
        val auxiliary = File.createTempFile("aux-legacy", ".wav")

        val response = server.handleRequestForTest(
            method = "POST",
            path = "/infer",
            body = """
                {
                  "backend": "CPU",
                  "modelId": "legacy",
                  "language": "zh",
                  "text": "hello",
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
                  "referenceText": "sample",
                  "auxReferenceAudioPaths": ["${auxiliary.absolutePath.replace(File.separatorChar, '/')}"]
                }
            """.trimIndent(),
        )

        assertEquals(400, response.statusCode)
        assertTrue(response.body, response.body.contains("does not support"))
        assertEquals(null, bridge.capturedRequest)
        auxiliary.delete()
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
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
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
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
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
            enabledBackends = setOf(ExecutionBackend.CPU),
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
                  "referenceAudioPath": "${FakeBridge.primaryReferenceJsonPath}",
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
        assertEquals(FakeBridge.primaryReferencePath, bridge.capturedRequest?.referenceAudioPath)
        assertEquals("sample", bridge.capturedRequest?.referenceText)
        assertEquals("zh", bridge.capturedRequest?.language)
    }

    @Test
    fun getInferUsesIndependentPromptLanguageFromActiveDefaults() {
        val bridge = FakeBridge(
            defaults = ActiveDefaults(
                backend = ExecutionBackend.CPU,
                modelId = "mansui",
                language = "en",
                promptLanguage = "zh",
                referenceAudioPath = FakeBridge.primaryReferencePath,
                referenceText = "sample",
                maxDecoderSteps = 500,
            ),
        )
        val server = LocalHttpApiServer(bridge)

        val response = server.handleRequestForTest(
            method = "GET",
            path = "/infer",
            queryParams = mapOf("text" to "hello world"),
        )

        assertEquals(200, response.statusCode)
        assertEquals("en", bridge.capturedRequest?.language)
        assertEquals("zh", bridge.capturedRequest?.promptLanguage)
    }

    private class FakeBridge(
        private val enabledBackends: Set<ExecutionBackend> = setOf(ExecutionBackend.CPU),
        private val models: List<CharacterModel> = CharacterModelCatalog.v2ProPlusFromCharacterIds(listOf("mansui")),
        private val defaults: ActiveDefaults = ActiveDefaults(
            backend = ExecutionBackend.CPU,
            modelId = "mansui",
            language = "zh",
            referenceAudioPath = FakeBridge.primaryReferencePath,
            referenceText = "sample",
            maxDecoderSteps = 500,
        ),
    ) : LocalHttpApiServer.Bridge {
        companion object {
            val primaryReferencePath: String = File.createTempFile("genie-primary", ".wav").apply {
                writeBytes(byteArrayOf(0))
                deleteOnExit()
            }.absolutePath
            val primaryReferenceJsonPath: String = primaryReferencePath.replace(File.separatorChar, '/')
        }

        var capturedBackend: ExecutionBackend? = null
        var capturedRequest: GenerationRequest? = null

        override fun currentState(): BackendServiceState = BackendServiceState(message = "Idle")

        override fun availableModels(): List<CharacterModel> = models

        override fun supportedBackends(): Set<ExecutionBackend> = enabledBackends

        override fun activeDefaults(): ActiveDefaults = defaults

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
