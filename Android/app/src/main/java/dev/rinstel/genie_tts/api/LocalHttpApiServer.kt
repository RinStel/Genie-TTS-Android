package dev.rinstel.genie_tts.api

import dev.rinstel.genie_tts.BackendServiceState
import dev.rinstel.genie_tts.inference.CharacterModel
import dev.rinstel.genie_tts.inference.ExecutionBackend
import dev.rinstel.genie_tts.inference.GenerationRequest
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LocalHttpApiServer(
    private val bridge: Bridge,
) {
    interface Bridge {
        fun currentState(): BackendServiceState
        fun availableModels(): List<CharacterModel>
        fun supportedBackends(): Set<ExecutionBackend>
        fun supportsAuxiliaryReferences(model: CharacterModel): Boolean =
            model.modelFiles.maxAuxiliaryReferenceCount > 0 &&
                model.modelFiles.optionalSessionModels.contains("prompt_encoder_multi_fp32.onnx")
        fun activeDefaults(): ActiveDefaults?
        fun infer(backend: ExecutionBackend, request: GenerationRequest): Boolean
        fun inferBlocking(backend: ExecutionBackend, request: GenerationRequest, timeoutMs: Long): InferResult
    }

    @Volatile
    private var serverSocket: ServerSocket? = null

    @Volatile
    private var acceptThread: Thread? = null

    private val requestExecutor: ExecutorService = Executors.newCachedThreadPool()

    fun start(port: Int) {
        stop()
        if (port !in MIN_API_PORT..MAX_API_PORT) {
            updateStatus(
                ApiServerStatus(
                    running = false,
                    port = port,
                    lastError = "API port must be between $MIN_API_PORT and $MAX_API_PORT.",
                ),
            )
            return
        }
        try {
            val socket = ServerSocket()
            socket.reuseAddress = true
            socket.bind(InetSocketAddress(InetAddress.getByName(LOOPBACK_ADDRESS), port))
            serverSocket = socket
            updateStatus(ApiServerStatus(running = true, port = port))
            acceptThread = thread(
                start = true,
                isDaemon = true,
                name = "GenieTtsLocalHttpApi",
            ) {
                while (!socket.isClosed) {
                    val client = runCatching { socket.accept() }.getOrNull() ?: break
                    requestExecutor.execute {
                        handleClient(client)
                    }
                }
            }
        } catch (error: Throwable) {
            updateStatus(
                ApiServerStatus(
                    running = false,
                    port = port,
                    lastError = error.message ?: error.javaClass.simpleName,
                ),
            )
        }
    }

    fun stop() {
        runCatching { serverSocket?.close() }
        runCatching { acceptThread?.interrupt() }
        acceptThread = null
        serverSocket = null
        val previousPort = status.port
        updateStatus(ApiServerStatus(running = false, port = previousPort))
    }

    private fun handleClient(socket: Socket) {
        socket.use { client ->
            if (!client.inetAddress.isLoopbackAddress) {
                writeResponse(client, HttpResponse(403, errorJson("Loopback access only.")))
                return
            }

            val request = runCatching { readRequest(client) }.getOrElse { error ->
                writeResponse(client, HttpResponse(400, errorJson(error.message ?: "Bad request.")))
                return
            }
            val response = route(
                method = request.method,
                path = request.path,
                body = request.body,
                queryParams = request.queryParams,
            )
            writeResponse(client, response)
        }
    }

    private fun route(method: String, path: String, body: String, queryParams: Map<String, String>): HttpResponse =
        when {
            method == "GET" && path == "/health" -> HttpResponse(
                200,
                jsonObject(
                    "ok" to true,
                    "address" to LOOPBACK_ADDRESS,
                    "port" to status.port,
                ),
            )

            method == "GET" && path == "/status" -> HttpResponse(
                200,
                jsonObject(
                    "ok" to true,
                    "state" to RawJson(bridge.currentState().toJsonString()),
                ),
            )

            method == "GET" && path == "/models" -> HttpResponse(
                200,
                jsonObject(
                    "ok" to true,
                    "models" to RawJson(bridge.availableModels().toJsonString()),
                ),
            )

            method == "GET" && path == "/result/latest" -> HttpResponse(
                200,
                jsonObject(
                    "ok" to true,
                    "latestOutputFilePath" to bridge.currentState().latestOutputFilePath,
                ),
            )

            method == "GET" && path == "/infer" -> inferGet(queryParams)
            method == "POST" && path == "/infer" -> infer(body)

            else -> HttpResponse(404, errorJson("Route not found."))
        }

    internal fun handleRequestForTest(
        method: String,
        path: String,
        body: String = "",
        queryParams: Map<String, String> = emptyMap(),
    ): HttpResponse = route(method, path, body, queryParams)

    private fun infer(body: String): HttpResponse {
        val payload = runCatching { parseInferPayload(body) }.getOrElse {
            return HttpResponse(400, errorJson("Invalid JSON body."))
        }
        return inferWithPayload(payload)
    }

    private fun inferGet(params: Map<String, String>): HttpResponse {
        val defaults = bridge.activeDefaults()
        val payload = InferPayload(
            backendName = params["backend"] ?: defaults?.backend?.name ?: "",
            modelId = params["modelId"] ?: defaults?.modelId ?: "",
            language = params["language"] ?: defaults?.language ?: "",
            promptLanguage = params["promptLanguage"] ?: params["language"] ?: defaults?.promptLanguage ?: "",
            text = params["text"] ?: "",
            referenceAudioPath = params["referenceAudioPath"] ?: defaults?.referenceAudioPath ?: "",
            referenceText = params["referenceText"] ?: defaults?.referenceText ?: "",
            maxDecoderSteps = params["maxDecoderSteps"]?.toIntOrNull() ?: defaults?.maxDecoderSteps ?: 500,
            auxiliaryReferenceAudioPaths = params["auxReferenceAudioPaths"]
                ?.split('|')
                ?.map(String::trim)
                ?.filter(String::isNotBlank)
                ?: defaults?.auxiliaryReferenceAudioPaths.orEmpty(),
        )
        return inferWithPayload(payload)
    }

    private fun inferWithPayload(payload: InferPayload): HttpResponse {
        val backend = payload.backendName
            .ifBlank { ExecutionBackend.CPU.name }
            .let { value -> ExecutionBackend.entries.firstOrNull { it.name.equals(value, ignoreCase = true) } }
            ?: return HttpResponse(400, errorJson("Unknown backend."))
        if (backend !in bridge.supportedBackends()) {
            return HttpResponse(400, errorJson("Backend is not enabled."))
        }

        val model = bridge.availableModels().firstOrNull { it.id == payload.modelId }
            ?: return HttpResponse(400, errorJson("Unknown modelId."))
        if (
            payload.language.isBlank() ||
            payload.text.isBlank() ||
            payload.referenceAudioPath.isBlank() ||
            payload.referenceText.isBlank()
        ) {
            return HttpResponse(400, errorJson("language, text, referenceAudioPath, and referenceText are required."))
        }
        validateReferenceAudioPath(payload.referenceAudioPath)?.let { error ->
            return HttpResponse(400, errorJson(error))
        }
        if (payload.maxDecoderSteps <= 0) {
            return HttpResponse(400, errorJson("maxDecoderSteps must be positive."))
        }
        validateAuxiliaryReferencePaths(
            paths = payload.auxiliaryReferenceAudioPaths,
            primaryPath = payload.referenceAudioPath,
        )?.let { error ->
            return HttpResponse(400, errorJson(error))
        }
        validateAuxiliaryReferenceCapability(model, payload.auxiliaryReferenceAudioPaths)?.let { error ->
            return HttpResponse(400, errorJson(error))
        }

        val result = bridge.inferBlocking(
            backend = backend,
            request = GenerationRequest(
                characterModel = model,
                language = payload.language,
                promptLanguage = payload.promptLanguage,
                synthesisText = payload.text,
                referenceAudioPath = payload.referenceAudioPath,
                referenceText = payload.referenceText,
                maxDecoderSteps = payload.maxDecoderSteps,
                auxiliaryReferenceAudioPaths = payload.auxiliaryReferenceAudioPaths,
            ),
            timeoutMs = INFER_TIMEOUT_MS,
        )
        return when (result) {
            is InferResult.Success -> {
                val wavBytes = runCatching { result.outputFile.readBytes() }.getOrElse {
                    return HttpResponse(500, errorJson("Failed to read output file: ${it.message}"))
                }
                HttpResponse(
                    statusCode = 200,
                    body = "",
                    contentType = "audio/wav",
                    binaryBody = wavBytes,
                )
            }
            is InferResult.Error -> HttpResponse(500, errorJson(result.message))
            InferResult.Busy -> HttpResponse(409, errorJson("Inference is already running."))
            InferResult.Timeout -> HttpResponse(504, errorJson("Inference timed out."))
        }
    }

    private fun validateReferenceAudioPath(path: String): String? {
        val file = File(path)
        val canonical = runCatching { file.canonicalFile }.getOrElse { file.absoluteFile }
        if (!canonical.isFile) {
            return "Reference audio not found: $path"
        }
        return null
    }

    private fun validateAuxiliaryReferencePaths(
        paths: List<String>,
        primaryPath: String? = null,
    ): String? {
        val canonicalPrimaryPath = primaryPath?.let { rawPath ->
            val file = File(rawPath)
            runCatching { file.canonicalFile }.getOrElse { file.absoluteFile }.absolutePath
        }
        val canonicalPaths = paths.map { rawPath ->
            if (rawPath.isBlank()) {
                return "Auxiliary reference paths cannot be blank."
            }
            val file = File(rawPath)
            val canonical = runCatching { file.canonicalFile }.getOrElse { file.absoluteFile }
            if (!canonical.isFile) {
                return "Auxiliary reference audio not found: $rawPath"
            }
            if (canonical.absolutePath == canonicalPrimaryPath) {
                return "Auxiliary reference path duplicates the primary reference."
            }
            canonical.absolutePath
        }
        if (canonicalPaths.distinct().size != canonicalPaths.size) {
            return "Auxiliary reference paths must be unique."
        }
        return null
    }

    private fun validateAuxiliaryReferenceCapability(
        model: CharacterModel,
        paths: List<String>,
    ): String? {
        if (paths.isEmpty()) return null
        val modelFiles = model.modelFiles
        if (
            modelFiles.maxAuxiliaryReferenceCount <= 0 ||
            !modelFiles.optionalSessionModels.contains("prompt_encoder_multi_fp32.onnx")
        ) {
            return "The selected model does not support auxiliary references."
        }
        if (!bridge.supportsAuxiliaryReferences(model)) {
            return "The selected model does not have the multi-reference prompt encoder installed."
        }
        if (paths.size > modelFiles.maxAuxiliaryReferenceCount) {
            return "At most ${modelFiles.maxAuxiliaryReferenceCount} auxiliary references are supported."
        }
        return null
    }

    private fun readRequest(socket: Socket): ParsedRequest {
        val input = BufferedInputStream(socket.getInputStream())
        val requestLine = readAsciiLine(input)
        val parts = requestLine.split(" ")
        require(parts.size >= 2) { "Malformed request line." }
        val method = parts[0].uppercase()
        val rawPath = parts[1]
        val path = rawPath.substringBefore('?')
        val queryString = rawPath.substringAfter('?', "")
        val queryParams = parseQueryParams(queryString)
        val headers = linkedMapOf<String, String>()
        while (true) {
            val line = readAsciiLine(input)
            if (line.isBlank()) {
                break
            }
            val separator = line.indexOf(':')
            if (separator > 0) {
                headers[line.substring(0, separator).trim().lowercase()] = line.substring(separator + 1).trim()
            }
        }
        val contentLength = headers["content-length"]?.toIntOrNull()?.coerceAtLeast(0) ?: 0
        val bodyBytes = input.readNBytes(contentLength)
        return ParsedRequest(
            method = method,
            path = path,
            body = String(bodyBytes, StandardCharsets.UTF_8),
            queryParams = queryParams,
        )
    }

    private fun parseQueryParams(query: String): Map<String, String> {
        if (query.isBlank()) return emptyMap()
        return query.split('&')
            .mapNotNull { pair ->
                val idx = pair.indexOf('=')
                if (idx > 0) {
                    val key = java.net.URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8)
                    val value = java.net.URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8)
                    key to value
                } else if (pair.isNotBlank()) {
                    java.net.URLDecoder.decode(pair, StandardCharsets.UTF_8) to ""
                } else {
                    null
                }
            }
            .toMap()
    }

    private fun writeResponse(socket: Socket, response: HttpResponse) {
        val output = BufferedOutputStream(socket.getOutputStream())
        val bodyBytes = response.binaryBody ?: response.body.toByteArray(StandardCharsets.UTF_8)
        val statusText = when (response.statusCode) {
            200 -> "OK"
            202 -> "Accepted"
            400 -> "Bad Request"
            403 -> "Forbidden"
            404 -> "Not Found"
            409 -> "Conflict"
            500 -> "Internal Server Error"
            504 -> "Gateway Timeout"
            else -> "Error"
        }
        val header = buildString {
            append("HTTP/1.1 ${response.statusCode} $statusText\r\n")
            append("Content-Type: ${response.contentType}\r\n")
            if (response.binaryBody != null) {
                append("Content-Disposition: inline; filename=\"tts_output.wav\"\r\n")
            }
            append("Content-Length: ${bodyBytes.size}\r\n")
            append("Connection: close\r\n")
            append("\r\n")
        }
        output.write(header.toByteArray(StandardCharsets.UTF_8))
        output.write(bodyBytes)
        output.flush()
    }

    private fun readAsciiLine(input: BufferedInputStream): String {
        val bytes = mutableListOf<Byte>()
        while (true) {
            val next = input.read()
            require(next >= 0) { "Unexpected end of stream." }
            if (next == '\n'.code) {
                break
            }
            if (next != '\r'.code) {
                bytes += next.toByte()
            }
        }
        return String(bytes.toByteArray(), StandardCharsets.UTF_8)
    }

    private fun errorJson(message: String): String =
        jsonObject(
            "ok" to false,
            "error" to message,
        )

    private data class ParsedRequest(
        val method: String,
        val path: String,
        val body: String,
        val queryParams: Map<String, String>,
    )

    companion object {
        private const val LOOPBACK_ADDRESS = "127.0.0.1"
        private const val MIN_API_PORT = 1024
        private const val MAX_API_PORT = 65535
        private const val INFER_TIMEOUT_MS = 300_000L

        @Volatile
        private var status: ApiServerStatus = ApiServerStatus(running = false, port = 0)

        fun currentStatus(): ApiServerStatus = status

        private fun updateStatus(value: ApiServerStatus) {
            status = value
        }
    }
}
