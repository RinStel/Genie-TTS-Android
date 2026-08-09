package dev.rinstel.genie_tts.inference

import ai.onnxruntime.OrtSession

object XnnpackRuntimeSupport {
    data class InspectionResult(
        val available: Boolean,
        val message: String,
    )

    fun inspect(): InspectionResult {
        var options: OrtSession.SessionOptions? = null
        return try {
            // Probe EP registration only; parity issues are handled by BackendCatalog/MainActivity.
            options = OrtSession.SessionOptions()
            OrtCpuBackend.configureXnnpackSessionOptions(options)
            InspectionResult(
                available = true,
                message = "Ready",
            )
        } catch (error: Throwable) {
            InspectionResult(
                available = false,
                message = error.message ?: error.javaClass.simpleName,
            )
        } finally {
            runCatching { options?.close() }
        }
    }
}
