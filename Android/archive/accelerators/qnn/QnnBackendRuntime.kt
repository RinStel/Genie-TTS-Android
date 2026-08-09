package dev.rinstel.genie_tts.inference

import android.content.Context

class QnnBackendRuntime(
    context: Context,
    modelRepository: ModelAssetRepository,
    runtimeRepository: RuntimeAssetRepository,
    traceLogger: InferenceTraceLogger = InferenceTraceLogger.None,
) : BackendRuntime {
    override val backend: ExecutionBackend = ExecutionBackend.QNN

    private val inspection = QnnRuntimeSupport.inspect(context)

    override val runtimeLabel: String =
        if (inspection.isHtp) "ORT QNN HTP (NPU)" else "ORT QNN CPU"

    private val delegate = OrtSessionRuntime(
        modelRepository = modelRepository,
        runtimeRepository = runtimeRepository,
        backendEngine = OrtCpuBackend(
            backend = ExecutionBackend.QNN,
            configureSessionOptions = { options ->
                ModelRoleSessionOptions.configure(
                    options,
                    backend = ExecutionBackend.QNN,
                    role = InferenceModelRole.T2S,
                    qnnProviderOptions = QnnRuntimeSupport.providerOptions(inspection),
                )
            },
            traceLogger = traceLogger,
            configureRoleSessionOptions = { options, role ->
                ModelRoleSessionOptions.configure(
                    options,
                    backend = ExecutionBackend.QNN,
                    role = role,
                    qnnProviderOptions = QnnRuntimeSupport.providerOptions(inspection),
                )
            },
        ),
        featureExtractorFactory = { backendEngine ->
            OrtRuntimeFeatureExtractor(
                context = context,
                runtimeAssets = runtimeRepository,
                backend = backendEngine,
                traceLogger = traceLogger,
            )
        },
        traceLogger = traceLogger,
    )

    override fun generate(
        request: GenerationRequest,
        callbacks: BackendRuntimeCallbacks,
    ): GeneratedAudioFile {
        require(inspection.available) {
            buildString {
                append(inspection.message)
                if (inspection.discoveredLibraries.isNotEmpty()) {
                    append(" Discovered native libs: ")
                    append(inspection.discoveredLibraries.joinToString(", "))
                }
            }
        }
        return delegate.generate(request, callbacks)
    }

    override fun close() {
        delegate.close()
    }
}
