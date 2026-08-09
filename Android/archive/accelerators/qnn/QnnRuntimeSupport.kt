package dev.rinstel.genie_tts.inference

import android.content.Context
import android.os.Build
import java.io.File

object QnnRuntimeSupport {
    private const val ONNX_RUNTIME_LIBRARY = "onnxruntime"
    private val preferredBackendLibraries = listOf(
        "QnnHtp",
        "QnnCpu",
    )

    private fun androidLibraryFileName(baseName: String): String = "lib$baseName.so"

    data class InspectionResult(
        val available: Boolean,
        val message: String,
        val runtimeLoaded: Boolean,
        val backendLibraryBaseName: String?,
        val backendLibraryName: String?,
        val discoveredLibraries: List<String>,
        val isHtp: Boolean,
    )

    fun inspect(context: Context): InspectionResult {
        val nativeRoot = File(context.applicationInfo.nativeLibraryDir.orEmpty())
        val discovered = nativeRoot.listFiles()
            ?.filter { it.isFile }
            ?.map(File::getName)
            ?.sorted()
            .orEmpty()
        val runtimeLoaded = runCatching {
            System.loadLibrary(ONNX_RUNTIME_LIBRARY)
            true
        }.getOrDefault(false)
        val backendLibraryBaseName = preferredBackendLibraries.firstOrNull { libraryName ->
            runCatching {
                System.loadLibrary(libraryName)
                true
            }.getOrDefault(false)
        }

        return inspectLibraries(
            discoveredLibraries = discovered,
            runtimeLoaded = runtimeLoaded,
            loadedBackendLibraryBaseName = backendLibraryBaseName,
        )
    }

    internal fun inspectLibraries(
        discoveredLibraries: List<String>,
        runtimeLoaded: Boolean,
        loadedBackendLibraryBaseName: String?,
    ): InspectionResult {
        val backendLibraryName = loadedBackendLibraryBaseName?.let(::androidLibraryFileName)
        val isHtp = loadedBackendLibraryBaseName == "QnnHtp"
        return when {
            !runtimeLoaded -> InspectionResult(
                available = false,
                message = "Failed to load onnxruntime via System.loadLibrary.",
                runtimeLoaded = false,
                backendLibraryBaseName = loadedBackendLibraryBaseName,
                backendLibraryName = backendLibraryName,
                discoveredLibraries = discoveredLibraries,
                isHtp = false,
            )
            backendLibraryName == null -> InspectionResult(
                available = false,
                message = "Failed to load a QNN backend via System.loadLibrary. Expected one of: ${preferredBackendLibraries.joinToString(", ")}",
                runtimeLoaded = true,
                backendLibraryBaseName = null,
                backendLibraryName = null,
                discoveredLibraries = discoveredLibraries,
                isHtp = false,
            )
            else -> InspectionResult(
                available = true,
                message = if (isHtp) {
                    "QNN HTP (NPU) runtime libraries loaded through System.loadLibrary."
                } else {
                    "QNN CPU fallback runtime libraries loaded through System.loadLibrary."
                },
                runtimeLoaded = true,
                backendLibraryBaseName = loadedBackendLibraryBaseName,
                backendLibraryName = backendLibraryName,
                discoveredLibraries = discoveredLibraries,
                isHtp = isHtp,
            )
        }
    }

    fun providerOptions(
        inspection: InspectionResult,
    ): Map<String, String> {
        // Provider options are separate from ORT session config entries such
        // as ep.context_enable; unsupported cache keys must not be invented.
        val backendLibraryName = requireNotNull(inspection.backendLibraryName) {
            "QNN backend library is unavailable."
        }
        val options = linkedMapOf(
            "backend_path" to backendLibraryName,
            "profiling_level" to "off",
            "rpc_control_latency" to "0",
        )

        if (inspection.isHtp) {
            options["htp_performance_mode"] = "burst"
            options["qnn_context_priority"] = "normal_high"
            options["htp_graph_finalization_optimization_mode"] = "3"
            options["enable_htp_fp16_precision"] = "1"
            options["enable_htp_spill_fill_buffer"] = "1"
            options["vtcm_mb"] = "8"
            options["device_id"] = "0"
            if (inspection.discoveredLibraries.contains("libcdsprpc.so")) {
                options["enable_htp_shared_memory_allocator"] = "1"
            }

            val socModel = socModelProviderOption(detectSocModel())
            if (socModel != null) {
                options["soc_model"] = socModel
            }

        }

        return options
    }

    internal fun socModelProviderOption(rawSocModel: String?): String? {
        val value = rawSocModel?.trim().orEmpty()
        // QNN parses soc_model as a numeric SDK model id; Android SOC_MODEL is often "SM8650".
        return value.takeIf { it.isNotEmpty() && it.all(Char::isDigit) }
    }

    private fun detectSocModel(): String? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val socModel = Build.SOC_MODEL
                if (socModel.isNotBlank()) socModel else null
            } else {
                null
            }
        } catch (error: Exception) {
            null
        }
    }
}
