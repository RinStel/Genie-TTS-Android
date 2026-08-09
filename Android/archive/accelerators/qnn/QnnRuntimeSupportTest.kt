package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class QnnRuntimeSupportTest {
    @Test
    fun reportsQnnAvailableWhenOrtAndBackendLibrariesAreLoaded() {
        val inspection = QnnRuntimeSupport.inspectLibraries(
            discoveredLibraries = emptyList(),
            runtimeLoaded = true,
            loadedBackendLibraryBaseName = "QnnHtp",
        )

        assertTrue(inspection.available)
        assertEquals("libQnnHtp.so", inspection.backendLibraryName)
        assertEquals("libQnnHtp.so", QnnRuntimeSupport.providerOptions(inspection)["backend_path"])
    }

    @Test
    fun enablesHtpProviderOptionsForNpuBackend() {
        val inspection = QnnRuntimeSupport.inspectLibraries(
            discoveredLibraries = listOf("libcdsprpc.so"),
            runtimeLoaded = true,
            loadedBackendLibraryBaseName = "QnnHtp",
        )

        val options = QnnRuntimeSupport.providerOptions(inspection)

        assertEquals("burst", options["htp_performance_mode"])
        assertEquals("normal_high", options["qnn_context_priority"])
        assertEquals("3", options["htp_graph_finalization_optimization_mode"])
        assertEquals("1", options["enable_htp_fp16_precision"])
        assertEquals("1", options["enable_htp_spill_fill_buffer"])
        assertEquals("1", options["enable_htp_shared_memory_allocator"])
        assertFalse(options.containsKey("htp_accuracy_level"))
        assertFalse(options.keys.any { it.startsWith("qnn_context_cache_") })
    }

    @Test
    fun skipsSharedMemoryAllocatorWhenCdspRpcLibraryIsUnavailable() {
        val inspection = QnnRuntimeSupport.inspectLibraries(
            discoveredLibraries = emptyList(),
            runtimeLoaded = true,
            loadedBackendLibraryBaseName = "QnnHtp",
        )

        val options = QnnRuntimeSupport.providerOptions(inspection)

        assertFalse(options.containsKey("enable_htp_shared_memory_allocator"))
    }

    @Test
    fun reportsMissingOrtWhenRuntimeLoadFails() {
        val inspection = QnnRuntimeSupport.inspectLibraries(
            discoveredLibraries = listOf("libQnnHtp.so"),
            runtimeLoaded = false,
            loadedBackendLibraryBaseName = "QnnHtp",
        )

        assertFalse(inspection.available)
        assertTrue(inspection.message.contains("onnxruntime"))
    }

    @Test
    fun ignoresNonNumericAndroidSocModelValues() {
        assertNull(QnnRuntimeSupport.socModelProviderOption("SM8650"))
    }

    @Test
    fun preservesNumericQnnSocModelValues() {
        assertEquals("43", QnnRuntimeSupport.socModelProviderOption("43"))
    }
}
