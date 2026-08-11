package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BackendCatalogTest {
    @Test
    fun exposesQnnOnlyWhenTheFullRuntimeIsAvailable() {
        val options = BackendCatalog.build(
            qnnAvailable = true,
            xnnpackAvailable = true,
        )

        assertEquals(
            listOf(ExecutionBackend.CPU, ExecutionBackend.QNN),
            options.map { it.backend },
        )
        assertTrue(options.all { it.enabled })
    }

    @Test
    fun keepsCpuAsDefaultWhenAcceleratorFlagsArePassed() {
        val options = BackendCatalog.build(
            qnnAvailable = true,
            xnnpackAvailable = true,
        )

        assertEquals(ExecutionBackend.CPU, BackendCatalog.defaultOption(options).backend)
    }

    @Test
    fun usesCpuAsDefaultWhenQnnIsUnavailable() {
        val options = BackendCatalog.build(
            qnnAvailable = false,
            qnnStatus = "QNN unavailable.",
            xnnpackAvailable = true,
        )

        assertEquals(ExecutionBackend.CPU, BackendCatalog.defaultOption(options).backend)
    }

    @Test
    fun keepsCpuLabelShort() {
        val options = BackendCatalog.build(
            qnnAvailable = true,
            xnnpackAvailable = true,
        )

        assertEquals("CPU", options.first { it.backend == ExecutionBackend.CPU }.label)
    }

    @Test
    fun preservesOriginalCharacterIdCaseInDisplayNames() {
        val v2ProPlus = CharacterModelCatalog.v2ProPlusFromCharacterIds(listOf("zH_Mix", "mansui"))
        val v2 = CharacterModelCatalog.v2FromCharacterIds(listOf("Alice_CN"))

        assertEquals(listOf("mansui", "zH_Mix"), v2ProPlus.map { it.displayName })
        assertEquals("Alice_CN", v2.single().displayName)
    }
}
