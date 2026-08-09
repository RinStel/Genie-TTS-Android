package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class RuntimeAssetRepositoryTest {
    @Test
    fun reportsMissingRequiredRuntimeAssets() {
        val root = createTempDir()
        val repository = RuntimeAssetRepository(root)

        val inspection = repository.inspect()

        assertTrue(inspection.missingFiles.contains("chinese-hubert-base/chinese-hubert-base.onnx"))
        assertTrue(inspection.missingFiles.contains("speaker_encoder.onnx"))
        assertEquals(File(root, "RuntimeAssets").absolutePath, repository.runtimeRoot().absolutePath)
    }
}
