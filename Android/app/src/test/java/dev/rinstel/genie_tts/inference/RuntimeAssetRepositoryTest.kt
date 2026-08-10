package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
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

    @Test
    fun materializesHubertFp32ExternalWeightsWithoutRetainingFp16Data() {
        val root = createTempDir()
        val hubertDirectory = File(root, "RuntimeAssets/chinese-hubert-base").apply { mkdirs() }
        File(hubertDirectory, "chinese-hubert-base_weights_fp16.bin").writeBytes(
            byteArrayOf(0x00, 0x3c, 0x00, 0xc0.toByte()),
        )

        RuntimeAssetRepository(root).prepareRuntimeWeights()

        val output = File(hubertDirectory, "chinese-hubert-base_weights.bin")
        assertTrue(output.isFile)
        assertEquals(8L, output.length())
        val values = ByteBuffer.wrap(output.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        assertArrayEquals(floatArrayOf(1.0f, -2.0f), FloatArray(2) { values.get() }, 0.0f)
    }
}
