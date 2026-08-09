package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class RobertaFeatureProviderTest {
    @Test
    fun resolvesPythonDownloadedRuntimeLayout() {
        val root = createTempDir()
        val directory = File(root, "roberta-wwm-ext-large-onnx").apply { mkdirs() }
        val model = File(directory, "model.onnx").apply { writeBytes(byteArrayOf(1)) }
        val tokenizer = File(directory, "tokenizer.json").apply { writeText("{}") }

        val assets = RobertaFeatureProvider.resolveAssetFiles(root)

        assertNotNull(assets)
        assertTrue(assets?.modelFile?.canonicalPath.equals(model.canonicalPath, ignoreCase = true))
        assertTrue(assets?.tokenizerFile?.canonicalPath.equals(tokenizer.canonicalPath, ignoreCase = true))
    }

    @Test
    fun resolvesLegacyNestedTokenizerLayout() {
        val root = createTempDir()
        val directory = File(root, "RoBERTa").apply { mkdirs() }
        val model = File(directory, "RoBERTa.onnx").apply { writeBytes(byteArrayOf(1)) }
        val tokenizer = File(directory, "roberta_tokenizer/tokenizer.json").apply {
            requireNotNull(parentFile).mkdirs()
            writeText("{}")
        }

        val assets = RobertaFeatureProvider.resolveAssetFiles(root)

        assertNotNull(assets)
        assertTrue(assets?.modelFile?.canonicalPath.equals(model.canonicalPath, ignoreCase = true))
        assertTrue(assets?.tokenizerFile?.canonicalPath.equals(tokenizer.canonicalPath, ignoreCase = true))
    }
}
