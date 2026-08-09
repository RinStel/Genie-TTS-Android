package dev.rinstel.genie_tts

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ResourcePackageImporterTest {
    @Test
    fun importsCompleteRuntimePackageIntoRuntimeAssets() {
        val filesRoot = createTempDir()
        val importer = ResourcePackageImporter(filesRoot)

        val result = importer.importRuntime(zipOf(runtimeEntries()))

        assertEquals(ResourcePackageImporter.PackageType.RUNTIME, result.packageType)
        assertEquals(6, result.fileCount)
        assertTrue(File(filesRoot, "RuntimeAssets/speaker_encoder.onnx").isFile)
        assertTrue(
            File(
                filesRoot,
                "RuntimeAssets/chinese-hubert-base/chinese-hubert-base_weights_fp16_manifest.json",
            ).isFile,
        )
    }

    @Test
    fun rejectsZipPathTraversalBeforeWritingOutsideStagingDirectory() {
        val filesRoot = createTempDir()
        val importer = ResourcePackageImporter(filesRoot)

        assertThrows<IllegalArgumentException> {
            importer.importRuntime(
                zipOf(
                    runtimeEntries() + ("RuntimeAssets/../escape.txt" to "escape"),
                ),
            )
        }

        assertFalse(File(filesRoot.parentFile, "escape.txt").exists())
        assertFalse(File(filesRoot, "RuntimeAssets").exists())
    }

    @Test
    fun rejectsRuntimePackageWithMissingRequiredFile() {
        val filesRoot = createTempDir()
        val oldRuntimeFile = File(filesRoot, "RuntimeAssets/old.marker").apply {
            parentFile.mkdirs()
            writeText("keep")
        }
        val importer = ResourcePackageImporter(filesRoot)

        assertThrows<IllegalArgumentException> {
            importer.importRuntime(zipOf(runtimeEntries().filterNot { it.first.endsWith("speaker_encoder.onnx") }))
        }

        assertTrue(oldRuntimeFile.isFile)
        assertEquals("keep", oldRuntimeFile.readText())
    }

    @Test
    fun replacesOnlyMatchingCharacterAndPreservesOtherCharacters() {
        val filesRoot = createTempDir()
        File(filesRoot, "CharacterModels/v2ProPlus/Alice/tts_models/old.onnx").apply {
            parentFile.mkdirs()
            writeText("old")
        }
        val otherCharacter = File(filesRoot, "CharacterModels/v2ProPlus/Bob/tts_models/model.onnx").apply {
            parentFile.mkdirs()
            writeText("bob")
        }
        val importer = ResourcePackageImporter(filesRoot)

        val result = importer.importCharacter(
            zipOf(
                listOf(
                    "CharacterModels/v2ProPlus/Alice/tts_models/model.onnx" to "new",
                    "CharacterModels/v2ProPlus/Alice/prompt_wav.json" to "{}",
                ),
            ),
        )

        assertEquals(listOf("v2ProPlus/Alice"), result.characterIds)
        assertFalse(File(filesRoot, "CharacterModels/v2ProPlus/Alice/tts_models/old.onnx").exists())
        assertEquals(
            "new",
            File(filesRoot, "CharacterModels/v2ProPlus/Alice/tts_models/model.onnx").readText(),
        )
        assertTrue(otherCharacter.isFile)
    }

    @Test
    fun rejectsCharacterPackageWithoutOnnxAndKeepsExistingCharacter() {
        val filesRoot = createTempDir()
        val oldModel = File(filesRoot, "CharacterModels/v2ProPlus/Alice/tts_models/model.onnx").apply {
            parentFile.mkdirs()
            writeText("old")
        }
        val importer = ResourcePackageImporter(filesRoot)

        assertThrows<IllegalArgumentException> {
            importer.importCharacter(
                zipOf(
                    listOf("CharacterModels/v2ProPlus/Alice/prompt_wav.json" to "{}"),
                ),
            )
        }

        assertEquals("old", oldModel.readText())
    }

    private fun runtimeEntries(): List<Pair<String, String>> = listOf(
        "RuntimeAssets/chinese-hubert-base/chinese-hubert-base.onnx" to "hubert",
        "RuntimeAssets/chinese-hubert-base/chinese-hubert-base_weights_fp16.bin" to "weights",
        "RuntimeAssets/chinese-hubert-base/chinese-hubert-base_weights_fp16_manifest.json" to "{}",
        "RuntimeAssets/speaker_encoder.onnx" to "speaker",
        "RuntimeAssets/roberta-wwm-ext-large-onnx/model.onnx" to "roberta",
        "RuntimeAssets/roberta-wwm-ext-large-onnx/roberta_tokenizer/tokenizer.json" to "{}",
    )

    private fun zipOf(entries: List<Pair<String, String>>): ByteArrayInputStream {
        val bytes = ByteArrayOutputStream()
        ZipOutputStream(bytes).use { zip ->
            entries.forEach { (name, content) ->
                zip.putNextEntry(ZipEntry(name))
                zip.write(content.toByteArray())
                zip.closeEntry()
            }
        }
        return ByteArrayInputStream(bytes.toByteArray())
    }

    private inline fun <reified T : Throwable> assertThrows(block: () -> Unit) {
        try {
            block()
        } catch (error: Throwable) {
            if (error is T) return
            throw error
        }
        throw AssertionError("Expected ${T::class.java.simpleName}")
    }
}
