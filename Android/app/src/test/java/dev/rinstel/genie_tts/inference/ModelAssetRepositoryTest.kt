package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ModelAssetRepositoryTest {
    @Test
    fun loadsReferenceExampleFromPromptJson() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        File(promptDirectory, "sample.wav").writeBytes(byteArrayOf(0))
        File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav.json").writeText(
            """
            {
              "Normal": {
                "wav": "sample.wav",
                "text": "reference text"
              }
            }
            """.trimIndent(),
        )

        val repository = ModelAssetRepository(filesRoot)
        val example = repository.findReferenceExample(modelV2ProPlus("mansui"))

        assertNotNull(example)
        assertEquals("reference text", example?.referenceText)
        assertTrue(example?.audioFile?.absolutePath?.endsWith("sample.wav") == true)
    }

    @Test
    fun selectsNormalReferenceWhenAnotherEmotionComesFirst() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        File(promptDirectory, "angry.wav").writeBytes(byteArrayOf(0))
        File(promptDirectory, "normal.wav").writeBytes(byteArrayOf(0))
        File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav.json").writeText(
            """
            {
              "Angry": {
                "wav": "angry.wav",
                "text": "angry text"
              },
              "Normal": {
                "wav": "normal.wav",
                "text": "normal text"
              }
            }
            """.trimIndent(),
        )

        val example = ModelAssetRepository(filesRoot).findReferenceExample(modelV2ProPlus("mansui"))

        assertEquals("normal text", example?.referenceText)
        assertEquals("normal.wav", example?.audioFile?.name)
    }

    @Test
    fun decodesEscapedNormalReferenceTextWithoutMixingEmotionFields() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        File(promptDirectory, "normal.wav").writeBytes(byteArrayOf(0))
        File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav.json").writeText(
            """
            {
              "Happy": {
                "wav": "missing.wav",
                "text": "happy text"
              },
              "Normal": {
                "wav": "normal.wav",
                "text": "She said: \"hello\"\nthen left."
              }
            }
            """.trimIndent(),
        )

        val example = ModelAssetRepository(filesRoot).findReferenceExample(modelV2ProPlus("mansui"))

        assertEquals("She said: \"hello\"\nthen left.", example?.referenceText)
        assertEquals("normal.wav", example?.audioFile?.name)
    }

    @Test
    fun fallsBackToPromptFilenameWhenJsonIsMissing() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        val wavFile = File(promptDirectory, "001-reference text.wav")
        wavFile.writeBytes(byteArrayOf(0))

        val repository = ModelAssetRepository(filesRoot)
        val example = repository.findReferenceExample(modelV2ProPlus("mansui"))

        assertNotNull(example)
        assertEquals("reference text", example?.referenceText)
        assertEquals(wavFile.absolutePath, example?.audioFile?.absolutePath)
    }

    @Test
    fun loadsReferenceExampleFromV2CharacterDirectory() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2/Alice_CN/prompt_wav").apply { mkdirs() }
        File(promptDirectory, "sample.wav").writeBytes(byteArrayOf(0))
        File(filesRoot, "CharacterModels/v2/Alice_CN/prompt_wav.json").writeText(
            """
            {
              "Normal": {
                "wav": "sample.wav",
                "text": "reference text"
              }
            }
            """.trimIndent(),
        )

        val repository = ModelAssetRepository(filesRoot)
        val example = repository.findReferenceExample(modelV2("Alice_CN"))

        assertNotNull(example)
        assertEquals("reference text", example?.referenceText)
        assertTrue(example?.audioFile?.absolutePath?.endsWith("sample.wav") == true)
    }

    @Test
    fun reportsMissingSessionOnnxFiles() {
        val filesRoot = createTempDir()
        val modelFiles = RequiredModelFiles.v2ProPlusCharacter("mansui")
        val modelDirectory = File(filesRoot, "CharacterModels/${modelFiles.assetSubdirectory}").apply { mkdirs() }
        modelFiles.files.forEach { fileName ->
            File(modelDirectory, fileName).writeBytes(byteArrayOf(0))
        }

        val inspection = ModelAssetRepository(filesRoot).inspect(modelFiles)

        assertTrue(inspection.missingFiles.contains("t2s_first_stage_decoder_fp32.onnx"))
    }

    private fun modelV2ProPlus(id: String): CharacterModel =
        CharacterModelCatalog.v2ProPlusFromCharacterIds(listOf(id)).first()

    private fun modelV2(id: String): CharacterModel =
        CharacterModelCatalog.v2FromCharacterIds(listOf(id)).first()
}
