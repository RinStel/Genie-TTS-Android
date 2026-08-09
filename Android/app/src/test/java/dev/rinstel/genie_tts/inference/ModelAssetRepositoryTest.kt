package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
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
    fun rejectsPromptJsonWithoutNormalInsteadOfPickingAnotherEmotion() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        File(promptDirectory, "angry.wav").writeBytes(byteArrayOf(0))
        File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav.json").writeText(
            """
            {
              "Angry": {
                "wav": "angry.wav",
                "text": "angry text"
              }
            }
            """.trimIndent(),
        )

        try {
            ModelAssetRepository(filesRoot).findReferenceExample(modelV2ProPlus("mansui"))
            fail("Missing Normal reference must be reported")
        } catch (error: IllegalArgumentException) {
            assertTrue(error.message.orEmpty().contains("Normal"))
        }
    }

    @Test
    fun rejectsMalformedPromptJsonInsteadOfPickingFirstAudio() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        File(promptDirectory, "angry.wav").writeBytes(byteArrayOf(0))
        File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav.json").writeText(
            "{ \"Angry\": { \"wav\": \"angry.wav\" }",
        )

        try {
            ModelAssetRepository(filesRoot).findReferenceExample(modelV2ProPlus("mansui"))
            fail("Malformed prompt JSON must be reported")
        } catch (error: IllegalArgumentException) {
            assertTrue(error.message.orEmpty().contains("prompt_wav.json"))
        }
    }

    @Test
    fun rejectsNormalReferenceWithBlankText() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        File(promptDirectory, "normal.wav").writeBytes(byteArrayOf(0))
        File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav.json").writeText(
            """
            {
              "Normal": {
                "wav": "normal.wav",
                "text": "   "
              }
            }
            """.trimIndent(),
        )

        try {
            ModelAssetRepository(filesRoot).findReferenceExample(modelV2ProPlus("mansui"))
            fail("Blank Normal reference text must be reported")
        } catch (error: IllegalArgumentException) {
            assertTrue(error.message.orEmpty().contains("Normal reference"))
        }
    }

    @Test
    fun rejectsMissingPromptJsonInsteadOfPickingFirstAudio() {
        val filesRoot = createTempDir()
        val promptDirectory = File(filesRoot, "CharacterModels/v2ProPlus/mansui/prompt_wav").apply { mkdirs() }
        val wavFile = File(promptDirectory, "001-reference text.wav")
        wavFile.writeBytes(byteArrayOf(0))

        try {
            ModelAssetRepository(filesRoot).findReferenceExample(modelV2ProPlus("mansui"))
            fail("Missing prompt_wav.json must be reported")
        } catch (error: IllegalArgumentException) {
            assertTrue(error.message.orEmpty().contains("prompt_wav.json"))
        }
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

    @Test
    fun prefersAvailableFp16SynthesisGraphs() {
        val filesRoot = createTempDir()
        val modelFiles = RequiredModelFiles.v2ProPlusCharacter("mansui")
        val modelDirectory = File(filesRoot, "CharacterModels/${modelFiles.assetSubdirectory}").apply { mkdirs() }
        File(modelDirectory, "t2s_first_stage_decoder_fp16.onnx").writeBytes(byteArrayOf(0))
        File(modelDirectory, "t2s_stage_decoder_fp16.onnx").writeBytes(byteArrayOf(0))
        File(modelDirectory, "vits_fp16.onnx").writeBytes(byteArrayOf(0))

        val selected = modelFiles.sessionModelNames(modelDirectory, preferFp16 = true)

        assertTrue(selected.contains("t2s_first_stage_decoder_fp16.onnx"))
        assertTrue(selected.contains("t2s_stage_decoder_fp16.onnx"))
        assertTrue(selected.contains("vits_fp16.onnx"))
        assertFalse(selected.contains("t2s_first_stage_decoder_fp32.onnx"))
        assertFalse(selected.contains("t2s_stage_decoder_fp32.onnx"))
        assertFalse(selected.contains("vits_fp32.onnx"))
    }

    @Test
    fun prefersFp32SynthesisGraphsByDefault() {
        val filesRoot = createTempDir()
        val modelFiles = RequiredModelFiles.v2ProPlusCharacter("mansui")
        val modelDirectory = File(filesRoot, "CharacterModels/${modelFiles.assetSubdirectory}").apply { mkdirs() }
        modelFiles.sessionModels.forEach { fileName ->
            File(modelDirectory, fileName).writeBytes(byteArrayOf(0))
        }
        modelFiles.sessionModelVariants.values.forEach { fileName ->
            File(modelDirectory, fileName).writeBytes(byteArrayOf(0))
        }

        val selected = modelFiles.sessionModelNames(modelDirectory)

        assertTrue(selected.contains("t2s_first_stage_decoder_fp32.onnx"))
        assertTrue(selected.contains("t2s_stage_decoder_fp32.onnx"))
        assertTrue(selected.contains("vits_fp32.onnx"))
        assertFalse(selected.contains("t2s_first_stage_decoder_fp16.onnx"))
        assertFalse(selected.contains("t2s_stage_decoder_fp16.onnx"))
        assertFalse(selected.contains("vits_fp16.onnx"))
    }

    @Test
    fun inspectionAcceptsFp16SynthesisGraphsAsSessionReplacements() {
        val filesRoot = createTempDir()
        val modelFiles = RequiredModelFiles.v2ProPlusCharacter("mansui")
        val modelDirectory = File(filesRoot, "CharacterModels/${modelFiles.assetSubdirectory}").apply { mkdirs() }
        modelFiles.files.forEach { fileName ->
            File(modelDirectory, fileName).writeBytes(byteArrayOf(0))
        }
        modelFiles.sessionModels
            .filterNot { it in modelFiles.sessionModelVariants.keys }
            .forEach { fileName ->
                File(modelDirectory, fileName).writeBytes(byteArrayOf(0))
            }
        modelFiles.sessionModelVariants.values.forEach { fileName ->
            File(modelDirectory, fileName).writeBytes(byteArrayOf(0))
        }

        val inspection = ModelAssetRepository(filesRoot).inspect(modelFiles)

        assertFalse(inspection.missingFiles.contains("t2s_first_stage_decoder_fp32.onnx"))
        assertFalse(inspection.missingFiles.contains("t2s_stage_decoder_fp32.onnx"))
        assertFalse(inspection.missingFiles.contains("vits_fp32.onnx"))
    }

    @Test
    fun keepsPromptWeightsInFp16WhenInitializerManifestIsPresent() {
        val filesRoot = createTempDir()
        val modelFiles = RequiredModelFiles.v2ProPlusCharacter("mansui")
        val modelDirectory = File(filesRoot, "CharacterModels/${modelFiles.assetSubdirectory}").apply { mkdirs() }

        File(modelDirectory, "prompt_encoder_fp16.bin").writeBytes(byteArrayOf(0, 0))
        modelFiles.files
            .filter { it != "prompt_encoder_fp32.bin" }
            .forEach { fileName -> File(modelDirectory, fileName).writeBytes(byteArrayOf(0, 0)) }
        modelFiles.derivedFiles
            .filter { it.outputFile == "prompt_encoder_fp32.bin" }
            .forEach { derived -> File(modelDirectory, derived.sourceFile).writeBytes(byteArrayOf(0, 0)) }
        modelFiles.sessionModels.forEach { fileName ->
            File(modelDirectory, fileName).writeBytes(byteArrayOf(0))
        }
        File(modelDirectory, Fp16ExternalInitializers.MANIFEST_FILE).writeText(
            """
            {
              "version": 1,
              "logical_data_type": "float32",
              "weight_file": "prompt_encoder_fp16.bin",
              "initializers": []
            }
            """.trimIndent(),
        )

        ModelAssetRepository(filesRoot).installToPrivateStorage(modelFiles)

        assertFalse(File(modelDirectory, "prompt_encoder_fp32.bin").exists())
    }

    private fun modelV2ProPlus(id: String): CharacterModel =
        CharacterModelCatalog.v2ProPlusFromCharacterIds(listOf(id)).first()

    private fun modelV2(id: String): CharacterModel =
        CharacterModelCatalog.v2FromCharacterIds(listOf(id)).first()
}
