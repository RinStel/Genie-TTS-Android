package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChineseTextFeatureExtractorTest {
    private class RecordingBertProvider : BertFeatureProvider {
        var recordedWord2Ph: List<Int> = emptyList()

        override fun computeBertFeatures(
            normalizedText: String,
            word2ph: List<Int>,
            numPhones: Int,
        ): FloatTensorData {
            recordedWord2Ph = word2ph
            return zeroBert(numPhones)
        }
    }

    private val resources = ChineseG2pResources(
        polyphonicPinyin = mapOf(
            "哪吒" to listOf("ne2", "zha1"),
            "你" to listOf("ni3"),
            "好" to listOf("hao3"),
        ),
        opencpopByPinyin = mapOf(
            "ne" to "n e",
            "zha" to "zh a",
            "ni" to "n i",
            "hao" to "h ao",
        ),
        singleCharPinyin = mapOf(
            "你" to "ni3",
            "好" to "hao3",
        ),
    )

    @Test
    fun mapsPolyphonicPhraseToPhonePairs() {
        val extractor = ChineseTextFeatureExtractor(resources)

        val features = extractor.extract("哪吒")

        assertEquals(listOf("n", "e2", "zh", "a1"), features.symbols)
        assertArrayEquals(longArrayOf(1L, 4L), features.phoneIds.shape)
        assertArrayEquals(longArrayOf(4L, 1024L), features.bert.shape)
    }

    @Test
    fun normalizesAndPrefixesOnlySynthesisText() {
        val extractor = ChineseTextFeatureExtractor(resources)

        val synthesis = extractor.extract(GenieTextConventions.prepareSynthesisText("zh", "你好"))
        val reference = extractor.extract("你好")

        assertTrue(synthesis.normalizedText.startsWith("."))
        assertEquals("你好", reference.normalizedText)
    }
    @Test
    fun usesContextualSegmentationForRepeatedPolyphones() {
        val contextualResources = resources.copy(
            polyphonicPinyin = resources.polyphonicPinyin + mapOf(
                "\u94f6\u884c" to listOf("yin2", "hang2"),
            ),
            opencpopByPinyin = resources.opencpopByPinyin + mapOf(
                "yin" to "y in",
                "hang" to "h ang",
                "xing" to "x ing",
                "zhang" to "zh ang",
            ),
            singleCharPinyin = mapOf(
                "银" to "yin2",
                "行" to "xing2",
                "长" to "zhang3",
            ),
            wordFrequencies = mapOf("银行" to 10, "银行行长" to 3),
            wordTags = mapOf("银行行长" to "n"),
            wordPinyin = mapOf(
                "银行行长" to listOf("yin2", "hang2", "xing2", "zhang3"),
            ),
        )

        val features = ChineseTextFeatureExtractor(contextualResources).extract("银行行长")

        assertEquals(
            listOf("y", "in2", "h", "ang2", "x", "ing5", "zh", "ang3"),
            features.symbols,
        )
    }

    @Test
    fun usesGeneratedWordEntriesForProductionSegmentation() {
        val productionResources = resources.copy(
            wordEntries = mapOf(
                "银行" to ChineseWordEntry(
                    frequency = 10,
                    tag = "n",
                    pinyinCsv = "yin2,hang2",
                ),
                "行长" to ChineseWordEntry(
                    frequency = 3,
                    tag = "n",
                    pinyinCsv = "xing2,zhang3",
                ),
            ),
            wordFrequencies = emptyMap(),
            wordTags = emptyMap(),
            wordPinyin = emptyMap(),
            polyphonicPinyin = resources.polyphonicPinyin + mapOf(
                "银" to listOf("yin2"),
                "行" to listOf("xing2"),
                "长" to listOf("zhang3"),
            ),
            opencpopByPinyin = resources.opencpopByPinyin + mapOf(
                "yin" to "y in",
                "hang" to "h ang",
                "xing" to "x ing",
                "zhang" to "zh ang",
            ),
            singleCharPinyin = mapOf(
                "银" to "yin2",
                "行" to "xing2",
                "长" to "zhang3",
            ),
        )

        val features = ChineseTextFeatureExtractor(productionResources).extract("银行行长")

        assertEquals(
            listOf("y", "in2", "h", "ang2", "x", "ing2", "zh", "ang3"),
            features.symbols,
        )
    }

    @Test
    fun skipsUnknownPinyinFromBertRepeatCountsAndPhones() {
        val bertProvider = RecordingBertProvider()
        val unknownResources = resources.copy(
            singleCharPinyin = resources.singleCharPinyin + ("坏" to "bad1"),
        )

        val features = ChineseTextFeatureExtractor(
            resources = unknownResources,
            robertaProvider = bertProvider,
        ).extract("你好坏")

        assertEquals(listOf(2, 2), bertProvider.recordedWord2Ph)
        assertEquals(listOf("n", "i2", "h", "ao3"), features.symbols)
        assertEquals(4L, features.bert.shape[0])
    }

    @Test
    fun includesPunctuationExactlyOnceInBertRepeatCounts() {
        val bertProvider = RecordingBertProvider()
        val extractor = ChineseTextFeatureExtractor(resources, robertaProvider = bertProvider)

        extractor.extract("\u4f60\u597d.")

        assertEquals(listOf(2, 2, 1), bertProvider.recordedWord2Ph)
    }

    @Test
    fun mapsPythonPunctuationBeforePhoneConversion() {
        val extractor = ChineseTextFeatureExtractor(resources)

        val features = extractor.extract(
            "\u4f60\u597d\uff1a\u4f60\u597d\uff1b\u4f60\u597d\u3001\u4f60\u597d\u2026",
        )

        assertEquals(
            "\u4f60\u597d,\u4f60\u597d,\u4f60\u597d,\u4f60\u597d\u2026",
            features.normalizedText,
        )
        assertEquals(3, features.symbols.count { it == "," })
        assertEquals(1, features.symbols.count { it == "\u2026" })
    }

    @Test
    fun mapsPythonMiddleDotAndDollarPunctuation() {
        val features = ChineseTextFeatureExtractor(resources).extract("你好·你好$")

        assertEquals("你好,你好.", features.normalizedText)
        assertEquals(1, features.symbols.count { it == "," })
        assertEquals(1, features.symbols.count { it == "." })
    }

    @Test
    fun appliesYiReduplicationRuleBeforeGlobalToneSandhi() {
        val reduplicationResources = resources.copy(
            singleCharPinyin = resources.singleCharPinyin + mapOf(
                "\u770b" to "kan4",
                "\u4e00" to "yi1",
            ),
            opencpopByPinyin = resources.opencpopByPinyin + mapOf(
                "kan" to "k an",
                "yi" to "y i",
            ),
            wordTags = mapOf("\u770b" to "v"),
        )

        val features = ChineseTextFeatureExtractor(reduplicationResources).extract("\u770b\u4e00\u770b")

        assertEquals(
            listOf("k", "an4", "y", "i5", "k", "an4"),
            features.symbols,
        )
    }

    @Test
    fun doesNotHideMissingRobertaBehindZeroTensor() {
        val extractor = ChineseTextFeatureExtractor(resources, allowZeroBert = false)

        val error = org.junit.Assert.assertThrows(RobertaFeatureException::class.java) {
            extractor.extract("\u4f60\u597d")
        }

        assertEquals("missing_model", error.code)
    }
}
