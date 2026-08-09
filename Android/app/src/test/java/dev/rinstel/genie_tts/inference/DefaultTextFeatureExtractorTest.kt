package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultTextFeatureExtractorTest {
    @Test
    fun autoDetectsHybridFrontendForChineseTextContainingEnglish() {
        val chinese = ChineseTextFeatureExtractor(
            ChineseG2pResources(
                polyphonicPinyin = emptyMap(),
                singleCharPinyin = mapOf(
                    "\u4f60" to "ni3",
                    "\u597d" to "hao3",
                ),
                opencpopByPinyin = mapOf(
                    "ni" to "n i",
                    "hao" to "h ao",
                ),
            ),
        )
        val extractor = DefaultTextFeatureExtractor(
            chinese = chinese,
            english = EnglishTextFeatureExtractor(),
            fallback = FallbackTextFeatureExtractor(),
        )

        val features = extractor.extract("zh", "\u4f60\u597dABC")

        assertEquals(listOf("n", "i2", "h", "ao3", "a", "b", "c"), features.symbols)
        assertEquals(7, features.phoneIds.values.size)
        assertEquals(7L, features.bert.shape[0])
    }

    @Test
    fun supportsExplicitHybridLanguage() {
        val chinese = ChineseTextFeatureExtractor(
            ChineseG2pResources(
                polyphonicPinyin = emptyMap(),
                singleCharPinyin = mapOf("\u4f60" to "ni3"),
                opencpopByPinyin = mapOf("ni" to "n i"),
            ),
        )
        val extractor = DefaultTextFeatureExtractor(
            chinese = chinese,
            english = EnglishTextFeatureExtractor(),
            fallback = FallbackTextFeatureExtractor(),
        )

        val features = extractor.extract("hybrid-chinese-english", "\u4f60A")

        assertEquals(listOf("n", "i3", "a"), features.symbols)
    }

    @Test
    fun usesChineseExtractorForChineseRequests() {
        val extractor = DefaultTextFeatureExtractor(
            chinese = ChineseTextFeatureExtractor(
                ChineseG2pResources(
                    polyphonicPinyin = mapOf("哪吒" to listOf("ne2", "zha1")),
                    opencpopByPinyin = mapOf("ne" to "n e", "zha" to "zh a"),
                    singleCharPinyin = emptyMap(),
                ),
            ),
            fallback = FallbackTextFeatureExtractor(),
        )

        val features = extractor.extract("zh", "哪吒")

        assertEquals(4, features.phoneIds.values.size)
        assertTrue(features.symbols.contains("zh"))
    }

    @Test
    fun rejectsNonChineseRequestsUntilOriginalFrontendIsPackaged() {
        val extractor = DefaultTextFeatureExtractor(
            chinese = ChineseTextFeatureExtractor(
                ChineseG2pResources(
                    polyphonicPinyin = emptyMap(),
                    opencpopByPinyin = emptyMap(),
                    singleCharPinyin = emptyMap(),
                ),
            ),
            fallback = FallbackTextFeatureExtractor(),
        )

        val error = assertThrows(UnsupportedOperationException::class.java) {
            extractor.extract("en", "test")
        }
        assertTrue(error.message.orEmpty().contains("not migrated"))
    }
}
