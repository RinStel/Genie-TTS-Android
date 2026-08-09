package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class ChineseTextNormalizerTest {
    @Test
    fun omitsLeadingOneForChineseTeenNumbers() {
        assertEquals("十三", ChineseTextNormalizer.normalize("13"))
        assertEquals("十", ChineseTextNormalizer.normalize("10"))
        assertEquals("二十", ChineseTextNormalizer.normalize("20"))
    }

    @Test
    fun convertsFullwidthDigitsBeforeNumberVerbalization() {
        assertEquals("十二", ChineseTextNormalizer.normalize("１２"))
    }

    @Test
    fun matchesPythonLongNumberDecimalAndNegativeRules() {
        assertEquals("幺二三", ChineseTextNormalizer.normalize("123"))
        assertEquals("负十", ChineseTextNormalizer.normalize("-10"))
        assertEquals("十二点三零", ChineseTextNormalizer.normalize("12.30"))
        assertEquals("二零二六年七月十三日", ChineseTextNormalizer.normalize("2026年7月13日"))
    }

    @Test
    fun convertsTraditionalCharactersFromExportedMapping() {
        val mapping = mapOf(
            '\u81fa' to "\u53f0",
            '\u767c' to "\u53d1",
        )

        assertEquals(
            "\u53f0\u53d1",
            ChineseTextNormalizer.normalize("\u81fa\u767c", mapping),
        )
    }

    @Test
    fun matchesPythonMathAndPunctuationNormalization() {
        assertEquals("三减一等于二", ChineseTextNormalizer.normalize("3-1=2"))
    }

    @Test
    fun matchesPythonDateTimeFractionAndPercentageRules() {
        assertEquals("二零二六年七月十三日", ChineseTextNormalizer.normalize("2026/07/13"))
        assertEquals("八点半至十二点半", ChineseTextNormalizer.normalize("8:30-12:30"))
        assertEquals("四分之三", ChineseTextNormalizer.normalize("3/4"))
        assertEquals("负百分之十二点五", ChineseTextNormalizer.normalize("-12.5%"))
    }

    @Test
    fun matchesPythonPhoneTemperatureMeasureRangeVersionAndQuantifierRules() {
        assertEquals("幺三八零零幺三八零零零", ChineseTextNormalizer.normalize("13800138000"))
        assertEquals("零下三度", ChineseTextNormalizer.normalize("-3°C"))
        assertEquals("十二千克", ChineseTextNormalizer.normalize("12kg"))
        assertEquals("三到五", ChineseTextNormalizer.normalize("3~5"))
        assertEquals("一点二点三点四", ChineseTextNormalizer.normalize("1.2.3.4"))
        assertEquals("三个人", ChineseTextNormalizer.normalize("3个人"))
    }

    @Test
    fun removesSpacesLikePythonChineseNormalizer() {
        assertEquals("你好世界", ChineseTextNormalizer.normalize("你 好  世界"))
    }
}
