package dev.rinstel.genie_tts.inference

object LanguageNormalizer {
    private val languageMap = mapOf(
        "chinese" to "chinese",
        "zh" to "chinese",
        "zh-cn" to "chinese",
        "zh-tw" to "chinese",
        "zh-hans" to "chinese",
        "zh-hant" to "chinese",

        "english" to "english",
        "en" to "english",
        "en-us" to "english",
        "en-gb" to "english",
        "eng" to "english",

        "japanese" to "japanese",
        "jp" to "japanese",
        "ja" to "japanese",
        "nihongo" to "japanese",

        "hybrid" to "hybrid-chinese-english",
        "hybrid-zh-en" to "hybrid-chinese-english",
        "hybrid-en-zh" to "hybrid-chinese-english",

        "korean" to "korean",
        "ko" to "korean",
        "kr" to "korean",
        "hangul" to "korean",
    )

    fun normalize(language: String): String =
        languageMap[language.lowercase().trim()] ?: language.lowercase().trim()

    fun isChinese(language: String): Boolean =
        normalize(language) == "chinese"

    fun isEnglish(language: String): Boolean =
        normalize(language) == "english"

    fun isJapanese(language: String): Boolean =
        normalize(language) == "japanese"

    fun isKorean(language: String): Boolean =
        normalize(language) == "korean"

    fun isHybridChineseEnglish(language: String): Boolean =
        normalize(language) == "hybrid-chinese-english"

    /** Mirrors the Python frontend guard against Chinese G2P dropping ASCII letters. */
    fun shouldAutoDetectHybrid(language: String, text: String): Boolean =
        isChinese(language) && text.any { it in 'a'..'z' || it in 'A'..'Z' }
}
