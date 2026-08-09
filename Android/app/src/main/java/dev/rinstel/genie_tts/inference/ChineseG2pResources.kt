package dev.rinstel.genie_tts.inference

data class ChineseWordEntry(
    val frequency: Int,
    val tag: String,
    val pinyinCsv: String,
)

data class ChineseG2pResources(
    val polyphonicPinyin: Map<String, List<String>>,
    val opencpopByPinyin: Map<String, String>,
    val singleCharPinyin: Map<String, String>,
    val wordFrequencies: Map<String, Int> = emptyMap(),
    val wordTags: Map<String, String> = emptyMap(),
    val wordPinyin: Map<String, List<String>> = emptyMap(),
    val wordEntries: Map<String, ChineseWordEntry> = emptyMap(),
    val traditionalToSimplified: Map<Char, String> = emptyMap(),
) {
    val maxPhraseLength: Int = polyphonicPinyin.keys.maxOfOrNull(String::length) ?: 1
    val maxWordLength: Int = when {
        wordEntries.isNotEmpty() -> wordEntries.keys.maxOfOrNull(String::length) ?: 1
        else -> wordFrequencies.keys.maxOfOrNull(String::length) ?: 1
    }
    val totalWordFrequency: Double = when {
        wordEntries.isNotEmpty() -> wordEntries.values.sumOf { it.frequency }.coerceAtLeast(1).toDouble()
        else -> wordFrequencies.values.sum().coerceAtLeast(1).toDouble()
    }

    fun wordFrequency(word: String): Int = wordEntries[word]?.frequency ?: wordFrequencies[word] ?: 0

    fun wordTag(word: String): String = wordEntries[word]?.tag ?: wordTags[word].orEmpty()

    fun wordPinyinFor(word: String): List<String>? {
        val csv = wordEntries[word]?.pinyinCsv
        if (csv != null) return csv.split(',')
        return wordPinyin[word]
    }
}
