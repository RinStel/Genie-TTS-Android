package dev.rinstel.genie_tts.inference

class KoreanTextFeatureExtractor : TextFeatureExtractor {

    override fun extract(language: String, text: String): TextFeatures {
        val normalizedText = text.trim()
        val symbols = mutableListOf<String>()

        for (char in normalizedText) {
            when {
                isPunctuation(char) -> symbols += punctuationToSymbol(char)
                isHangul(char) -> {
                    val jamo = decomposeHangul(char)
                    symbols += jamo
                }
                char.isWhitespace() -> symbols += "_"
                else -> Unit
            }
        }

        val filteredSymbols = symbols.filter { GenieSymbols.symbolToId.containsKey(it) }.ifEmpty { listOf("UNK") }
        val phoneIds = symbolsToTensor(filteredSymbols)
        return TextFeatures(
            normalizedText = normalizedText,
            symbols = filteredSymbols,
            phoneIds = phoneIds,
            bert = zeroBert(filteredSymbols.size),
        )
    }

    private fun isHangul(char: Char): Boolean = char.code in 0xAC00..0xD7A3

    private fun isPunctuation(char: Char): Boolean =
        char in setOf(',', '.', '!', '?', '-', '…', '~')

    private fun punctuationToSymbol(char: Char): String = when (char) {
        ',' -> ","
        '.' -> "."
        '!' -> "!"
        '?' -> "?"
        '-' -> "-"
        '…' -> "…"
        '~' -> "…"
        else -> "UNK"
    }

    private fun decomposeHangul(syllable: Char): List<String> {
        val code = syllable.code - 0xAC00
        val leadIndex = code / 588
        val vowelIndex = (code % 588) / 28
        val tailIndex = code % 28

        val lead = leadConsonants.getOrNull(leadIndex)
        val vowel = vowels.getOrNull(vowelIndex)
        val tail = if (tailIndex > 0) tailConsonants.getOrNull(tailIndex) else null

        val result = mutableListOf<String>()
        if (lead != null && GenieSymbols.symbolToId.containsKey(lead)) result.add(lead)
        if (vowel != null && GenieSymbols.symbolToId.containsKey(vowel)) result.add(vowel)
        if (tail != null && GenieSymbols.symbolToId.containsKey(tail)) result.add(tail)
        return result.ifEmpty { listOf("UNK") }
    }

    companion object {
        private val leadConsonants = listOf(
            "g", "kk", "n", "d", "tt", "r", "m", "b", "pp",
            "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h",
        )
        private val vowels = listOf(
            "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o",
            "wa", "wae", "oe", "yo", "u", "weo", "we", "wi",
            "yu", "eu", "ui", "i",
        )
        private val tailConsonants = listOf(
            "", "k", "k", "k", "n", "n", "n", "t", "l", "l", "l",
            "l", "l", "l", "l", "l", "m", "p", "p", "p", "s",
            "s", "s", "s", "t", "t", "t", "p", "p", "h",
        )
    }
}
