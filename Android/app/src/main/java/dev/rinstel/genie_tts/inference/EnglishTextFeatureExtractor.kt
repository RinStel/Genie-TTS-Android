package dev.rinstel.genie_tts.inference

class EnglishTextFeatureExtractor : TextFeatureExtractor {

    override fun extract(language: String, text: String): TextFeatures {
        val normalizedText = normalizeText(text)
        val symbols = mutableListOf<String>()

        for (char in normalizedText) {
            when {
                isPunctuation(char) -> symbols += punctuationToSymbol(char)
                char.isLetter() -> symbols += char.lowercase()
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

    private fun normalizeText(text: String): String {
        val repMap = mapOf(
            ';' to ',', ':' to ',', '，' to ',', '。' to '.',
            '！' to '!', '？' to '?', '"' to "'", '\u2019' to "'",
        )
        return buildString(text.length) {
            for (char in text.trim()) {
                append(repMap[char] ?: char)
            }
        }
    }

    private fun isPunctuation(char: Char): Boolean =
        char in setOf(',', '.', '!', '?', '-', '…', '_')

    private fun punctuationToSymbol(char: Char): String = when (char) {
        ',', '，' -> ","
        '.', '。' -> "."
        '!', '！' -> "!"
        '?', '？' -> "?"
        '-' -> "-"
        '…' -> "…"
        '_' -> "_"
        else -> "UNK"
    }
}
