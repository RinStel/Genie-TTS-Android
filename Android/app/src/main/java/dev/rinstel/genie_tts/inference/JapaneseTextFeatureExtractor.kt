package dev.rinstel.genie_tts.inference

class JapaneseTextFeatureExtractor : TextFeatureExtractor {

    override fun extract(language: String, text: String): TextFeatures {
        val normalizedText = normalizeText(text)
        val symbols = mutableListOf<String>()

        var index = 0
        while (index < normalizedText.length) {
            val char = normalizedText[index]
            when {
                isPunctuation(char) -> symbols += punctuationToSymbol(char)
                isHiragana(char) || isKatakana(char) -> {
                    val phonemes = kanaToPhonemes(normalizedText, index)
                    if (phonemes != null) {
                        symbols += phonemes.first
                        index += phonemes.second
                        continue
                    }
                    index += 1
                }
                char.isLetter() -> {
                    val romajiPhonemes = romajiCharToPhonemes(char)
                    symbols += romajiPhonemes
                }
                else -> Unit
            }
            index += 1
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
        var result = text.trim().lowercase()
        result = result.replace("％", "パーセント").replace("%", "パーセント")
        result = duplicatePunctRegex.replace(result, "$1")
        return result
    }

    private fun kanaToPhonemes(text: String, startIndex: Int): Pair<List<String>, Int>? {
        val remaining = text.substring(startIndex)
        for ((kana, phonemes) in kanaPhonemeMap) {
            if (remaining.startsWith(kana)) {
                return phonemes to kana.length
            }
        }
        return null
    }

    private fun romajiCharToPhonemes(char: Char): List<String> =
        listOf(char.toString())

    private fun isHiragana(char: Char): Boolean = char.code in 0x3040..0x309F
    private fun isKatakana(char: Char): Boolean = char.code in 0x30A0..0x30FF

    private fun isPunctuation(char: Char): Boolean =
        char in setOf('。', '、', '，', '．', '！', '？', '!', '?', ',', '.', '-', '…', '〜', '~')

    private fun punctuationToSymbol(char: Char): String = when (char) {
        '。', '.', '．' -> "."
        '、', '，', ',' -> ","
        '！', '!' -> "!"
        '？', '?' -> "?"
        '-', '〜', '~' -> "-"
        '…' -> "…"
        else -> "UNK"
    }

    companion object {
        private val duplicatePunctRegex = Regex("([,./?!~…・])\\1+")

        private val kanaPhonemeMap: List<Pair<String, List<String>>> = listOf(
            "あ" to listOf("a"), "い" to listOf("i"), "う" to listOf("u"), "え" to listOf("e"), "お" to listOf("o"),
            "か" to listOf("k", "a"), "き" to listOf("k", "i"), "く" to listOf("k", "u"), "け" to listOf("k", "e"), "こ" to listOf("k", "o"),
            "さ" to listOf("s", "a"), "し" to listOf("sh", "i"), "す" to listOf("s", "u"), "せ" to listOf("s", "e"), "そ" to listOf("s", "o"),
            "た" to listOf("t", "a"), "ち" to listOf("ch", "i"), "つ" to listOf("ts", "u"), "て" to listOf("t", "e"), "と" to listOf("t", "o"),
            "な" to listOf("n", "a"), "に" to listOf("ny", "i"), "ぬ" to listOf("n", "u"), "ね" to listOf("n", "e"), "の" to listOf("n", "o"),
            "は" to listOf("h", "a"), "ひ" to listOf("h", "i"), "ふ" to listOf("f", "u"), "へ" to listOf("h", "e"), "ほ" to listOf("h", "o"),
            "ま" to listOf("m", "a"), "み" to listOf("m", "i"), "む" to listOf("m", "u"), "め" to listOf("m", "e"), "も" to listOf("m", "o"),
            "や" to listOf("y", "a"), "ゆ" to listOf("y", "u"), "よ" to listOf("y", "o"),
            "ら" to listOf("r", "a"), "り" to listOf("r", "i"), "る" to listOf("r", "u"), "れ" to listOf("r", "e"), "ろ" to listOf("r", "o"),
            "わ" to listOf("w", "a"), "を" to listOf("w", "o"), "ん" to listOf("N"),
            "が" to listOf("g", "a"), "ぎ" to listOf("g", "i"), "ぐ" to listOf("g", "u"), "げ" to listOf("g", "e"), "ご" to listOf("g", "o"),
            "ざ" to listOf("z", "a"), "じ" to listOf("j", "i"), "ず" to listOf("z", "u"), "ぜ" to listOf("z", "e"), "ぞ" to listOf("z", "o"),
            "だ" to listOf("d", "a"), "ぢ" to listOf("j", "i"), "づ" to listOf("z", "u"), "で" to listOf("d", "e"), "ど" to listOf("d", "o"),
            "ば" to listOf("b", "a"), "び" to listOf("b", "i"), "ぶ" to listOf("b", "u"), "べ" to listOf("b", "e"), "ぼ" to listOf("b", "o"),
            "ぱ" to listOf("p", "a"), "ぴ" to listOf("p", "i"), "ぷ" to listOf("p", "u"), "ぺ" to listOf("p", "e"), "ぽ" to listOf("p", "o"),
            "きゃ" to listOf("ky", "a"), "きゅ" to listOf("ky", "u"), "きょ" to listOf("ky", "o"),
            "しゃ" to listOf("sh", "a"), "しゅ" to listOf("sh", "u"), "しょ" to listOf("sh", "o"),
            "ちゃ" to listOf("ch", "a"), "ちゅ" to listOf("ch", "u"), "ちょ" to listOf("ch", "o"),
            "にゃ" to listOf("ny", "a"), "にゅ" to listOf("ny", "u"), "にょ" to listOf("ny", "o"),
            "ひゃ" to listOf("hy", "a"), "ひゅ" to listOf("hy", "u"), "ひょ" to listOf("hy", "o"),
            "みゃ" to listOf("my", "a"), "みゅ" to listOf("my", "u"), "みょ" to listOf("my", "o"),
            "りゃ" to listOf("ry", "a"), "りゅ" to listOf("ry", "u"), "りょ" to listOf("ry", "o"),
            "ぎゃ" to listOf("gy", "a"), "ぎゅ" to listOf("gy", "u"), "ぎょ" to listOf("gy", "o"),
            "じゃ" to listOf("j", "a"), "じゅ" to listOf("j", "u"), "じょ" to listOf("j", "o"),
            "びゃ" to listOf("by", "a"), "びゅ" to listOf("by", "u"), "びょ" to listOf("by", "o"),
            "ぴゃ" to listOf("py", "a"), "ぴゅ" to listOf("py", "u"), "ぴょ" to listOf("py", "o"),
            "ア" to listOf("a"), "イ" to listOf("i"), "ウ" to listOf("u"), "エ" to listOf("e"), "オ" to listOf("o"),
            "カ" to listOf("k", "a"), "キ" to listOf("k", "i"), "ク" to listOf("k", "u"), "ケ" to listOf("k", "e"), "コ" to listOf("k", "o"),
            "サ" to listOf("s", "a"), "シ" to listOf("sh", "i"), "ス" to listOf("s", "u"), "セ" to listOf("s", "e"), "ソ" to listOf("s", "o"),
            "タ" to listOf("t", "a"), "チ" to listOf("ch", "i"), "ツ" to listOf("ts", "u"), "テ" to listOf("t", "e"), "ト" to listOf("t", "o"),
            "ナ" to listOf("n", "a"), "ニ" to listOf("ny", "i"), "ヌ" to listOf("n", "u"), "ネ" to listOf("n", "e"), "ノ" to listOf("n", "o"),
            "ハ" to listOf("h", "a"), "ヒ" to listOf("h", "i"), "フ" to listOf("f", "u"), "ヘ" to listOf("h", "e"), "ホ" to listOf("h", "o"),
            "マ" to listOf("m", "a"), "ミ" to listOf("m", "i"), "ム" to listOf("m", "u"), "メ" to listOf("m", "e"), "モ" to listOf("m", "o"),
            "ヤ" to listOf("y", "a"), "ユ" to listOf("y", "u"), "ヨ" to listOf("y", "o"),
            "ラ" to listOf("r", "a"), "リ" to listOf("r", "i"), "ル" to listOf("r", "u"), "レ" to listOf("r", "e"), "ロ" to listOf("r", "o"),
            "ワ" to listOf("w", "a"), "ヲ" to listOf("w", "o"), "ン" to listOf("N"),
            "ガ" to listOf("g", "a"), "ギ" to listOf("g", "i"), "グ" to listOf("g", "u"), "ゲ" to listOf("g", "e"), "ゴ" to listOf("g", "o"),
            "ザ" to listOf("z", "a"), "ジ" to listOf("j", "i"), "ズ" to listOf("z", "u"), "ゼ" to listOf("z", "e"), "ゾ" to listOf("z", "o"),
            "ダ" to listOf("d", "a"), "ヂ" to listOf("j", "i"), "ヅ" to listOf("z", "u"), "デ" to listOf("d", "e"), "ド" to listOf("d", "o"),
            "バ" to listOf("b", "a"), "ビ" to listOf("b", "i"), "ブ" to listOf("b", "u"), "ベ" to listOf("b", "e"), "ボ" to listOf("b", "o"),
            "パ" to listOf("p", "a"), "ピ" to listOf("p", "i"), "プ" to listOf("p", "u"), "ペ" to listOf("p", "e"), "ポ" to listOf("p", "o"),
            "キャ" to listOf("ky", "a"), "キュ" to listOf("ky", "u"), "キョ" to listOf("ky", "o"),
            "シャ" to listOf("sh", "a"), "シュ" to listOf("sh", "u"), "ショ" to listOf("sh", "o"),
            "チャ" to listOf("ch", "a"), "チュ" to listOf("ch", "u"), "チョ" to listOf("ch", "o"),
            "ニャ" to listOf("ny", "a"), "ニュ" to listOf("ny", "u"), "ニョ" to listOf("ny", "o"),
            "ヒャ" to listOf("hy", "a"), "ヒュ" to listOf("hy", "u"), "ヒョ" to listOf("hy", "o"),
            "ミャ" to listOf("my", "a"), "ミュ" to listOf("my", "u"), "ミョ" to listOf("my", "o"),
            "リャ" to listOf("ry", "a"), "リュ" to listOf("ry", "u"), "リョ" to listOf("ry", "o"),
            "ギャ" to listOf("gy", "a"), "ギュ" to listOf("gy", "u"), "ギョ" to listOf("gy", "o"),
            "ジャ" to listOf("j", "a"), "ジュ" to listOf("j", "u"), "ジョ" to listOf("j", "o"),
            "ビャ" to listOf("by", "a"), "ビュ" to listOf("by", "u"), "ビョ" to listOf("by", "o"),
            "ピャ" to listOf("py", "a"), "ピュ" to listOf("py", "u"), "ピョ" to listOf("py", "o"),
            "ー" to listOf("_"),
        )
    }
}
