package dev.rinstel.genie_tts.inference

import android.content.Context

interface TextFeatureExtractor {
    fun extract(language: String, text: String): TextFeatures
}

class DefaultTextFeatureExtractor(
    private val chinese: ChineseTextFeatureExtractor,
    private val japanese: TextFeatureExtractor = UnsupportedLanguageTextFeatureExtractor(),
    private val english: TextFeatureExtractor = UnsupportedLanguageTextFeatureExtractor(),
    private val korean: TextFeatureExtractor = UnsupportedLanguageTextFeatureExtractor(),
    private val fallback: TextFeatureExtractor,
) : TextFeatureExtractor {
    override fun extract(language: String, text: String): TextFeatures {
        val normalized = LanguageNormalizer.normalize(language)
        return when {
            LanguageNormalizer.isHybridChineseEnglish(normalized) ||
                LanguageNormalizer.shouldAutoDetectHybrid(normalized, text) ->
                hybrid.extract("hybrid-chinese-english", text)
            LanguageNormalizer.isChinese(normalized) -> chinese.extract(text)
            LanguageNormalizer.isJapanese(normalized) -> japanese.extract(language, text)
            LanguageNormalizer.isEnglish(normalized) -> english.extract(language, text)
            LanguageNormalizer.isKorean(normalized) -> korean.extract(language, text)
            else -> fallback.extract(language, text)
        }
    }

    private val hybrid: TextFeatureExtractor = HybridTextFeatureExtractor(chinese, english)

    companion object {
        fun fromAssets(
            context: Context,
            robertaProvider: RobertaFeatureProvider? = null,
        ): DefaultTextFeatureExtractor =
            DefaultTextFeatureExtractor(
                chinese = ChineseTextFeatureExtractor(
                    ChineseG2pAssetRepository.load(context),
                    robertaProvider,
                    allowZeroBert = false,
                ),
                english = EnglishTextFeatureExtractor(),
                fallback = UnsupportedLanguageTextFeatureExtractor(),
            )
    }
}

/** Combines language-specific outputs in the same order as Python split_language. */
class HybridTextFeatureExtractor(
    private val chinese: ChineseTextFeatureExtractor,
    private val english: TextFeatureExtractor,
) : TextFeatureExtractor {
    private val englishRun = Regex("[A-Za-z]+")

    override fun extract(language: String, text: String): TextFeatures {
        val chunks = mutableListOf<TextFeatures>()
        var cursor = 0
        englishRun.findAll(text).forEach { match ->
            appendChineseChunk(chunks, text.substring(cursor, match.range.first))
            chunks += english.extract("english", match.value)
            cursor = match.range.last + 1
        }
        appendChineseChunk(chunks, text.substring(cursor))

        if (chunks.isEmpty()) {
            return chinese.extract(text)
        }

        val symbols = chunks.flatMap(TextFeatures::symbols)
        val phoneValues = LongArray(symbols.size)
        var phoneOffset = 0
        val bertValues = FloatArray(symbols.size * ChineseTextFeatureExtractor.BERT_FEATURE_DIM)
        var bertOffset = 0
        for (chunk in chunks) {
            chunk.phoneIds.values.copyInto(phoneValues, phoneOffset)
            phoneOffset += chunk.phoneIds.values.size
            chunk.bert.values.copyInto(bertValues, bertOffset)
            bertOffset += chunk.bert.values.size
        }

        return TextFeatures(
            normalizedText = chunks.joinToString(separator = "") { it.normalizedText },
            symbols = symbols,
            phoneIds = LongTensorData(
                values = phoneValues,
                shape = longArrayOf(1L, phoneValues.size.toLong()),
            ),
            bert = FloatTensorData(
                values = bertValues,
                shape = longArrayOf(
                    symbols.size.toLong(),
                    ChineseTextFeatureExtractor.BERT_FEATURE_DIM.toLong(),
                ),
            ),
        )
    }

    private fun appendChineseChunk(chunks: MutableList<TextFeatures>, chunk: String) {
        if (chunk.isNotBlank()) {
            chunks += chinese.extract(chunk)
        }
    }
}

class FallbackTextFeatureExtractor : TextFeatureExtractor {
    override fun extract(language: String, text: String): TextFeatures {
        val normalizedText = text.trim()
        val symbols = normalizedText.map { charToSymbol(it) }.ifEmpty { listOf("UNK") }
        val phoneIds = symbolsToTensor(symbols)
        return TextFeatures(
            normalizedText = normalizedText,
            symbols = symbols,
            phoneIds = phoneIds,
            bert = zeroBert(phoneIds.values.size),
        )
    }

    private fun charToSymbol(char: Char): String =
        when (char) {
            ' ', '\t', '\r', '\n' -> "_"
            '.', '\u3002' -> "."
            ',', '\uff0c', '\u3001' -> ","
            '!', '\uff01' -> "!"
            '?', '\uff1f' -> "?"
            '-' -> "-"
            else -> "UNK"
        }
}

class UnsupportedLanguageTextFeatureExtractor : TextFeatureExtractor {
    override fun extract(language: String, text: String): TextFeatures {
        throw UnsupportedOperationException(
            "Language '$language' is not migrated on Android yet. " +
                "Current real text frontend support is limited to Chinese. " +
                "English, Japanese, Korean, and hybrid paths still need dedicated G2P and BERT ports.",
        )
    }
}

object GenieTextConventions {
    fun prepareSynthesisText(language: String, text: String): String {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) {
            return trimmed
        }
        return "\u3002$trimmed"
    }
}

data class TextFeatures(
    val normalizedText: String,
    val symbols: List<String>,
    val phoneIds: LongTensorData,
    val bert: FloatTensorData,
)

internal fun symbolsToTensor(symbols: List<String>): LongTensorData =
    LongTensorData(
        values = symbols
            .map { symbol -> GenieSymbols.symbolToId[symbol] ?: GenieSymbols.symbolToId.getValue("UNK") }
            .map(Int::toLong)
            .toLongArray(),
        shape = longArrayOf(1L, symbols.size.toLong()),
    )

internal fun zeroBert(phoneCount: Int): FloatTensorData =
    FloatTensorData(
        values = FloatArray(phoneCount * ChineseTextFeatureExtractor.BERT_FEATURE_DIM),
        shape = longArrayOf(phoneCount.toLong(), ChineseTextFeatureExtractor.BERT_FEATURE_DIM.toLong()),
    )
