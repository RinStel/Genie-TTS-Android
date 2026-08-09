package dev.rinstel.genie_tts.inference

class ChineseTextFeatureExtractor(
    private val resources: ChineseG2pResources,
    private val robertaProvider: BertFeatureProvider? = null,
    private val toneSandhi: ToneSandhiProcessor = ToneSandhiProcessor(),
    private val erhua: ErhuaProcessor = ErhuaProcessor(),
    private val allowZeroBert: Boolean = true,
) {
    fun extract(text: String): TextFeatures {
        val normalizedText = normalizeText(text)

        val pinyinList = mutableListOf<String>()
        val isPunctList = mutableListOf<Boolean>()
        val word2phRaw = mutableListOf<Int>()
        var index = 0
        while (index < normalizedText.length) {
            val char = normalizedText[index]
            if (isSupportedPunctuation(char)) {
                pinyinList += char.toString()
                isPunctList += true
                index += 1
                continue
            }
            if (!isChineseCharacter(char)) {
                index += 1
                continue
            }

            val start = index
            while (index < normalizedText.length && isChineseCharacter(normalizedText[index])) index += 1
            val words = preMergeToneWords(segmentChineseText(normalizedText.substring(start, index)))
            for (segment in words) {
                val wordPinyin = pinyinForWord(segment.text)
                val tag = resources.wordTag(segment.text)
                val modifiedPinyin = toneSandhi.modifyWord(segment.text, tag, wordPinyin)
                val erhuaPinyin = erhua.modifyWord(modifiedPinyin, segment.text, tag)
                for (pinyin in erhuaPinyin) {
                    pinyinList += pinyin
                    isPunctList += false
                }
            }
        }

        // Python performs cross-word tone repair by pre-merging words before
        // the word-local rules. Do not apply a second sentence-wide pass here.
        val modifiedPinyins = pinyinList

        val symbols = mutableListOf<String>()
        for (i in modifiedPinyins.indices) {
            if (isPunctList[i]) {
                symbols += punctuationToSymbol(pinyinList[i][0])
                word2phRaw.add(1)
            } else {
                // Match Python G2P: an unknown OpenCPOP key contributes no
                // phones and must not leave a stale RoBERTa repeat count.
                val phones = pinyinToSymbols(modifiedPinyins[i]) ?: continue
                symbols += phones
                word2phRaw.add(phones.size)
            }
        }

        val filteredSymbols = symbols.filter { GenieSymbols.symbolToId.containsKey(it) }.ifEmpty { listOf("UNK") }
        val phoneIds = symbolsToTensor(filteredSymbols)

        val bert = if (word2phRaw.isEmpty()) {
            zeroBert(filteredSymbols.size)
        } else if (robertaProvider != null) {
            robertaProvider.computeBertFeatures(normalizedText, word2phRaw, filteredSymbols.size)
        } else if (allowZeroBert) {
            zeroBert(filteredSymbols.size)
        } else {
            throw RobertaFeatureException("missing_model", "Chinese RoBERTa model is not configured.")
        }

        return TextFeatures(
            normalizedText = normalizedText,
            symbols = filteredSymbols,
            phoneIds = phoneIds,
            bert = bert,
        )
    }

    private data class WordSegment(val text: String, val tag: String)

    private fun segmentChineseText(text: String): List<WordSegment> {
        if (text.isEmpty()) return emptyList()
        // The packaged exporter populates wordEntries; keep the legacy maps
        // supported for tests and older resource bundles.
        if (resources.wordFrequencies.isEmpty() && resources.wordEntries.isEmpty()) {
            return fallbackSegments(text)
        }

        val totalFrequency = resources.totalWordFrequency
        val scores = DoubleArray(text.length + 1) { Double.NEGATIVE_INFINITY }
        val lengths = IntArray(text.length)
        scores[text.length] = 0.0
        for (start in text.lastIndex downTo 0) {
            val maxLength = minOf(resources.maxWordLength, text.length - start)
            for (length in 1..maxLength) {
                val word = text.substring(start, start + length)
                val frequency = resources.wordFrequency(word)
                if (frequency <= 0) continue
                val suffixScore = scores[start + length]
                if (suffixScore.isInfinite() && suffixScore < 0) continue
                val score = kotlin.math.ln(frequency.toDouble()) - kotlin.math.ln(totalFrequency) + suffixScore
                if (score > scores[start]) {
                    scores[start] = score
                    lengths[start] = length
                }
            }
            if (lengths[start] == 0) {
                lengths[start] = 1
                scores[start] = kotlin.math.ln(1.0) - kotlin.math.ln(totalFrequency) + scores[start + 1]
            }
        }

        val result = mutableListOf<WordSegment>()
        var index = 0
        while (index < text.length) {
            val length = lengths[index].coerceAtLeast(1)
            val word = text.substring(index, minOf(text.length, index + length))
            result += WordSegment(word, resources.wordTag(word))
            index += length
        }
        return result
    }

    private fun fallbackSegments(text: String): List<WordSegment> {
        val result = mutableListOf<WordSegment>()
        var index = 0
        while (index < text.length) {
            val phraseMatch = longestPhraseMatch(text, index)
            if (phraseMatch != null) {
                result += WordSegment(phraseMatch.first, resources.wordTag(phraseMatch.first))
                index += phraseMatch.first.length
            } else {
                val word = text[index].toString()
                result += WordSegment(word, resources.wordTag(word))
                index += 1
            }
        }
        return result
    }

    private fun pinyinForWord(word: String): List<String> {
        val exportedPinyin = resources.wordPinyinFor(word)
        if (exportedPinyin != null && exportedPinyin.size == word.length) return exportedPinyin
        val phrasePinyin = resources.polyphonicPinyin[word]
        if (phrasePinyin != null && phrasePinyin.size == word.length) return phrasePinyin
        return word.map { char -> singleCharacterPinyin(char) ?: "" }
    }

    /** Mirrors Python ToneSandhi.pre_merge_for_modify before word-local rules run. */
    private fun preMergeToneWords(segments: List<WordSegment>): List<WordSegment> {
        var merged = mergeBuWords(segments)
        merged = mergeYiWords(merged)
        merged = mergeAdjacentReduplication(merged)
        merged = mergeContinuousThirdToneWords(merged, allWordsToneThree = true)
        merged = mergeContinuousThirdToneWords(merged, allWordsToneThree = false)
        return mergeErWords(merged)
    }

    private fun mergeBuWords(segments: List<WordSegment>): List<WordSegment> {
        val result = mutableListOf<WordSegment>()
        for (segment in segments) {
            val previous = result.lastOrNull()
            if (previous?.text == "\u4e0d") {
                result[result.lastIndex] = mergeWords(previous, segment)
            } else {
                result += segment
            }
        }
        return result
    }

    private fun mergeYiWords(segments: List<WordSegment>): List<WordSegment> {
        val result = mutableListOf<WordSegment>()
        var index = 0
        while (index < segments.size) {
            if (index + 2 < segments.size &&
                segments[index + 1].text == "\u4e00" &&
                segments[index].text == segments[index + 2].text &&
                segments[index].tag == "v" &&
                segments[index + 2].tag == "v"
            ) {
                result += mergeWords(segments[index], segments[index + 1], segments[index + 2])
                index += 3
            } else {
                result += segments[index]
                index += 1
            }
        }

        val appended = mutableListOf<WordSegment>()
        for (segment in result) {
            val previous = appended.lastOrNull()
            if (previous?.text == "\u4e00") {
                appended[appended.lastIndex] = mergeWords(previous, segment)
            } else {
                appended += segment
            }
        }
        return appended
    }

    private fun mergeAdjacentReduplication(segments: List<WordSegment>): List<WordSegment> {
        val result = mutableListOf<WordSegment>()
        for (segment in segments) {
            val previous = result.lastOrNull()
            if (previous != null && previous.text == segment.text) {
                result[result.lastIndex] = mergeWords(previous, segment)
            } else {
                result += segment
            }
        }
        return result
    }

    private fun mergeContinuousThirdToneWords(
        segments: List<WordSegment>,
        allWordsToneThree: Boolean,
    ): List<WordSegment> {
        val result = mutableListOf<WordSegment>()
        var previousWasMerged = false
        for (segment in segments) {
            val previous = result.lastOrNull()
            val shouldMerge = previous != null &&
                !previousWasMerged &&
                previous.text.length + segment.text.length <= 3 &&
                !isReduplication(previous.text) &&
                if (allWordsToneThree) {
                    allToneThree(previous.text) && allToneThree(segment.text)
                } else {
                    lastToneIsThree(previous.text) && firstToneIsThree(segment.text)
                }
            if (shouldMerge) {
                result[result.lastIndex] = mergeWords(previous!!, segment)
                previousWasMerged = true
            } else {
                result += segment
                previousWasMerged = false
            }
        }
        return result
    }

    private fun mergeErWords(segments: List<WordSegment>): List<WordSegment> {
        val result = mutableListOf<WordSegment>()
        for (segment in segments) {
            val previous = result.lastOrNull()
            if (segment.text == "\u513f" && previous != null && previous.text != "#") {
                result[result.lastIndex] = mergeWords(previous, segment)
            } else {
                result += segment
            }
        }
        return result
    }

    private fun mergeWords(vararg words: WordSegment): WordSegment {
        val text = words.joinToString(separator = "") { it.text }
        val tag = resources.wordTag(text).ifEmpty { words.first().tag }
        return WordSegment(text, tag)
    }

    private fun allToneThree(word: String): Boolean =
        pinyinForWord(word).let { pinyins ->
            pinyins.size == word.length && pinyins.all { toneOf(it) == 3 }
        }

    private fun lastToneIsThree(word: String): Boolean =
        toneOf(pinyinForWord(word).lastOrNull().orEmpty()) == 3

    private fun firstToneIsThree(word: String): Boolean =
        toneOf(pinyinForWord(word).firstOrNull().orEmpty()) == 3

    private fun isReduplication(word: String): Boolean =
        word.length == 2 && word[0] == word[1]

    private fun toneOf(pinyin: String): Int =
        pinyin.lastOrNull()?.digitToIntOrNull() ?: 0

    private fun normalizeText(text: String): String {
        val deepNormalized = ChineseTextNormalizer
            .normalize(text, resources.traditionalToSimplified)
            .replace("...", "\u2026")
        val trimmed = deepNormalized.trim()
        val replaced = buildString(trimmed.length) {
            trimmed.forEach { char ->
                when (char) {
                    '\n', '\r', '。' -> append('.')
                    '：', '、', '，', '；', ':', ';', ',' -> append(',')
                    '！', '!' -> append('!')
                    '？', '?' -> append('?')
                    '…', '~', '～' -> append('…')
                    '·' -> append(',')
                    '$' -> append('.')
                    '.' -> append('.')
                    ' ', '\t' -> Unit
                    else -> if (isChineseCharacter(char)) {
                        append(char)
                    }
                }
            }
        }
        return duplicatePunctuationRegex.replace(replaced, "$1")
    }

    private fun longestPhraseMatch(text: String, startIndex: Int): Pair<String, List<String>>? {
        val maxLength = minOf(resources.maxPhraseLength, text.length - startIndex)
        for (length in maxLength downTo 2) {
            val candidate = text.substring(startIndex, startIndex + length)
            val pinyin = resources.polyphonicPinyin[candidate] ?: continue
            return candidate to pinyin
        }
        return null
    }

    private fun singleCharacterPinyin(char: Char): String? {
        val override = resources.polyphonicPinyin[char.toString()]?.firstOrNull()
        if (override != null) {
            return override
        }
        return resources.singleCharPinyin[char.toString()]
    }

    private fun pinyinToSymbols(pinyinWithTone: String): List<String>? {
        val tone = pinyinWithTone.lastOrNull()?.takeIf(Char::isDigit) ?: return null
        val base = pinyinWithTone.dropLast(1)
        val initial = initials.firstOrNull(base::startsWith).orEmpty()
        val final = base.removePrefix(initial)
        val pinyinKey = if (initial.isNotEmpty()) {
            initial + (vReplacementMap[final] ?: final)
        } else {
            val direct = pinyinReplacementMap[final]
            if (direct != null) {
                direct
            } else {
                val first = final.firstOrNull()
                val single = first?.let(singleReplacementMap::get)
                if (single != null) {
                    single + final.drop(1)
                } else {
                    final
                }
            }
        }
        val phones = resources.opencpopByPinyin[pinyinKey]?.split(' ') ?: return null
        require(phones.size == 2) { "Unexpected opencpop mapping for $pinyinKey" }
        return listOf(phones[0], phones[1] + tone)
    }

    private fun punctuationToSymbol(char: Char): String =
        when (char) {
            '.', ',', '!', '?', '…' -> char.toString()
            '-' -> "-"
            else -> "UNK"
        }

    private fun isSupportedPunctuation(char: Char): Boolean =
        char == '.' || char == ',' || char == '!' || char == '?' || char == '…' || char == '-'

    private fun isChineseCharacter(char: Char): Boolean =
        char.code in 0x4e00..0x9fff

    companion object {
        const val BERT_FEATURE_DIM = 1024

        private val initials = listOf("zh", "ch", "sh", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "w", "x", "y", "z")
        private val vReplacementMap = mapOf("uei" to "ui", "iou" to "iu", "uen" to "un")
        private val pinyinReplacementMap = mapOf("ing" to "ying", "i" to "yi", "in" to "yin", "u" to "wu")
        private val singleReplacementMap = mapOf('v' to "yu", 'e' to "e", 'i' to "y", 'u' to "w")
        private val duplicatePunctuationRegex = Regex("([。！，？：；、,.!?…-])\\1+")
    }
}
