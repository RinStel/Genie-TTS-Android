package dev.rinstel.genie_tts.inference

object ChineseTextNormalizer {

    private val circledNumberMap = mapOf(
        '①' to "一", '②' to "二", '③' to "三", '④' to "四", '⑤' to "五",
        '⑥' to "六", '⑦' to "七", '⑧' to "八", '⑨' to "九", '⑩' to "十",
    )

    private val greekMap = mapOf(
        'α' to "阿尔法", 'β' to "贝塔", 'γ' to "伽玛", 'Γ' to "伽玛",
        'δ' to "德尔塔", 'Δ' to "德尔塔", 'ε' to "艾普西龙", 'ζ' to "捷塔",
        'η' to "依塔", 'θ' to "西塔", 'Θ' to "西塔", 'ι' to "艾欧塔",
        'κ' to "喀帕", 'λ' to "拉姆达", 'Λ' to "拉姆达", 'μ' to "缪",
        'ν' to "拗", 'ξ' to "克西", 'Ξ' to "克西", 'ο' to "欧米克伦",
        'π' to "派", 'Π' to "派", 'ρ' to "肉",
        'ς' to "西格玛", 'Σ' to "西格玛", 'σ' to "西格玛",
        'τ' to "套", 'υ' to "宇普西龙",
        'φ' to "服艾", 'Φ' to "服艾", 'χ' to "器",
        'ψ' to "普赛", 'Ψ' to "普赛", 'ω' to "欧米伽", 'Ω' to "欧米伽",
    )

    private val mathSymbolMap = mapOf(
        '+' to "加", '-' to "减", '×' to "乘", '÷' to "除", '=' to "等于",
    )

    private val fullwidthDigitMap: Map<Char, Char> = (0xFF10..0xFF19).mapIndexed { i, code ->
        code.toChar() to ('0' + i).toChar()
    }.toMap()

    private val fullwidthLetterMap: Map<Char, Char> =
        ((0xFF21..0xFF3A).mapIndexed { i, code -> code.toChar() to ('A' + i).toChar() } +
            (0xFF41..0xFF5A).mapIndexed { i, code -> code.toChar() to ('a' + i).toChar() }).toMap()

    private val fullwidthSpace = '\u3000' to ' '

    private val specialCharFilter = Regex("[——《》【】<>{}()（）#&@“”^_|\\\\]")
    private val sentenceSplitRegex = Regex("([：、，；。？！,;?!][”’]?)")

    // Keep the order aligned with text_normlization.py: specialized patterns
    // must consume their numeric text before the generic number rule runs.
    private val dateRegex = Regex(
        """(\d{4}|\d{2})年(?:(0?[1-9]|1[0-2])月)?(?:(0?[1-9]|[12]\d|30|31)([日号]))?""",
    )
    private val slashDateRegex = Regex(
        """(\d{4})([- /.])(0[1-9]|1[0-2])\2(0[1-9]|[12]\d|3[01])""",
    )
    private val timeRangeRegex = Regex(
        """([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?[-~]([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?""",
    )
    private val timeRegex = Regex("""([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?""")
    private val unitRangeRegex = Regex(
        """(?:-?(?:\d+(?:\.\d+)?|\.\d+))(?:%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)~(?:-?(?:\d+(?:\.\d+)?|\.\d+))(?:%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)""",
    )
    private val temperatureRegex = Regex("""(-?)(\d+(?:\.\d+)?)(°C|℃|度|摄氏度)""")
    private val measureRegex = Regex("""cm2|cm²|cm3|cm³|m2|m²|m³|m3|ml|mm|cm|db|ds|kg|km|s|m""")
    private val asmdRegex = Regex(
        """((?:-?(?:\d+(?:\.\d+)?|\.\d+)|[A-Za-z]+))([+\-×÷=])((?:-?(?:\d+(?:\.\d+)?|\.\d+)|[A-Za-z]+))""",
    )
    private val powerRegex = Regex("[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]+")
    private val fractionRegex = Regex("""(-?)(\d+)/(\d+)""")
    private val percentageRegex = Regex("""(-?)(\d+(?:\.\d+)?)%""")
    private val mobilePhoneRegex = Regex(
        """(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)""",
    )
    private val telephoneRegex = Regex(
        """(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)""",
    )
    private val nationalPhoneRegex = Regex("""400-?\d{3}-?\d{4}""")
    private val numberRangeRegex = Regex(
        """(?<![\d+\-×÷=])(-?(?:\d+(?:\.\d+)?|\.\d+))[-~](-?(?:\d+(?:\.\d+)?|\.\d+))(?![\d+\-×÷=])""",
    )
    private val negativeIntegerRegex = Regex("""-(\d+)""")
    private val versionRegex = Regex("""\d+\.\d+\.\d+(?:\.\d+)+""")
    private val numberRegex = Regex("""-?(?:\d+(?:\.\d+)?|\.\d+)""")
    private val quantifierRegex = Regex(
        """(\d+)([多余几+])?(个|人|位|名|件|本|页|家|户|层|条|张|只|支|辆|首|篇|段|次|遍|岁|年|月|日|小时|时|分|秒|米|厘米|毫米|千克|公斤|杯|瓶|袋|盒|份|场|回|所|间|台|把|套|种|门|门)""",
    )
    private val digitToChinese = mapOf(
        '0' to "零", '1' to "一", '2' to "二", '3' to "三", '4' to "四",
        '5' to "五", '6' to "六", '7' to "七", '8' to "八", '9' to "九",
    )

    private val powerMap = mapOf(
        '⁰' to '0', '¹' to '1', '²' to '2', '³' to '3', '⁴' to '4',
        '⁵' to '5', '⁶' to '6', '⁷' to '7', '⁸' to '8', '⁹' to '9',
        'ˣ' to 'x', 'ʸ' to 'y', 'ⁿ' to 'n',
    )
    private val measureMap = mapOf(
        "cm2" to "平方厘米", "cm²" to "平方厘米", "cm3" to "立方厘米", "cm³" to "立方厘米",
        "cm" to "厘米", "db" to "分贝", "ds" to "毫秒", "kg" to "千克", "km" to "千米",
        "m2" to "平方米", "m²" to "平方米", "m³" to "立方米", "m3" to "立方米",
        "ml" to "毫升", "m" to "米", "mm" to "毫米", "s" to "秒",
    )

    fun normalize(text: String, traditionalToSimplified: Map<Char, String> = emptyMap()): String {
        val filtered = specialCharFilter.replace(text.replace(" ", ""), "")
        val sentences = splitSentences(filtered)
        return sentences.joinToString("") { normalizeSentence(it, traditionalToSimplified) }
    }

    private fun splitSentences(text: String): List<String> {
        val withBreaks = sentenceSplitRegex.replace(text) { "${it.value}\n" }
        return withBreaks.split(Regex("\\n+")).map { it.trim() }.filter { it.isNotEmpty() }
    }

    private fun normalizeSentence(
        sentence: String,
        traditionalToSimplified: Map<Char, String>,
    ): String {
        var result = convertTraditionalToSimplified(sentence, traditionalToSimplified)
        result = convertFullwidthToHalfwidth(result)
        result = normalizeDates(result)
        result = normalizeTimes(result)
        result = unitRangeRegex.replace(result) { it.value.replace("~", "至") }
        result = temperatureRegex.replace(result) { match ->
            val sign = if (match.groupValues[1].isEmpty()) "" else "零下"
            val unit = if (match.groupValues[3] == "摄氏度") "摄氏度" else "度"
            sign + convertNumber(match.groupValues[2], useSpokenOne = false) + unit
        }
        result = measureRegex.replace(result) { measureMap[it.value] ?: it.value }
        while (asmdRegex.containsMatchIn(result)) {
            result = asmdRegex.replace(result) { match ->
                val operator = mathSymbolMap[match.groupValues[2][0]] ?: match.groupValues[2]
                match.groupValues[1] + operator + match.groupValues[3]
            }
        }
        result = powerRegex.replace(result) { match ->
            "的" + match.value.map { powerMap[it] ?: it }.joinToString("") + "次方"
        }
        result = fractionRegex.replace(result) { match ->
            val sign = if (match.groupValues[1].isEmpty()) "" else "负"
            sign + convertNumber(match.groupValues[3], useSpokenOne = false) + "分之" +
                convertNumber(match.groupValues[2], useSpokenOne = false)
        }
        result = percentageRegex.replace(result) { match ->
            val sign = if (match.groupValues[1].isEmpty()) "" else "负"
            sign + "百分之" + convertNumber(match.groupValues[2], useSpokenOne = false)
        }
        result = mobilePhoneRegex.replace(result) { spellMobilePhone(it.value) }
        result = telephoneRegex.replace(result) { spellPhone(it.value, mobile = false) }
        result = nationalPhoneRegex.replace(result) { spellPhone(it.value, mobile = false) }
        result = numberRangeRegex.replace(result) { match ->
            convertNumber(match.groupValues[1], useSpokenOne = false) + "到" +
                convertNumber(match.groupValues[2], useSpokenOne = false)
        }
        result = negativeIntegerRegex.replace(result) { "负${it.groupValues[1]}" }
        result = versionRegex.replace(result) { match ->
            match.value.map { if (it == '.') '点' else digitToChinese[it] ?: it }.joinToString("")
        }
        result = quantifierRegex.replace(result) { match ->
            val modifier = when (match.groupValues[2]) {
                "+" -> "多"
                else -> match.groupValues[2]
            }
            val number = convertNumber(match.groupValues[1], useSpokenOne = false)
                .let { if (it == "二") "两" else it }
            number + modifier + match.groupValues[3]
        }
        result = numberRegex.replace(result) { convertNumber(it.value, useSpokenOne = it.value.length >= 3) }
        result = replaceCircledNumbers(result)
        result = replaceGreek(result)
        result = replaceMathSymbols(result)
        return result.replace("/", "每")
    }

    private fun normalizeDates(text: String): String {
        var result = dateRegex.replace(text) { match ->
            val year = spellDigits(match.groupValues[1])
            val month = match.groupValues[2].takeIf { it.isNotEmpty() }
                ?.let { convertNumber(it, useSpokenOne = false) + "月" } ?: ""
            val day = match.groupValues[3].takeIf { it.isNotEmpty() }
                ?.let { convertNumber(it, useSpokenOne = false) + match.groupValues[4] } ?: ""
            year + "年" + month + day
        }
        return slashDateRegex.replace(result) { match ->
            spellDigits(match.groupValues[1]) + "年" +
                convertNumber(match.groupValues[3], useSpokenOne = false) + "月" +
                convertNumber(match.groupValues[4], useSpokenOne = false) + "日"
        }
    }

    private fun normalizeTimes(text: String): String {
        var result = timeRangeRegex.replace(text) { match ->
            formatTime(match.groupValues[1], match.groupValues[2], match.groupValues[3]) + "至" +
                formatTime(match.groupValues[4], match.groupValues[5], match.groupValues[6])
        }
        return timeRegex.replace(result) { match ->
            formatTime(match.groupValues[1], match.groupValues[2], match.groupValues[3])
        }
    }

    private fun formatTime(hour: String, minute: String, second: String): String {
        val result = StringBuilder(convertNumber(hour, useSpokenOne = false)).append("点")
        if (minute.trimStart('0').isNotEmpty()) {
            if (minute.toInt() == 30) result.append("半")
            else result.append(timeNumber(minute)).append("分")
        }
        if (second.isNotEmpty() && second.trimStart('0').isNotEmpty()) {
            result.append(timeNumber(second)).append("秒")
        }
        return result.toString()
    }

    private fun timeNumber(value: String): String {
        val stripped = value.trimStart('0')
        return if (value.startsWith('0')) "零" + convertNumber(stripped.ifEmpty { "0" }, false)
        else convertNumber(value, useSpokenOne = false)
    }

    private fun spellMobilePhone(value: String): String =
        value.trimStart('+').split(Regex("\\s+")).joinToString("，") { spellDigits(it, altOne = true) }

    private fun spellPhone(value: String, mobile: Boolean): String {
        val parts = if (mobile) value.trimStart('+').split(Regex("\\s+")) else value.split("-")
        return parts.joinToString("，") { spellDigits(it, altOne = true) }
    }

    private fun spellDigits(value: String, altOne: Boolean = false): String = buildString(value.length) {
        for (char in value) {
            val digit = digitToChinese[char]
            append(if (altOne && digit == "一") "幺" else digit ?: char.toString())
        }
    }

    private fun convertTraditionalToSimplified(
        text: String,
        mapping: Map<Char, String>,
    ): String = buildString(text.length) {
        for (char in text) append(mapping[char] ?: char)
    }

    private fun convertFullwidthToHalfwidth(text: String): String = buildString(text.length) {
        for (char in text) {
            when {
                char == fullwidthSpace.first -> append(fullwidthSpace.second)
                fullwidthDigitMap.containsKey(char) -> append(fullwidthDigitMap.getValue(char))
                fullwidthLetterMap.containsKey(char) -> append(fullwidthLetterMap.getValue(char))
                else -> append(char)
            }
        }
    }

    private fun convertNumber(number: String, useSpokenOne: Boolean): String {
        val decimal = number.indexOf('.')
        if (decimal >= 0) {
            val integerPart = convertInteger(number.substring(0, decimal), useSpokenOne = false)
            val decimalPart = number.substring(decimal + 1)
                .let { if (it.endsWith('0')) it.trimEnd('0') + "0" else it.trimEnd('0') }
                .map { digitToChinese[it] ?: it.toString() }
                .joinToString("")
            return "${integerPart}点$decimalPart"
        }
        return convertInteger(number, useSpokenOne)
    }

    private fun convertInteger(number: String, useSpokenOne: Boolean): String {
        if (useSpokenOne) return spellDigits(number, altOne = true)
        return convertCardinal(number)
    }

    private fun convertCardinal(number: String): String {
        val normalized = number.trimStart('0').ifEmpty { "0" }
        if (normalized == "0") return "零"
        val units = arrayOf("", "十", "百", "千", "万", "十万", "百万", "千万", "亿", "十亿", "百亿", "千亿", "兆")
        val result = StringBuilder()
        var zeroPending = false
        normalized.forEachIndexed { index, char ->
            val digit = char - '0'
            val position = normalized.length - index - 1
            if (digit == 0) {
                if (result.isNotEmpty()) zeroPending = true
                return@forEachIndexed
            }
            if (zeroPending && result.last() != '零') result.append('零')
            if (!(digit == 1 && position == 1 && result.isEmpty())) {
                result.append(digitToChinese[char] ?: char)
            }
            result.append(units.getOrElse(position) { "" })
            zeroPending = false
        }
        return result.toString().removePrefix("一十")
    }

    private fun replaceCircledNumbers(text: String): String = buildString(text.length) {
        for (char in text) append(circledNumberMap[char] ?: char)
    }

    private fun replaceGreek(text: String): String = buildString(text.length) {
        for (char in text) append(greekMap[char] ?: char)
    }

    private fun replaceMathSymbols(text: String): String = buildString(text.length) {
        for (char in text) append(mathSymbolMap[char] ?: char)
    }
}
