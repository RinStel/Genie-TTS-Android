package dev.rinstel.genie_tts.inference

class ErhuaProcessor {

    private val mustErhua: Set<String> = setOf(
        "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿",
    )

    private val notErhua: Set<String> = setOf(
        "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿",
        "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿",
        "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿",
        "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿",
        "猪儿", "猫儿", "狗儿", "少儿",
    )

    fun modifyWord(pinyins: List<String>, word: String, partOfSpeech: String): List<String> {
        if (pinyins.size != word.length || word.isEmpty()) return pinyins
        val result = pinyins.toMutableList()
        val lastIndex = word.lastIndex
        if (word[lastIndex] == '儿' && result[lastIndex] == "er1") {
            result[lastIndex] = "er2"
        }
        if (word !in mustErhua && (word in notErhua || partOfSpeech in setOf("a", "j", "nr"))) {
            return result
        }
        val lastPinyin = result[lastIndex]
        if (word[lastIndex] == '儿' && lastPinyin in setOf("er2", "er5") &&
            word.takeLast(2) !in notErhua && lastIndex > 0) {
            val previousTone = result[lastIndex - 1].lastOrNull()
            if (previousTone?.isDigit() == true) result[lastIndex] = "er$previousTone"
        }
        return result
    }
}
