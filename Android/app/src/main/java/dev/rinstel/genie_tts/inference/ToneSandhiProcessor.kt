package dev.rinstel.genie_tts.inference

class ToneSandhiProcessor {

    private val mustNeuralToneWords: Set<String> = setOf(
        "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨",
        "风筝", "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠",
        "钥匙", "里脊", "里头", "部分", "那么", "道士", "造化", "迷糊", "连累", "这么",
        "这个", "运气", "过去", "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主",
        "豆腐", "讲究", "记性", "记号", "认识", "规矩", "见识", "裁缝", "补丁", "衣裳",
        "衣服", "衙门", "街坊", "行李", "行当", "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄",
        "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在",
        "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同",
        "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆", "老头", "老太",
        "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂", "精神",
        "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿",
        "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗",
        "砚台", "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛",
        "相声", "盘算", "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意",
        "甘蔗", "琵琶", "琢磨", "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务",
        "牲口", "牙碜", "牌楼", "爽快", "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心",
        "炊帚", "灯笼", "火候", "漂亮", "滑溜", "溜达", "温和", "清楚", "消息", "浪头",
        "活泼", "比方", "正经", "欺负", "模糊", "槟榔", "棺材", "棒槌", "棉花", "核桃",
        "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事", "木头", "木匠", "朋友",
        "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾", "收成", "提防",
        "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼", "抬举",
        "护士", "折腾", "扫帚", "打量", "打算", "打点", "打扮", "打听", "打发", "扎实",
        "扁担", "戒指", "懒得", "意识", "意思", "情形", "悟性", "怪物", "思量", "怎么",
        "念头", "念叨", "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通",
        "应酬", "庄稼", "干事", "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌",
        "差事", "工夫", "岁数", "屁股", "尾巴", "少爷", "小气", "小伙", "将就", "对头",
        "对付", "寡妇", "家伙", "客气", "实在", "官司", "学问", "学生", "字号", "嫁妆",
        "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫", "妯娌", "妥当", "妖精",
        "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫", "多少", "多么",
        "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔", "嘀咕",
        "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚",
        "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤",
        "口袋", "厚道", "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹",
        "功夫", "力气", "前头", "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析",
        "出息", "凑合", "凉快", "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟",
        "便宜", "使唤", "佩服", "作坊", "体面", "位置", "似的", "伙计", "休息", "什么",
        "人家", "亲戚", "亲家", "交情", "云彩", "事情", "买卖", "主意", "丫头", "丧气",
        "两口", "东西", "东家", "世故", "不由", "不在", "下水", "下巴", "上头", "上司",
        "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜", "邋遢", "费用",
        "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划", "扑腾",
        "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜",
        "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
    )

    private val mustNotNeuralToneWords: Set<String> = setOf(
        "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人",
        "虎虎", "幺幺", "干嘛", "学子", "哈哈", "数数", "袅袅", "局地", "以下", "娃哈哈",
        "花花草草", "留得", "耕地", "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳",
        "佼佼", "吵吵", "打打", "考考", "整整", "莘莘", "落地", "算子", "家家户户", "青青",
    )

    private val particleChars = "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶"
    private val deChars = "的地得"
    private val leZheGuo = "了着过"
    private val menZi = "们子"
    private val shangXiaLi = "上下里"
    private val laiQuPreceded = "上下进出回过起开"
    private val laiQu = "来去"
    private val gePreceded = "几有两半多各整每做是"

    /** Apply Python's word-local tone rules before cross-word third-tone repair. */
    fun modifyWord(word: String, partOfSpeech: String, pinyins: List<String>): List<String> {
        if (word.isEmpty() || pinyins.isEmpty()) return pinyins
        val result = pinyins.toMutableList()
        applyBuWord(word, result)
        applyYiWord(word, result)
        applyNeuralWord(word, partOfSpeech, result)
        applyThreeWord(result)
        return result
    }

    fun apply(pinyins: List<String>, isPunctuation: List<Boolean>): List<String> {
        if (pinyins.isEmpty()) return pinyins
        val result = pinyins.toMutableList()
        applyBuSandhi(result, isPunctuation)
        applyYiSandhi(result, isPunctuation)
        applyThreeSandhi(result, isPunctuation)
        applyNeuralSandhi(result, isPunctuation)
        return result
    }

    private fun applyBuWord(word: String, pinyins: MutableList<String>) {
        if ('不' in word && word.length == 3 && word[1] == '不') {
            pinyins[1] = changeTone(pinyins[1], 5)
            return
        }
        for (index in 0 until minOf(word.length, pinyins.size - 1)) {
            if (word[index] == '不' && getTone(pinyins[index + 1]) == 4) {
                pinyins[index] = changeTone(pinyins[index], 2)
            }
        }
    }

    private fun applyYiWord(word: String, pinyins: MutableList<String>) {
        val yiIndex = word.indexOf('一')
        if (yiIndex < 0 || yiIndex >= pinyins.size) return
        val numberChars = word.filter { it != '一' }
        if (numberChars.isNotEmpty() && numberChars.all { it in "零〇二三四五六七八九十百千万亿" }) return
        if (word.length == 3 && word[0] == word[2]) {
            pinyins[yiIndex] = changeTone(pinyins[yiIndex], 5)
            return
        }
        if (word.startsWith("第一") && yiIndex + 1 < pinyins.size) {
            pinyins[yiIndex] = changeTone(pinyins[yiIndex], 1)
            return
        }
        if (yiIndex + 1 < pinyins.size && word[yiIndex + 1] !in punctuationChars) {
            pinyins[yiIndex] = changeTone(
                pinyins[yiIndex],
                if (getTone(pinyins[yiIndex + 1]) == 4) 2 else 4,
            )
        }
    }

    private fun applyNeuralWord(word: String, partOfSpeech: String, pinyins: MutableList<String>) {
        for (index in 1 until minOf(word.length, pinyins.size)) {
            if (word[index] == word[index - 1] && partOfSpeech.firstOrNull() in setOf('n', 'v', 'a') &&
                word !in mustNotNeuralToneWords) {
                pinyins[index] = changeTone(pinyins[index], 5)
            }
        }
        val last = word.lastIndex
        val lastChar = word.lastOrNull()
        if (lastChar != null && (lastChar in particleChars || lastChar in deChars)) {
            pinyins[last] = changeTone(pinyins[last], 5)
        } else if (word.length == 1 && word[0] in leZheGuo && partOfSpeech in setOf("ul", "uz", "ug")) {
            pinyins[last] = changeTone(pinyins[last], 5)
        } else if (word.length > 1 && word.last() in menZi && partOfSpeech in setOf("r", "n") &&
            word !in mustNotNeuralToneWords) {
            pinyins[last] = changeTone(pinyins[last], 5)
        } else if (word.length > 1 && word.last() in shangXiaLi && partOfSpeech in setOf("s", "l", "f")) {
            pinyins[last] = changeTone(pinyins[last], 5)
        } else if (word.length > 1 && word.last() in laiQu && word[word.lastIndex - 1] in laiQuPreceded) {
            pinyins[last] = changeTone(pinyins[last], 5)
        } else {
            val geIndex = word.indexOf('个')
            if (geIndex > 0 && geIndex < pinyins.size &&
                (word[geIndex - 1].isDigit() || word[geIndex - 1] in gePreceded)) {
                pinyins[geIndex] = changeTone(pinyins[geIndex], 5)
            }
        }
        if (word in mustNeuralToneWords || word.takeLast(2) in mustNeuralToneWords) {
            pinyins[last] = changeTone(pinyins[last], 5)
        }
    }

    private fun applyThreeWord(pinyins: MutableList<String>) {
        if (pinyins.size == 2 && pinyins.all { getTone(it) == 3 }) {
            pinyins[0] = changeTone(pinyins[0], 2)
        } else if (pinyins.size == 3) {
            if (getTone(pinyins[0]) == 3 && getTone(pinyins[1]) == 3) {
                pinyins[0] = changeTone(pinyins[0], 2)
            }
            if (getTone(pinyins[1]) == 3 && getTone(pinyins[2]) == 3) {
                pinyins[1] = changeTone(pinyins[1], 2)
            }
        } else if (pinyins.size == 4) {
            for (start in listOf(0, 2)) {
                if (getTone(pinyins[start]) == 3 && getTone(pinyins[start + 1]) == 3) {
                    pinyins[start] = changeTone(pinyins[start], 2)
                }
            }
        }
    }

    fun applyToWord(word: String, pinyins: MutableList<String>, isPunctuation: List<Boolean>, startIndex: Int) {
        if (word in mustNeuralToneWords || word.takeLast(2) in mustNeuralToneWords) {
            if (pinyins.isNotEmpty()) {
                val lastPinyinIndex = startIndex + word.length - 1
                if (lastPinyinIndex < pinyins.size && !isPunctuation[lastPinyinIndex]) {
                    pinyins[lastPinyinIndex] = changeTone(pinyins[lastPinyinIndex], 5)
                }
            }
        }
    }

    private fun applyBuSandhi(pinyins: MutableList<String>, isPunctuation: List<Boolean>) {
        for (i in pinyins.indices) {
            if (isPunctuation[i]) continue
            val pinyin = pinyins[i]
            if (pinyin.startsWith("bu") && getTone(pinyin) == 4) {
                if (i + 1 < pinyins.size && !isPunctuation[i + 1] && getTone(pinyins[i + 1]) == 4) {
                    pinyins[i] = changeTone(pinyin, 2)
                }
            }
        }
    }

    private fun applyYiSandhi(pinyins: MutableList<String>, isPunctuation: List<Boolean>) {
        for (i in pinyins.indices) {
            if (isPunctuation[i]) continue
            val pinyin = pinyins[i]
            if (pinyin.startsWith("yi") && getTone(pinyin) == 1) {
                if (i + 1 >= pinyins.size || isPunctuation[i + 1]) continue
                val nextTone = getTone(pinyins[i + 1])
                if (nextTone == 4) {
                    pinyins[i] = changeTone(pinyin, 2)
                } else {
                    pinyins[i] = changeTone(pinyin, 4)
                }
            }
        }
    }

    private fun applyThreeSandhi(pinyins: MutableList<String>, isPunctuation: List<Boolean>) {
        for (i in 0 until pinyins.size - 1) {
            if (isPunctuation[i] || isPunctuation[i + 1]) continue
            if (getTone(pinyins[i]) == 3 && getTone(pinyins[i + 1]) == 3) {
                pinyins[i] = changeTone(pinyins[i], 2)
            }
        }
    }

    private fun applyNeuralSandhi(pinyins: MutableList<String>, isPunctuation: List<Boolean>) {
        for (i in pinyins.indices) {
            if (isPunctuation[i]) continue
            val pinyin = pinyins[i]
            val base = pinyin.dropLast(1)

            if (i > 0 && !isPunctuation[i - 1] && base in listOf("ba", "ne", "ha", "a", "na", "sai", "ma", "ya", "hi", "na", "o", "da", "e", "di", "li", "yo", "lou", "luo", "ye", "wo", "ei")) {
                pinyins[i] = changeTone(pinyin, 5)
            }
            if (base == "de" || base == "di" || base == "dede") {
                pinyins[i] = changeTone(pinyin, 5)
            }
            if (base == "le" || base == "zhe" || base == "guo") {
                if (i > 0 && !isPunctuation[i - 1]) {
                    pinyins[i] = changeTone(pinyin, 5)
                }
            }
            if (base == "men" || base == "zi") {
                if (i > 0 && !isPunctuation[i - 1]) {
                    pinyins[i] = changeTone(pinyin, 5)
                }
            }
            if (base in listOf("shang", "xia", "li")) {
                if (i > 0 && !isPunctuation[i - 1]) {
                    pinyins[i] = changeTone(pinyin, 5)
                }
            }
            if (base == "lai" || base == "qu") {
                if (i > 0 && !isPunctuation[i - 1]) {
                    val prevBase = pinyins[i - 1].dropLast(1)
                    if (prevBase in listOf("shang", "xia", "jin", "chu", "hui", "guo", "qi", "kai")) {
                        pinyins[i] = changeTone(pinyin, 5)
                    }
                }
            }
        }
    }

    private fun getTone(pinyin: String): Int {
        val last = pinyin.lastOrNull()
        return if (last?.isDigit() == true) last.toString().toInt() else 0
    }

    private fun changeTone(pinyin: String, newTone: Int): String {
        val last = pinyin.lastOrNull()
        return if (last?.isDigit() == true) {
            pinyin.dropLast(1) + newTone
        } else {
            pinyin + newTone
        }
    }

    companion object {
        private const val punctuationChars = "：，；。？！,;?!"
    }
}
