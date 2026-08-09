package dev.rinstel.genie_tts.inference

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

object ChineseG2pAssetRepository {
    @Volatile
    private var cached: ChineseG2pResources? = null

    fun load(context: Context): ChineseG2pResources {
        val existing = cached
        if (existing != null) {
            return existing
        }
        return synchronized(this) {
            cached ?: loadUncached(context).also { cached = it }
        }
    }

    private fun loadUncached(context: Context): ChineseG2pResources {
        val manifest = context.assets.open("chinese_g2p/frontend_manifest.json")
            .bufferedReader(Charsets.UTF_8)
            .use { JSONObject(it.readText()) }
        require(manifest.optInt("schema_version", -1) == 1) {
            "Unsupported Chinese frontend resource schema."
        }
        val polyphonicJson = context.assets.open("chinese_g2p/polyphonic.json")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
        val opencpopText = context.assets.open("chinese_g2p/opencpop-strict.txt")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
        val singleCharJson = context.assets.open("chinese_g2p/single_char_pinyin.json")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }

        val polyphonicObject = JSONObject(polyphonicJson)
        val singleCharObject = JSONObject(singleCharJson)
        val polyphonic = buildMap(polyphonicObject.length()) {
            for (key in polyphonicObject.keys()) {
                val value = polyphonicObject.get(key)
                put(key, jsonValueToStringList(value))
            }
        }
        val singleChar = buildMap(singleCharObject.length()) {
            for (key in singleCharObject.keys()) {
                put(key, singleCharObject.getString(key))
            }
        }
        val opencpop = opencpopText.lineSequence()
            .filter(String::isNotBlank)
            .associate { line ->
                val parts = line.split('\t')
                require(parts.size >= 2) { "Invalid opencpop line: $line" }
                parts[0] to parts[1]
            }

        val wordEntries = mutableMapOf<String, ChineseWordEntry>()
        context.assets.open("chinese_g2p/contextual_lexicon.tsv")
            .bufferedReader(Charsets.UTF_8)
            .useLines { lines ->
                lines.forEach { line ->
                    val parts = line.split('\t', limit = 4)
                    if (parts.size < 2) return@forEach
                    val frequency = parts[1].toIntOrNull() ?: return@forEach
                    wordEntries[parts[0]] = ChineseWordEntry(
                        frequency = frequency,
                        tag = parts.getOrNull(2).orEmpty(),
                        pinyinCsv = parts.getOrNull(3).orEmpty(),
                    )
                }
            }

        val traditionalToSimplified = context.assets.open("chinese_g2p/traditional_to_simplified.tsv")
            .bufferedReader(Charsets.UTF_8)
            .useLines { lines ->
                lines.mapNotNull { line ->
                    val parts = line.split('\t', limit = 2)
                    val traditional = parts.getOrNull(0)?.singleOrNull()
                    val simplified = parts.getOrNull(1)?.takeIf(String::isNotEmpty)
                    if (traditional != null && simplified != null) traditional to simplified else null
                }.toMap()
            }

        return ChineseG2pResources(
            polyphonicPinyin = polyphonic,
            opencpopByPinyin = opencpop,
            singleCharPinyin = singleChar,
            wordEntries = wordEntries,
            traditionalToSimplified = traditionalToSimplified,
        )
    }

    private fun jsonValueToStringList(value: Any): List<String> =
        when (value) {
            is JSONArray -> List(value.length()) { index -> value.getString(index) }
            is String -> listOf(value)
            else -> error("Unsupported polyphonic entry type: ${value::class.java.name}")
        }
}
