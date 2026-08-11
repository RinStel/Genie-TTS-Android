package dev.rinstel.genie_tts.inference

data class ReferenceAudioFingerprint(
    val canonicalPath: String,
    val size: Long,
    val lastModified: Long,
)

data class ReferenceConditioningCacheKey(
    val characterModelId: String,
    val backend: ExecutionBackend,
    val language: String,
    val referenceAudioPath: String,
    val referenceText: String,
    val usesPromptEncoder: Boolean,
    val referenceAudioSize: Long = -1L,
    val referenceAudioLastModified: Long = -1L,
    val modelInterfaceVersion: String = "legacy",
    val preprocessingVersion: String = "legacy",
    val conditioningRole: InferenceModelRole = InferenceModelRole.TEXT_FRONTEND,
    val auxiliaryReferenceAudioFingerprints: List<ReferenceAudioFingerprint> = emptyList(),
)

data class ReferenceConditioning(
    val refSeq: LongTensorData,
    val refBert: FloatTensorData,
    val sslContent: FloatTensorData,
    val globalEmbedding: FloatTensorData,
    val advancedGlobalEmbedding: FloatTensorData,
    val refAudio32k: FloatTensorData?,
    val primaryAudio32k: FloatArray? = null,
    val primaryAudio16k: FloatArray? = null,
    val primarySpeakerEmbedding: FloatTensorData? = null,
)

data class PromptConditioningCacheKey(
    val characterModelId: String,
    val backend: ExecutionBackend,
    val modelInterfaceVersion: String,
    val preprocessingVersion: String,
    val conditioningRole: InferenceModelRole,
    val primaryReferenceAudio: ReferenceAudioFingerprint? = null,
    val auxiliaryReferenceAudioFingerprints: List<ReferenceAudioFingerprint> = emptyList(),
)

/**
 * Cache only the active audiobook context. Switching character/reference
 * replaces the previous entry instead of retaining a hidden model library.
 */
internal object AudiobookCachePolicy {
    const val MAX_REFERENCE_CONDITIONING_ENTRIES = 1
    const val MAX_PROMPT_CONDITIONING_ENTRIES = 1
    const val MAX_AUXILIARY_REFERENCE_ENTRIES = 1
}

class ReferenceConditioningCache(
    private val maxEntries: Int = AudiobookCachePolicy.MAX_REFERENCE_CONDITIONING_ENTRIES,
) {
    init {
        require(maxEntries > 0) { "Reference conditioning cache capacity must be positive." }
    }

    private val entries = object : LinkedHashMap<ReferenceConditioningCacheKey, ReferenceConditioning>(
        maxEntries,
        LOAD_FACTOR,
        true,
    ) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<ReferenceConditioningCacheKey, ReferenceConditioning>,
        ): Boolean = size > maxEntries
    }

    @Synchronized
    fun get(key: ReferenceConditioningCacheKey): ReferenceConditioning? = entries[key]

    @Synchronized
    fun getOrPut(
        key: ReferenceConditioningCacheKey,
        factory: () -> ReferenceConditioning,
    ): ReferenceConditioning {
        entries[key]?.let { return it }
        return factory().also { entries[key] = it }
    }

    @Synchronized
    fun clear() {
        entries.clear()
    }

    companion object {
        private const val LOAD_FACTOR = 0.75f
    }
}

class PromptConditioningCache(
    private val maxEntries: Int = AudiobookCachePolicy.MAX_PROMPT_CONDITIONING_ENTRIES,
) {
    init {
        require(maxEntries > 0) { "Prompt conditioning cache capacity must be positive." }
    }

    private val entries = object : LinkedHashMap<PromptConditioningCacheKey, PromptEmbeddings>(
        maxEntries,
        LOAD_FACTOR,
        true,
    ) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<PromptConditioningCacheKey, PromptEmbeddings>,
        ): Boolean = size > maxEntries
    }

    @Synchronized
    fun get(key: PromptConditioningCacheKey): PromptEmbeddings? = entries[key]

    @Synchronized
    fun getOrPut(
        key: PromptConditioningCacheKey,
        factory: () -> PromptEmbeddings,
    ): PromptEmbeddings {
        entries[key]?.let { return it }
        return factory().also { entries[key] = it }
    }

    @Synchronized
    fun clear() {
        entries.clear()
    }

    companion object {
        private const val LOAD_FACTOR = 0.75f
    }
}
