package dev.rinstel.genie_tts.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertSame
import org.junit.Test

class ReferenceConditioningCacheTest {
    @Test
    fun promptCacheKeepsIndependentAuxiliaryBundles() {
        val cache = PromptConditioningCache(maxEntries = 2)
        val primary = ReferenceAudioFingerprint("/ref/primary.wav", 1L, 1L)
        val firstAuxiliary = ReferenceAudioFingerprint("/ref/first.wav", 2L, 2L)
        val secondAuxiliary = ReferenceAudioFingerprint("/ref/second.wav", 3L, 3L)
        val firstKey = promptKey(primary, listOf(firstAuxiliary))
        val secondKey = promptKey(primary, listOf(secondAuxiliary))
        var factoryCalls = 0

        val first = cache.getOrPut(firstKey) {
            factoryCalls += 1
            PromptEmbeddings(FloatTensorData(floatArrayOf(1f), longArrayOf(1L)), FloatTensorData(floatArrayOf(1f), longArrayOf(1L)))
        }
        val second = cache.getOrPut(secondKey) {
            factoryCalls += 1
            PromptEmbeddings(FloatTensorData(floatArrayOf(2f), longArrayOf(1L)), FloatTensorData(floatArrayOf(2f), longArrayOf(1L)))
        }

        assertEquals(2, factoryCalls)
        assertEquals(1f, first.globalEmbedding.values[0], 0f)
        assertEquals(2f, second.globalEmbedding.values[0], 0f)
        assertEquals(first, cache.get(firstKey))
    }

    @Test
    fun auxiliaryPromptKeyIncludesPrimaryReferenceFingerprint() {
        val auxiliary = listOf(ReferenceAudioFingerprint("/ref/aux.wav", 2L, 2L))

        val first = promptKey(
            primary = ReferenceAudioFingerprint("/ref/primary-one.wav", 1L, 1L),
            auxiliary = auxiliary,
        )
        val replacementPrimary = promptKey(
            primary = ReferenceAudioFingerprint("/ref/primary-two.wav", 3L, 3L),
            auxiliary = auxiliary,
        )

        assertNotEquals(first, replacementPrimary)
    }

    @Test
    fun reusesConditioningForSameReferenceKey() {
        val cache = ReferenceConditioningCache(maxEntries = 2)
        val key = key(referenceText = "same reference")
        var factoryCalls = 0

        val first = cache.getOrPut(key) {
            factoryCalls += 1
            conditioning(1)
        }
        val second = cache.getOrPut(key) {
            factoryCalls += 1
            conditioning(2)
        }

        assertSame(first, second)
        assertEquals(1, factoryCalls)
        assertEquals(longArrayOf(1L).toList(), second.refSeq.values.toList())
    }

    @Test
    fun missesWhenReferenceTextChanges() {
        val cache = ReferenceConditioningCache(maxEntries = 2)
        var factoryCalls = 0

        cache.getOrPut(key(referenceText = "reference one")) {
            factoryCalls += 1
            conditioning(1)
        }
        val second = cache.getOrPut(key(referenceText = "reference two")) {
            factoryCalls += 1
            conditioning(2)
        }

        assertEquals(2, factoryCalls)
        assertEquals(longArrayOf(2L).toList(), second.refSeq.values.toList())
    }

    @Test
    fun missesWhenReferenceFileFingerprintChanges() {
        val cache = ReferenceConditioningCache(maxEntries = 2)
        val base = key()
        var factoryCalls = 0

        cache.getOrPut(base) {
            factoryCalls += 1
            conditioning(1)
        }
        val replacement = base.copy(referenceAudioSize = 42L, referenceAudioLastModified = 7L)
        cache.getOrPut(replacement) {
            factoryCalls += 1
            conditioning(2)
        }

        assertEquals(2, factoryCalls)
    }

    @Test
    fun missesWhenPreprocessingOrModelInterfaceChanges() {
        val cache = ReferenceConditioningCache(maxEntries = 4)
        val base = key()
        var factoryCalls = 0

        cache.getOrPut(base) {
            factoryCalls += 1
            conditioning(1)
        }
        cache.getOrPut(base.copy(preprocessingVersion = "reference-conditioning-v3")) {
            factoryCalls += 1
            conditioning(2)
        }
        cache.getOrPut(base.copy(modelInterfaceVersion = "new-interface")) {
            factoryCalls += 1
            conditioning(3)
        }

        assertEquals(3, factoryCalls)
    }

    @Test
    fun evictsLeastRecentlyUsedEntry() {
        val cache = ReferenceConditioningCache(maxEntries = 2)
        val firstKey = key(referenceAudioPath = "/ref/one.wav")
        val secondKey = key(referenceAudioPath = "/ref/two.wav")
        val thirdKey = key(referenceAudioPath = "/ref/three.wav")

        cache.getOrPut(firstKey) { conditioning(1) }
        cache.getOrPut(secondKey) { conditioning(2) }
        cache.getOrPut(firstKey) { conditioning(99) }
        cache.getOrPut(thirdKey) { conditioning(3) }

        assertEquals(conditioning(1), cache.get(firstKey))
        assertEquals(null, cache.get(secondKey))
        assertEquals(conditioning(3), cache.get(thirdKey))
    }

    private fun key(
        referenceAudioPath: String = "/ref/sample.wav",
        referenceText: String = "reference",
    ): ReferenceConditioningCacheKey =
        ReferenceConditioningCacheKey(
            characterModelId = "mansui",
            backend = ExecutionBackend.CPU,
            language = "zh",
            referenceAudioPath = referenceAudioPath,
            referenceText = referenceText,
            usesPromptEncoder = true,
        )

    private fun conditioning(value: Long): ReferenceConditioning =
        ReferenceConditioning(
            refSeq = LongTensorData(longArrayOf(value), longArrayOf(1L, 1L)),
            refBert = FloatTensorData(floatArrayOf(value.toFloat()), longArrayOf(1L, 1L)),
            sslContent = FloatTensorData(floatArrayOf(value.toFloat()), longArrayOf(1L, 1L, 1L)),
            globalEmbedding = FloatTensorData(floatArrayOf(value.toFloat()), longArrayOf(1L, 1L)),
            advancedGlobalEmbedding = FloatTensorData(floatArrayOf(value.toFloat()), longArrayOf(1L, 1L)),
            refAudio32k = null,
        )

    private fun promptKey(
        primary: ReferenceAudioFingerprint,
        auxiliary: List<ReferenceAudioFingerprint>,
    ): PromptConditioningCacheKey =
        PromptConditioningCacheKey(
            characterModelId = "mansui",
            backend = ExecutionBackend.CPU,
            modelInterfaceVersion = "prompt-v2",
            preprocessingVersion = "reference-conditioning-v2",
            conditioningRole = InferenceModelRole.PROMPT_ENCODER,
            primaryReferenceAudio = primary,
            auxiliaryReferenceAudioFingerprints = auxiliary,
        )
}
