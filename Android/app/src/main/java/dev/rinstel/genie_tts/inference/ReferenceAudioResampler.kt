package dev.rinstel.genie_tts.inference

/** Native band-limited resampler used for reference conditioning audio. */
object ReferenceAudioResampler {
    private val nativeAvailable: Boolean

    init {
        nativeAvailable = runCatching {
            System.loadLibrary("native_resampler")
            true
        }.getOrDefault(false)
    }

    fun resample(input: FloatArray, sourceRate: Int, targetRate: Int): FloatArray {
        require(sourceRate > 0) { "Source sample rate must be positive." }
        require(targetRate > 0) { "Target sample rate must be positive." }
        if (input.isEmpty() || sourceRate == targetRate) return input
        check(nativeAvailable) {
            "Native reference resampler is unavailable; linear reference resampling is not supported."
        }
        return nativeResample(input, sourceRate, targetRate)
    }

    @JvmStatic
    private external fun nativeResample(
        input: FloatArray,
        sourceRate: Int,
        targetRate: Int,
    ): FloatArray
}
