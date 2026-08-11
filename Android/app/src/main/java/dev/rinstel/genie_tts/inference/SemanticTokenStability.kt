package dev.rinstel.genie_tts.inference

/** Bounds retries for the decoder's stochastic early-stop path. */
object SemanticTokenStability {
    const val MAX_RETRIES = 2

    private const val MIN_ACCEPTED_TOKENS = 6
    private const val MAX_TEXT_DERIVED_TOKENS = 16

    fun minimumAcceptedTokens(textTokenCount: Int): Int =
        maxOf(
            MIN_ACCEPTED_TOKENS,
            minOf(MAX_TEXT_DERIVED_TOKENS, textTokenCount.coerceAtLeast(0)),
        )

    fun shouldRetry(
        candidateTokenCount: Int,
        textTokenCount: Int,
        retryCount: Int,
    ): Boolean =
        retryCount in 0 until MAX_RETRIES &&
            candidateTokenCount < minimumAcceptedTokens(textTokenCount)
}
