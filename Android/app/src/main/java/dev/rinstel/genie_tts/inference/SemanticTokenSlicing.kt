package dev.rinstel.genie_tts.inference

object SemanticTokenSlicing {
    const val TERMINAL_TOKEN: Long = 0L

    /**
     * Keeps the decoder's appended suffix and removes only special outputs.
     * The stop flag is independent from the final token, so a normal token on
     * the stopping step remains valid vocoder input.
     */
    fun completedTokens(
        decoderTokenOutputs: LongArray,
        completedCount: Int,
        terminalToken: Long = TERMINAL_TOKEN,
    ): LongArray {
        val count = completedCount.coerceIn(0, decoderTokenOutputs.size)
        if (count == 0) return longArrayOf()

        val firstCompleted = decoderTokenOutputs.size - count
        var semantic = decoderTokenOutputs.copyOfRange(firstCompleted, decoderTokenOutputs.size)
        val firstSpecialToken = semantic.indexOfFirst { it >= 1024L }
        if (firstSpecialToken >= 0) {
            semantic = semantic.copyOf(firstSpecialToken)
        }
        if (semantic.isNotEmpty() && semantic.last() == terminalToken) {
            semantic = semantic.copyOf(semantic.lastIndex)
        }
        return semantic
    }
}
