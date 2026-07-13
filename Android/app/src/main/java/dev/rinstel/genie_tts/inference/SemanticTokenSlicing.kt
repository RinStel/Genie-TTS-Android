package dev.rinstel.genie_tts.inference

object SemanticTokenSlicing {
    const val TERMINAL_TOKEN: Long = 0L

    /**
     * Excludes the final decoder output from the completed generated tokens.
     * That output is the decoder boundary whether it is EOS, a max-step value,
     * or EOS already replaced with [terminalToken].
     */
    fun completedTokens(
        decoderTokenOutputs: LongArray,
        completedCount: Int,
        terminalToken: Long = TERMINAL_TOKEN,
    ): LongArray {
        val count = completedCount.coerceIn(0, decoderTokenOutputs.size)
        if (count == 0) return longArrayOf()

        val firstCompleted = decoderTokenOutputs.size - count
        val terminalIndex = decoderTokenOutputs.lastIndex
        val completed = decoderTokenOutputs.copyOfRange(firstCompleted, terminalIndex + 1)
        return if (completed.last() == terminalToken) {
            completed.copyOf(completed.lastIndex)
        } else {
            completed.copyOf(completed.lastIndex)
        }
    }
}
