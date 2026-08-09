package dev.rinstel.genie_tts

import java.util.Locale

data class BackendProgressModel(
    val percent: Int,
    val indeterminate: Boolean,
)

object BackendStatusFormatter {
    fun format(
        state: BackendServiceState,
        isPlaying: Boolean,
    ): String {
        val lines = mutableListOf<String>()
        lines += buildBackendLine(state)
        if (state.runtimeLabel.isNotBlank()) {
            lines += "\u8fd0\u884c\u65f6: ${state.runtimeLabel}"
        }
        lines += state.message.ifBlank { "\u5f85\u547d" }
        state.progressLabel?.takeIf { it.isNotBlank() && it != state.message }?.let { lines += it }
        state.generationDurationMs?.let { lines += "\u8017\u65f6: ${formatDuration(it)}" }
        if (isPlaying) {
            lines += "\u64ad\u653e\u4e2d"
        }
        return lines.joinToString(separator = "\n")
    }

    fun progressModel(state: BackendServiceState): BackendProgressModel {
        val explicitPercent = state.progressPercent?.coerceIn(0, 100)
        if (explicitPercent != null) {
            return BackendProgressModel(
                percent = explicitPercent,
                indeterminate = false,
            )
        }
        if (state.busy) {
            return BackendProgressModel(
                percent = 0,
                indeterminate = true,
            )
        }
        return BackendProgressModel(
            percent = state.stage.progressPercent,
            indeterminate = false,
        )
    }

    private fun buildBackendLine(state: BackendServiceState): String {
        val requested = state.requestedBackend.label
        val resolved = state.resolvedBackend.label
        return if (requested == resolved) {
            "\u540e\u7aef: $resolved"
        } else {
            "\u540e\u7aef: $requested -> $resolved"
        }
    }

    private fun formatDuration(durationMs: Long): String =
        if (durationMs < 1_000L) {
            "${durationMs}ms"
        } else {
            String.format(Locale.US, "%.1fs", durationMs / 1_000.0)
        }
}
