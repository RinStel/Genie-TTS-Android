package dev.rinstel.genie_tts

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File

class ReferenceAudioPicker(
    private val context: Context,
) {
    data class PickedReferenceAudio(
        val cachedFile: File,
        val displayName: String,
    )

    fun copyToAppCache(uri: Uri): PickedReferenceAudio {
        val displayName = resolveDisplayName(uri)
        val cacheDirectory = File(context.cacheDir, "reference_audio").apply { mkdirs() }
        val outputFile = File(cacheDirectory, "${System.currentTimeMillis()}-$displayName")
        context.contentResolver.openInputStream(uri)?.use { input ->
            outputFile.outputStream().use { output ->
                input.copyTo(output)
            }
        } ?: error("Unable to open selected audio.")
        return PickedReferenceAudio(
            cachedFile = outputFile,
            displayName = displayName,
        )
    }

    private fun resolveDisplayName(uri: Uri): String {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index >= 0) {
                        val value = cursor.getString(index)?.trim().orEmpty()
                        if (value.isNotBlank()) {
                            return sanitizeFilename(value)
                        }
                    }
                }
            }
        return sanitizeFilename(uri.lastPathSegment?.substringAfterLast('/') ?: "reference.wav")
    }

    private fun sanitizeFilename(name: String): String =
        name.replace(Regex("""[\\/:*?"<>|]"""), "_")
}
