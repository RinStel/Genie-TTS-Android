package dev.rinstel.genie_tts.inference

import java.io.File

data class ReferenceExample(
    val audioFile: File,
    val referenceText: String,
)
