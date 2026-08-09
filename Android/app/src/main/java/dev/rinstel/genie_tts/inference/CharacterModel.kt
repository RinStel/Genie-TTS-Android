package dev.rinstel.genie_tts.inference

data class CharacterModel(
    val id: String,
    val displayName: String,
    val modelFiles: RequiredModelFiles,
) {
    val relativeModelDirectory: String
        get() = modelFiles.assetSubdirectory
}
