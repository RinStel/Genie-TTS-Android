package dev.rinstel.genie_tts.inference

object CharacterModelCatalog {
    fun v2ProPlusFromCharacterIds(characterIds: List<String>): List<CharacterModel> =
        characterIds
            .distinct()
            .sortedWith(compareBy<String> { if (it == "mansui") 0 else 1 }.thenBy { it })
            .map { characterId ->
                CharacterModel(
                    id = characterId,
                    displayName = characterId,
                    modelFiles = RequiredModelFiles.v2ProPlusCharacter(characterId),
                )
            }

    fun v2FromCharacterIds(characterIds: List<String>): List<CharacterModel> =
        characterIds
            .distinct()
            .sorted()
            .map { characterId ->
                CharacterModel(
                    id = characterId,
                    displayName = characterId,
                    modelFiles = RequiredModelFiles.v2Character(characterId),
                )
            }
}
