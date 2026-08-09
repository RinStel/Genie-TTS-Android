package dev.rinstel.genie_tts

data class AuxiliaryReferenceAudio(
    val backendAudioPath: String,
    val displayAudioName: String,
)

data class ReferenceInputState(
    val source: Source,
    val backendAudioPath: String,
    val displayAudioName: String,
    val referenceText: String,
    val hint: String,
    val auxiliaryReferences: List<AuxiliaryReferenceAudio> = emptyList(),
) {
    enum class Source {
        NONE,
        DEFAULT_SAMPLE,
        MANUAL_OVERRIDE,
    }

    val isReady: Boolean
        get() = backendAudioPath.isNotBlank() && referenceText.isNotBlank()

    fun withReferenceText(value: String): ReferenceInputState =
        copy(referenceText = value)

    fun withAuxiliaryReferences(value: List<AuxiliaryReferenceAudio>): ReferenceInputState =
        copy(auxiliaryReferences = value.toList())

    companion object {
        fun empty(): ReferenceInputState = ReferenceInputState(
            source = Source.NONE,
            backendAudioPath = "",
            displayAudioName = "",
            referenceText = "",
            hint = "",
        )

        fun defaultSample(
            backendAudioPath: String,
            displayAudioName: String,
            referenceText: String,
            hint: String,
        ): ReferenceInputState = ReferenceInputState(
            source = Source.DEFAULT_SAMPLE,
            backendAudioPath = backendAudioPath,
            displayAudioName = displayAudioName,
            referenceText = referenceText,
            hint = hint,
        )

        fun manualOverride(
            backendAudioPath: String,
            displayAudioName: String,
            referenceText: String,
            hint: String,
            auxiliaryReferences: List<AuxiliaryReferenceAudio> = emptyList(),
        ): ReferenceInputState = ReferenceInputState(
            source = Source.MANUAL_OVERRIDE,
            backendAudioPath = backendAudioPath,
            displayAudioName = displayAudioName,
            referenceText = referenceText,
            hint = hint,
            auxiliaryReferences = auxiliaryReferences,
        )
    }
}
