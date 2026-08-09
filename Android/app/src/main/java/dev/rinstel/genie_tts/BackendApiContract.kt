package dev.rinstel.genie_tts

object BackendApiContract {
    const val ACTION_BIND_API = "dev.rinstel.genie_tts.action.BIND_BACKEND_API"

    const val MSG_REGISTER_CLIENT = 1
    const val MSG_UNREGISTER_CLIENT = 2
    const val MSG_GET_STATUS = 3
    const val MSG_SYNTHESIZE = 4
    const val MSG_STATE_CHANGED = 5

    const val KEY_BACKEND = "backend"
    const val KEY_MODEL_ID = "model_id"
    const val KEY_LANGUAGE = "language"
    const val KEY_PROMPT_LANGUAGE = "prompt_language"
    const val KEY_SYNTHESIS_TEXT = "synthesis_text"
    const val KEY_REFERENCE_AUDIO_PATH = "reference_audio_path"
    const val KEY_REFERENCE_TEXT = "reference_text"
    const val KEY_AUXILIARY_REFERENCE_AUDIO_PATHS = "auxiliary_reference_audio_paths"
    const val KEY_MAX_DECODER_STEPS = "max_decoder_steps"

    const val KEY_STAGE = "stage"
    const val KEY_MESSAGE = "message"
    const val KEY_BUSY = "busy"
    const val KEY_REQUESTED_BACKEND = "requested_backend"
    const val KEY_RESOLVED_BACKEND = "resolved_backend"
    const val KEY_RUNTIME_LABEL = "runtime_label"
    const val KEY_INITIALIZED_MODEL_ID = "initialized_model_id"
    const val KEY_LATEST_OUTPUT_FILE_PATH = "latest_output_file_path"
}
