package dev.rinstel.genie_tts

fun interface BackendServiceListener {
    fun onStateChanged(state: BackendServiceState)
}
