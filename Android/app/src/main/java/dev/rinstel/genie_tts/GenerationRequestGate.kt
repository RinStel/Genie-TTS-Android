package dev.rinstel.genie_tts

import java.util.concurrent.atomic.AtomicBoolean

/** Closes the enqueue-to-first-progress race for UI and HTTP synthesis requests. */
class GenerationRequestGate {
    private val claimed = AtomicBoolean(false)

    fun tryAcquire(): Boolean = claimed.compareAndSet(false, true)

    fun release() {
        claimed.set(false)
    }

    fun isClaimed(): Boolean = claimed.get()
}
