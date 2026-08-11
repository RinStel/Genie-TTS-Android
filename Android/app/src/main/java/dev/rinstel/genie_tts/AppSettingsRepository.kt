package dev.rinstel.genie_tts

import android.content.Context
import androidx.appcompat.app.AppCompatDelegate

class AppSettingsRepository(
    private val storage: Storage,
) {
    constructor(context: Context) : this(
        SharedPreferencesStorage(
            context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE),
        ),
    )

    enum class ThemeMode(
        val storageValue: String,
        val nightMode: Int,
    ) {
        SYSTEM("system", AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM),
        LIGHT("light", AppCompatDelegate.MODE_NIGHT_NO),
        DARK("dark", AppCompatDelegate.MODE_NIGHT_YES),
    }

    interface Storage {
        fun getString(key: String, defaultValue: String): String
        fun getBoolean(key: String, defaultValue: Boolean): Boolean
        fun getInt(key: String, defaultValue: Int): Int
        fun putString(key: String, value: String)
        fun putBoolean(key: String, value: Boolean)
        fun putInt(key: String, value: Int)
    }

    var themeMode: ThemeMode
        get() = ThemeMode.entries.firstOrNull {
            it.storageValue == storage.getString(KEY_THEME_MODE, ThemeMode.SYSTEM.storageValue)
        } ?: ThemeMode.SYSTEM
        set(value) {
            storage.putString(KEY_THEME_MODE, value.storageValue)
        }

    var apiEnabled: Boolean
        get() = storage.getBoolean(KEY_API_ENABLED, true)
        set(value) {
            storage.putBoolean(KEY_API_ENABLED, value)
        }

    var apiPort: Int
        get() = storage.getInt(KEY_API_PORT, DEFAULT_API_PORT).coerceIn(MIN_API_PORT, MAX_API_PORT)
        set(value) {
            storage.putInt(KEY_API_PORT, value.coerceIn(MIN_API_PORT, MAX_API_PORT))
        }

    var t2sCpuThreads: Int
        get() = cpuThreadSetting(KEY_T2S_CPU_THREADS)
        set(value) {
            storage.putInt(KEY_T2S_CPU_THREADS, value.coerceAtLeast(AUTO_CPU_THREADS))
        }

    var vocoderCpuThreads: Int
        get() = cpuThreadSetting(KEY_VOCODER_CPU_THREADS)
        set(value) {
            storage.putInt(KEY_VOCODER_CPU_THREADS, value.coerceAtLeast(AUTO_CPU_THREADS))
        }

    fun cpuThreadOptions(availableProcessors: Int): List<Int> {
        val processors = availableProcessors.coerceAtLeast(1)
        return buildList {
            add(AUTO_CPU_THREADS)
            addAll(1..processors)
        }
    }

    fun applyThemeMode() {
        AppCompatDelegate.setDefaultNightMode(themeMode.nightMode)
    }

    private fun cpuThreadSetting(key: String): Int =
        storage.getInt(key, AUTO_CPU_THREADS).coerceAtLeast(AUTO_CPU_THREADS)

    private class SharedPreferencesStorage(
        private val preferences: android.content.SharedPreferences,
    ) : Storage {
        override fun getString(key: String, defaultValue: String): String =
            preferences.getString(key, defaultValue) ?: defaultValue

        override fun getBoolean(key: String, defaultValue: Boolean): Boolean =
            preferences.getBoolean(key, defaultValue)

        override fun getInt(key: String, defaultValue: Int): Int =
            preferences.getInt(key, defaultValue)

        override fun putString(key: String, value: String) {
            preferences.edit().putString(key, value).apply()
        }

        override fun putBoolean(key: String, value: Boolean) {
            preferences.edit().putBoolean(key, value).apply()
        }

        override fun putInt(key: String, value: Int) {
            preferences.edit().putInt(key, value).apply()
        }
    }

    companion object {
        private const val PREFERENCES_NAME = "genie_tts_settings"
        private const val KEY_THEME_MODE = "theme_mode"
        private const val KEY_API_ENABLED = "api_enabled"
        private const val KEY_API_PORT = "api_port"
        private const val KEY_T2S_CPU_THREADS = "t2s_cpu_threads"
        private const val KEY_VOCODER_CPU_THREADS = "vocoder_cpu_threads"
        private const val MIN_API_PORT = 1024
        private const val MAX_API_PORT = 65535
        const val AUTO_CPU_THREADS = 0
        const val DEFAULT_API_PORT = 16580
    }
}
