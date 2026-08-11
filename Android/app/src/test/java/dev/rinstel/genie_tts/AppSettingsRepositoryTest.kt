package dev.rinstel.genie_tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AppSettingsRepositoryTest {
    @Test
    fun usesStableDefaults() {
        val repository = AppSettingsRepository(FakeStorage())

        assertEquals(AppSettingsRepository.ThemeMode.SYSTEM, repository.themeMode)
        assertTrue(repository.apiEnabled)
        assertEquals(AppSettingsRepository.DEFAULT_API_PORT, repository.apiPort)
        assertEquals(AppSettingsRepository.AUTO_CPU_THREADS, repository.t2sCpuThreads)
        assertEquals(AppSettingsRepository.AUTO_CPU_THREADS, repository.vocoderCpuThreads)
        assertEquals(
            listOf(0, 1, 2, 3, 4),
            repository.cpuThreadOptions(4),
        )
    }

    @Test
    fun persistsThemeAndApiSettings() {
        val storage = FakeStorage()
        val repository = AppSettingsRepository(storage)

        repository.themeMode = AppSettingsRepository.ThemeMode.DARK
        repository.apiEnabled = false
        repository.apiPort = 19090
        repository.t2sCpuThreads = 3
        repository.vocoderCpuThreads = 8

        assertEquals(AppSettingsRepository.ThemeMode.DARK, repository.themeMode)
        assertEquals(false, repository.apiEnabled)
        assertEquals(19090, repository.apiPort)
        assertEquals(3, repository.t2sCpuThreads)
        assertEquals(8, repository.vocoderCpuThreads)
    }

    @Test
    fun clampsLegacyStoredPortsBeforeTheServerUsesThem() {
        val storage = FakeStorage()
        storage.seedInt("api_port", 80)

        assertEquals(1024, AppSettingsRepository(storage).apiPort)
    }

    @Test
    fun normalizesNegativeCpuThreadValuesToAutomaticMode() {
        val storage = FakeStorage()
        storage.seedInt("t2s_cpu_threads", -1)
        storage.seedInt("vocoder_cpu_threads", -4)

        val repository = AppSettingsRepository(storage)

        assertEquals(AppSettingsRepository.AUTO_CPU_THREADS, repository.t2sCpuThreads)
        assertEquals(AppSettingsRepository.AUTO_CPU_THREADS, repository.vocoderCpuThreads)
    }

    private class FakeStorage : AppSettingsRepository.Storage {
        private val strings = mutableMapOf<String, String>()
        private val booleans = mutableMapOf<String, Boolean>()
        private val ints = mutableMapOf<String, Int>()

        override fun getString(key: String, defaultValue: String): String =
            strings[key] ?: defaultValue

        override fun getBoolean(key: String, defaultValue: Boolean): Boolean =
            booleans[key] ?: defaultValue

        override fun getInt(key: String, defaultValue: Int): Int =
            ints[key] ?: defaultValue

        override fun putString(key: String, value: String) {
            strings[key] = value
        }

        override fun putBoolean(key: String, value: Boolean) {
            booleans[key] = value
        }

        override fun putInt(key: String, value: Int) {
            ints[key] = value
        }

        fun seedInt(key: String, value: Int) {
            ints[key] = value
        }
    }
}
