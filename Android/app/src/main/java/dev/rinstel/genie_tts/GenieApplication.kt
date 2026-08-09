package dev.rinstel.genie_tts

import android.app.Application
import com.google.android.material.color.DynamicColors

class GenieApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        AppSettingsRepository(this).applyThemeMode()
        DynamicColors.applyToActivitiesIfAvailable(this)
    }
}
