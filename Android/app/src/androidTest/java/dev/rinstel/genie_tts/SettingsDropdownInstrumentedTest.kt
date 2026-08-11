package dev.rinstel.genie_tts

import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.espresso.matcher.RootMatchers.isPlatformPopup
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SettingsDropdownInstrumentedTest {
    @Test
    fun threadSelectorOpensItsOptions() {
        ActivityScenario.launch(SettingsActivity::class.java).use { scenario ->
            onView(withId(R.id.t2sCpuThreadSelector)).perform(click())
            onView(withText(R.string.prototype_settings_cpu_threads_auto))
                .inRoot(isPlatformPopup())
                .check(matches(isDisplayed()))
        }
    }
}
