package dev.rinstel.genie_tts

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.Settings
import android.view.LayoutInflater
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.textfield.TextInputEditText
import dev.rinstel.genie_tts.api.LocalHttpApiServer
import dev.rinstel.genie_tts.inference.ModelAssetRepository
import dev.rinstel.genie_tts.inference.RuntimeAssetRepository
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class SettingsActivity : AppCompatActivity() {
    private lateinit var topAppBar: MaterialToolbar
    private lateinit var themeSelector: AutoCompleteTextView
    private lateinit var apiEnabledSwitch: MaterialSwitch
    private lateinit var apiPortInput: TextInputEditText
    private lateinit var applyApiPortButton: MaterialButton
    private lateinit var apiAddressValue: TextView
    private lateinit var apiStatusValue: TextView
    private lateinit var apiExamplesContainer: LinearLayout
    private lateinit var batteryOptimizationLabel: TextView
    private lateinit var batteryOptimizationValue: TextView
    private lateinit var openBatteryOptimizationButton: MaterialButton
    private lateinit var importRuntimePackageButton: MaterialButton
    private lateinit var importCharacterModelButton: MaterialButton
    private lateinit var resourceImportProgress: LinearProgressIndicator
    private lateinit var resourceImportStatusValue: TextView
    private lateinit var modelPathValue: TextView
    private lateinit var runtimePathValue: TextView
    private lateinit var outputPathValue: TextView

    private lateinit var settingsRepository: AppSettingsRepository
    private lateinit var modelRepository: ModelAssetRepository
    private lateinit var runtimeRepository: RuntimeAssetRepository

    private val importExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var resourceImportInProgress = false
    private var resourceImportStatus: String? = null

    private val runtimePackagePicker = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        uri?.let { importPackage(ResourcePackageImporter.PackageType.RUNTIME, it) }
    }

    private val characterModelPackagePicker = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        uri?.let { importPackage(ResourcePackageImporter.PackageType.CHARACTER, it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        settingsRepository = AppSettingsRepository(this)
        modelRepository = ModelAssetRepository(this)
        runtimeRepository = RuntimeAssetRepository(this)

        bindViews()
        setSupportActionBar(topAppBar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        topAppBar.setNavigationOnClickListener { finish() }
        applySystemBarInsets()
        configureThemeSelector()
        render()
        configureActions()
    }

    override fun onResume() {
        super.onResume()
        if (::settingsRepository.isInitialized) {
            render()
        }
    }

    override fun onDestroy() {
        importExecutor.shutdownNow()
        super.onDestroy()
    }

    private fun bindViews() {
        topAppBar = findViewById(R.id.topAppBar)
        themeSelector = findViewById(R.id.themeSelector)
        apiEnabledSwitch = findViewById(R.id.apiEnabledSwitch)
        apiPortInput = findViewById(R.id.apiPortInput)
        applyApiPortButton = findViewById(R.id.applyApiPortButton)
        apiAddressValue = findViewById(R.id.apiAddressValue)
        apiStatusValue = findViewById(R.id.apiStatusValue)
        apiExamplesContainer = findViewById(R.id.apiExamplesContainer)
        batteryOptimizationLabel = findViewById(R.id.batteryOptimizationLabel)
        batteryOptimizationValue = findViewById(R.id.batteryOptimizationValue)
        openBatteryOptimizationButton = findViewById(R.id.openBatteryOptimizationButton)
        importRuntimePackageButton = findViewById(R.id.importRuntimePackageButton)
        importCharacterModelButton = findViewById(R.id.importCharacterModelButton)
        resourceImportProgress = findViewById(R.id.resourceImportProgress)
        resourceImportStatusValue = findViewById(R.id.resourceImportStatusValue)
        modelPathValue = findViewById(R.id.modelPathValue)
        runtimePathValue = findViewById(R.id.runtimePathValue)
        outputPathValue = findViewById(R.id.outputPathValue)
    }

    private fun applySystemBarInsets() {
        val rootLayout = findViewById<android.view.View>(R.id.rootLayout)
        val rootTopPadding = rootLayout.paddingTop
        val rootBottomPadding = rootLayout.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            rootLayout.updatePadding(
                top = rootTopPadding + systemBars.top,
                bottom = rootBottomPadding + systemBars.bottom,
            )
            insets
        }
        ViewCompat.requestApplyInsets(rootLayout)
    }

    private fun configureThemeSelector() {
        themeSelector.setAdapter(
            ArrayAdapter(
                this,
                android.R.layout.simple_list_item_1,
                THEME_OPTIONS.map(ThemeOption::label),
            ),
        )
    }

    private fun render() {
        val selectedTheme = THEME_OPTIONS.firstOrNull {
            it.mode == settingsRepository.themeMode
        } ?: THEME_OPTIONS.first()
        themeSelector.setText(selectedTheme.label, false)
        apiEnabledSwitch.isChecked = settingsRepository.apiEnabled
        apiPortInput.setText(settingsRepository.apiPort.toString())
        apiPortInput.setSelection(apiPortInput.text?.length ?: 0)
        applyApiPortButton.text = "\u5e94\u7528\u7aef\u53e3"
        apiAddressValue.text = "127.0.0.1:${settingsRepository.apiPort}"
        val apiStatus = LocalHttpApiServer.currentStatus()
        apiStatusValue.text = when {
            !settingsRepository.apiEnabled -> getString(R.string.prototype_settings_api_status_disabled)
            apiStatus.running -> getString(R.string.prototype_settings_api_status_running, apiStatus.port)
            apiStatus.lastError.isNullOrBlank() -> getString(R.string.prototype_settings_api_status_enabled)
            else -> getString(R.string.prototype_settings_api_status_error, apiStatus.lastError)
        }
        renderApiExamples(settingsRepository.apiPort)
        batteryOptimizationLabel.text = "\u540e\u53f0\u7535\u6c60\u9650\u5236"
        val ignoringBatteryOptimizations = isIgnoringBatteryOptimizations()
        batteryOptimizationValue.text = if (ignoringBatteryOptimizations) {
            "\u5df2\u8bbe\u4e3a\u4e0d\u53d7\u9650\u5236\uff0c\u540e\u53f0\u63a8\u7406\u4e0d\u4f1a\u88ab\u5e38\u89c4\u7701\u7535\u7b56\u7565\u964d\u901f\u3002"
        } else {
            "\u672a\u8c41\u514d\uff1a\u7cfb\u7edf\u53ef\u80fd\u5728\u540e\u53f0\u964d\u4f4e CPU/NPU \u8c03\u5ea6\u3002"
        }
        openBatteryOptimizationButton.text = if (ignoringBatteryOptimizations) {
            "\u67e5\u770b\u7535\u6c60\u8bbe\u7f6e"
        } else {
            "\u5141\u8bb8\u540e\u53f0\u4e0d\u53d7\u9650\u5236"
        }
        modelPathValue.text = modelRepository.modelRootPath()
        runtimePathValue.text = runtimeRepository.runtimeRoot().absolutePath
        outputPathValue.text = runtimeRepository.outputRoot().absolutePath
        resourceImportStatusValue.text = resourceImportStatus
            ?: getString(R.string.prototype_settings_import_idle)
        resourceImportProgress.visibility = if (resourceImportInProgress) View.VISIBLE else View.GONE
        importRuntimePackageButton.isEnabled = !resourceImportInProgress
        importCharacterModelButton.isEnabled = !resourceImportInProgress
    }

    private fun configureActions() {
        themeSelector.setOnItemClickListener { _, _, position, _ ->
            val theme = THEME_OPTIONS[position]
            settingsRepository.themeMode = theme.mode
            settingsRepository.applyThemeMode()
        }
        apiEnabledSwitch.setOnCheckedChangeListener { _, isChecked ->
            settingsRepository.apiEnabled = isChecked
            notifyBackendServiceOfSettingsChange()
            render()
        }
        applyApiPortButton.setOnClickListener {
            applyApiPortFromInput()
        }
        apiPortInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                applyApiPortFromInput()
                true
            } else {
                false
            }
        }
        apiPortInput.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                applyApiPortFromInput()
            }
        }
        openBatteryOptimizationButton.setOnClickListener {
            openBatteryOptimizationSettings()
        }
        importRuntimePackageButton.setOnClickListener {
            runtimePackagePicker.launch(PACKAGE_MIME_TYPES)
        }
        importCharacterModelButton.setOnClickListener {
            characterModelPackagePicker.launch(PACKAGE_MIME_TYPES)
        }
    }

    private fun importPackage(
        packageType: ResourcePackageImporter.PackageType,
        uri: Uri,
    ) {
        if (resourceImportInProgress) return
        resourceImportInProgress = true
        resourceImportStatus = getString(R.string.prototype_settings_import_running)
        render()

        importExecutor.execute {
            try {
                val result = contentResolver.openInputStream(uri)?.use { input ->
                    val importer = ResourcePackageImporter(
                        requireNotNull(getExternalFilesDir(null)),
                    )
                    when (packageType) {
                        ResourcePackageImporter.PackageType.RUNTIME -> importer.importRuntime(input)
                        ResourcePackageImporter.PackageType.CHARACTER -> importer.importCharacter(input)
                    }
                } ?: error("无法读取所选文件。")
                runOnUiThread { finishPackageImport(result) }
            } catch (error: Throwable) {
                runOnUiThread {
                    resourceImportInProgress = false
                    resourceImportStatus = getString(
                        R.string.prototype_settings_import_failed,
                        error.message ?: error.javaClass.simpleName,
                    )
                    render()
                    showImportMessage(resourceImportStatus.orEmpty())
                }
            }
        }
    }

    private fun finishPackageImport(result: ResourcePackageImporter.ImportResult) {
        resourceImportInProgress = false
        val size = formatBytes(result.totalBytes)
        resourceImportStatus = when (result.packageType) {
            ResourcePackageImporter.PackageType.RUNTIME -> getString(
                R.string.prototype_settings_import_runtime_success,
                result.fileCount,
                size,
            )
            ResourcePackageImporter.PackageType.CHARACTER -> getString(
                R.string.prototype_settings_import_character_success,
                result.characterIds.joinToString(", "),
                result.fileCount,
                size,
            )
        }
        render()
        notifyBackendServiceOfResourceImport()
        showImportMessage(resourceImportStatus.orEmpty())
    }

    private fun notifyBackendServiceOfResourceImport() {
        runCatching {
            ContextCompat.startForegroundService(
                this,
                GenieBackendService.reloadResourcesIntent(this),
            )
        }
    }

    private fun showImportMessage(message: String) {
        Snackbar.make(findViewById(R.id.rootLayout), message, Snackbar.LENGTH_LONG).show()
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1024L * 1024L * 1024L -> String.format(
            Locale.US,
            "%.1f GiB",
            bytes.toDouble() / (1024L * 1024L * 1024L),
        )
        bytes >= 1024L * 1024L -> String.format(
            Locale.US,
            "%.1f MiB",
            bytes.toDouble() / (1024L * 1024L),
        )
        else -> String.format(Locale.US, "%.1f KiB", bytes.toDouble() / 1024L)
    }

    private fun notifyBackendServiceOfSettingsChange() {
        ContextCompat.startForegroundService(this, GenieBackendService.serviceIntent(this))
    }

    private fun renderApiExamples(port: Int) {
        apiExamplesContainer.removeAllViews()
        val base = "http://127.0.0.1:$port"
        val defaults = GenieBackendService.companionDefaults
        val defaultText = "我想说的是，你怎么看待这个疯狂的世界呢？"
        val backend = defaults?.backend?.name ?: "CPU"
        val modelId = defaults?.modelId ?: "<modelId>"
        val language = defaults?.language ?: "zh"
        val refAudio = defaults?.referenceAudioPath ?: "<referenceAudioPath>"
        val refText = defaults?.referenceText ?: "<referenceText>"
        val maxSteps = defaults?.maxDecoderSteps ?: 500
        val examples = buildList {
            add(ApiExample(
                label = getString(R.string.prototype_settings_api_example_health),
                command = "$base/health",
            ))
            add(ApiExample(
                label = getString(R.string.prototype_settings_api_example_models),
                command = "$base/models",
            ))
            add(ApiExample(
                label = getString(R.string.prototype_settings_api_example_status),
                command = "$base/status",
            ))
            add(ApiExample(
                label = getString(R.string.prototype_settings_api_example_result),
                command = "$base/result/latest",
            ))
            add(ApiExample(
                label = getString(R.string.prototype_settings_api_example_infer_get),
                command = "$base/infer?backend=$backend&modelId=$modelId&language=$language&text=$defaultText&referenceAudioPath=$refAudio&referenceText=$refText&maxDecoderSteps=$maxSteps",
            ))
            add(ApiExample(
                label = getString(R.string.prototype_settings_api_example_infer_post),
                command = """
                    curl -o output.wav -X POST $base/infer \
                      -H "Content-Type: application/json" \
                      -d '{"backend":"$backend","modelId":"$modelId","language":"$language","text":"$defaultText","referenceAudioPath":"$refAudio","referenceText":"$refText","maxDecoderSteps":$maxSteps,"auxReferenceAudioPaths":[]}'
                """.trimIndent(),
            ))
        }
        val inflater = LayoutInflater.from(this)
        for (example in examples) {
            val view = inflater.inflate(R.layout.item_api_example, apiExamplesContainer, false)
            view.findViewById<TextView>(R.id.exampleLabel).text = example.label
            view.findViewById<TextView>(R.id.exampleCommand).text = example.command
            view.findViewById<MaterialButton>(R.id.copyButton).setOnClickListener {
                copyToClipboard(example.command)
            }
            apiExamplesContainer.addView(view)
        }
    }

    private fun copyToClipboard(text: String) {
        val clipboard = getSystemService(ClipboardManager::class.java)
        clipboard?.setPrimaryClip(ClipData.newPlainText("curl", text))
        val rootView = findViewById<View>(R.id.rootLayout)
        Snackbar.make(rootView, R.string.prototype_settings_api_copied, Snackbar.LENGTH_SHORT).show()
    }

    private fun applyApiPortFromInput() {
        val oldPort = settingsRepository.apiPort
        val value = apiPortInput.text?.toString()?.toIntOrNull()
        settingsRepository.apiPort = value ?: AppSettingsRepository.DEFAULT_API_PORT
        val newPort = settingsRepository.apiPort
        if (newPort != oldPort || settingsRepository.apiEnabled) {
            notifyBackendServiceOfSettingsChange()
        }
        render()
    }

    private fun isIgnoringBatteryOptimizations(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            return true
        }
        val powerManager = getSystemService(PowerManager::class.java)
        return powerManager?.isIgnoringBatteryOptimizations(packageName) ?: false
    }

    private fun openBatteryOptimizationSettings() {
        val intent = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !isIgnoringBatteryOptimizations()) {
            Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                data = Uri.parse("package:$packageName")
            }
        } else {
            Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
        }
        runCatching { startActivity(intent) }.onFailure {
            startActivity(Intent(Settings.ACTION_SETTINGS))
        }
    }

    companion object {
        private val PACKAGE_MIME_TYPES = arrayOf("application/zip", "application/octet-stream")

        private val THEME_OPTIONS = listOf(
            ThemeOption("\u8ddf\u968f\u7cfb\u7edf", AppSettingsRepository.ThemeMode.SYSTEM),
            ThemeOption("\u6d45\u8272", AppSettingsRepository.ThemeMode.LIGHT),
            ThemeOption("\u6df1\u8272", AppSettingsRepository.ThemeMode.DARK),
        )
    }

    private data class ThemeOption(
        val label: String,
        val mode: AppSettingsRepository.ThemeMode,
    )

    private data class ApiExample(
        val label: String,
        val command: String,
    )
}
