package dev.rinstel.genie_tts

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.content.Intent
import android.view.Menu
import android.view.MenuItem
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import dev.rinstel.genie_tts.inference.BackendCatalog
import dev.rinstel.genie_tts.inference.BackendOption
import dev.rinstel.genie_tts.inference.CharacterModel
import dev.rinstel.genie_tts.inference.ExecutionBackend
import dev.rinstel.genie_tts.inference.GenerationRequest
import dev.rinstel.genie_tts.inference.GenerationStage
import dev.rinstel.genie_tts.inference.ModelAssetRepository
import dev.rinstel.genie_tts.inference.QnnRuntimeSupport
import dev.rinstel.genie_tts.inference.RuntimeAssetRepository
import dev.rinstel.genie_tts.inference.XnnpackRuntimeSupport
import dev.rinstel.genie_tts.api.ActiveDefaults
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var topAppBar: MaterialToolbar
    private lateinit var contentScroll: ScrollView
    private lateinit var backendSelector: AutoCompleteTextView
    private lateinit var modelSelector: AutoCompleteTextView
    private lateinit var languageSelector: AutoCompleteTextView
    private lateinit var promptLanguageSelector: AutoCompleteTextView
    private lateinit var synthesisTextInput: TextInputEditText
    private lateinit var referenceSourceValue: TextView
    private lateinit var referenceAudioValue: TextView
    private lateinit var referenceTextInput: TextInputEditText
    private lateinit var referenceHintView: TextView
    private lateinit var chooseReferenceButton: MaterialButton
    private lateinit var restoreReferenceButton: MaterialButton
    private lateinit var decoderStepsValueView: TextView
    private lateinit var decoderStepsSlider: Slider
    private lateinit var playButton: MaterialButton
    private lateinit var runButton: MaterialButton
    private lateinit var statusProgress: LinearProgressIndicator
    private lateinit var statusView: TextView
    private lateinit var modelPathView: TextView
    private lateinit var runtimePathView: TextView
    private lateinit var outputPathView: TextView

    private lateinit var backendOptions: List<BackendOption>
    private var modelOptions: List<CharacterModel> = emptyList()

    private lateinit var modelRepository: ModelAssetRepository
    private lateinit var runtimeRepository: RuntimeAssetRepository
    private lateinit var referenceAudioPicker: ReferenceAudioPicker
    private var backendService: GenieBackendService? = null
    private var serviceBound = false
    private var mediaPlayer: MediaPlayer? = null
    private var latestOutputFilePath: String? = null
    private var lastServiceState: BackendServiceState = BackendServiceState()
    private var defaultReferenceState: ReferenceInputState = ReferenceInputState.empty()
    private var referenceInputState: ReferenceInputState = ReferenceInputState.empty()
    private var suppressReferenceTextChanges = false

    private val pickReferenceAudio = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            handlePickedReferenceAudio(uri)
        }
    }

    private val backendListener = BackendServiceListener { state ->
        runOnUiThread {
            renderServiceState(state)
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as? GenieBackendService.LocalBinder ?: return
            backendService = binder.getService()
            serviceBound = true
            backendService?.addListener(backendListener)
            syncPathViews()
            pushActiveDefaults()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            backendService?.removeListener(backendListener)
            backendService = null
            serviceBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelRepository = ModelAssetRepository(this)
        runtimeRepository = RuntimeAssetRepository(this)
        referenceAudioPicker = ReferenceAudioPicker(this)
        val qnnInspection = QnnRuntimeSupport.inspect(this)
        val xnnpackInspection = XnnpackRuntimeSupport.inspect()
        backendOptions = BackendCatalog.build(
            qnnAvailable = qnnInspection.available,
            qnnStatus = qnnInspection.message,
            // Keep XNNPACK wired for internal debugging and API development only.
            // The main UI stays on CPU/QNN until XNN output parity is verified.
            xnnpackAvailable = false,
            xnnpackStatus = if (xnnpackInspection.available) {
                "XNNPACK output parity is not verified."
            } else {
                xnnpackInspection.message
            },
        )
        modelOptions = modelRepository.discoverCharacterModels()

        bindViews()
        setSupportActionBar(topAppBar)
        applySystemBarInsets()
        configureSelectors()
        configureActions()
        renderInitialState()
        requestNotificationPermissionIfNeeded()
        startAndBindBackendService()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_actions, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return if (item.itemId == R.id.action_settings) {
            startActivity(Intent(this, SettingsActivity::class.java))
            true
        } else {
            super.onOptionsItemSelected(item)
        }
    }

    override fun onDestroy() {
        if (serviceBound) {
            backendService?.removeListener(backendListener)
            unbindService(serviceConnection)
        }
        releasePlayer()
        super.onDestroy()
    }

    private fun bindViews() {
        topAppBar = findViewById(R.id.topAppBar)
        contentScroll = findViewById(R.id.contentScroll)
        backendSelector = findViewById(R.id.backendSelector)
        modelSelector = findViewById(R.id.modelSelector)
        languageSelector = findViewById(R.id.languageSelector)
        promptLanguageSelector = findViewById(R.id.promptLanguageSelector)
        synthesisTextInput = findViewById(R.id.synthesisTextInput)
        referenceSourceValue = findViewById(R.id.referenceSourceValue)
        referenceAudioValue = findViewById(R.id.referenceAudioValue)
        referenceTextInput = findViewById(R.id.referenceTextInput)
        referenceHintView = findViewById(R.id.referenceHintValue)
        chooseReferenceButton = findViewById(R.id.chooseReferenceButton)
        restoreReferenceButton = findViewById(R.id.restoreReferenceButton)
        decoderStepsValueView = findViewById(R.id.decoderStepsValue)
        decoderStepsSlider = findViewById(R.id.decoderStepsSlider)
        playButton = findViewById(R.id.playButton)
        runButton = findViewById(R.id.runButton)
        statusProgress = findViewById(R.id.statusProgress)
        statusView = findViewById(R.id.statusValue)
        modelPathView = findViewById(R.id.modelPathValue)
        runtimePathView = findViewById(R.id.runtimePathValue)
        outputPathView = findViewById(R.id.outputPathValue)
    }

    // Keep the toolbar clear of the status bar and preserve bottom space for gesture nav.
    private fun applySystemBarInsets() {
        val rootLayout = findViewById<android.view.View>(R.id.rootLayout)
        val rootTopPadding = rootLayout.paddingTop
        val contentBottomPadding = contentScroll.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            rootLayout.updatePadding(top = rootTopPadding + systemBars.top)
            contentScroll.updatePadding(bottom = contentBottomPadding + systemBars.bottom)
            insets
        }
        ViewCompat.requestApplyInsets(rootLayout)
    }

    private fun configureSelectors() {
        backendSelector.setAdapter(
            ArrayAdapter(this, android.R.layout.simple_list_item_1, backendOptions.map(BackendOption::label)),
        )
        backendSelector.setText(BackendCatalog.defaultOption(backendOptions).label, false)
        backendSelector.setOnItemClickListener { _, _, _, _ ->
            validateSelectedResources(showReady = false)
            pushActiveDefaults()
        }

        modelSelector.setAdapter(
            ArrayAdapter(this, android.R.layout.simple_list_item_1, modelOptions.map(CharacterModel::displayName)),
        )
        modelSelector.setText(modelOptions.firstOrNull()?.displayName.orEmpty(), false)
        modelSelector.setOnItemClickListener { _, _, _, _ ->
            updateSelectedModelUi()
        }

        languageSelector.setAdapter(
            ArrayAdapter(this, android.R.layout.simple_list_item_1, LANGUAGE_OPTIONS.map(LanguageOption::label)),
        )
        languageSelector.setText(LANGUAGE_OPTIONS.first().label, false)
        languageSelector.setOnItemClickListener { _, _, _, _ ->
            pushActiveDefaults()
        }

        promptLanguageSelector.setAdapter(
            ArrayAdapter(this, android.R.layout.simple_list_item_1, LANGUAGE_OPTIONS.map(LanguageOption::label)),
        )
        promptLanguageSelector.setText(LANGUAGE_OPTIONS.first().label, false)

        decoderStepsSlider.value = DEFAULT_MAX_DECODER_STEPS.toFloat()
        updateDecoderStepsSummary(DEFAULT_MAX_DECODER_STEPS)
        decoderStepsSlider.addOnChangeListener { _, value, _ ->
            updateDecoderStepsSummary(value.toInt())
            pushActiveDefaults()
        }
    }

    private fun configureActions() {
        chooseReferenceButton.setOnClickListener {
            pickReferenceAudio.launch("audio/*")
        }
        restoreReferenceButton.setOnClickListener {
            restoreDefaultReference(manualTrigger = true)
        }
        runButton.setOnClickListener {
            runGeneration()
        }
        playButton.setOnClickListener {
            togglePlayback()
        }
    }

    private fun renderInitialState() {
        syncPathViews()
        lastServiceState = BackendServiceState(
            message = getString(R.string.prototype_service_connecting),
            requestedBackend = selectedBackend().backend,
        )
        renderStatusText()
        if (selectedModelOrNull() == null) {
            updateLocalStatus(getString(R.string.prototype_no_models_found, modelRepository.modelRootPath()))
            renderReferenceState()
            return
        }
        updateSelectedModelUi()
    }

    private fun updateSelectedModelUi() {
        val selectedModel = selectedModelOrNull() ?: return
        syncPathViews(selectedModel)
        loadDefaultReferenceForSelectedModel(manualTrigger = false)
        validateSelectedResources(showReady = false)
        pushActiveDefaults()
    }

    private fun loadDefaultReferenceForSelectedModel(manualTrigger: Boolean) {
        val selectedModel = selectedModelOrNull() ?: return
        val example = modelRepository.findReferenceExample(selectedModel)
        defaultReferenceState = example?.let { ref ->
            ReferenceInputState.defaultSample(
                backendAudioPath = ref.audioFile.absolutePath,
                displayAudioName = ref.audioFile.name,
                referenceText = ref.referenceText,
                hint = ref.audioFile.absolutePath,
            )
        } ?: ReferenceInputState.empty()

        if (referenceInputState.source != ReferenceInputState.Source.MANUAL_OVERRIDE) {
            referenceInputState = defaultReferenceState
        }
        renderReferenceState()

        if (manualTrigger) {
            updateLocalStatus(
                if (defaultReferenceState.isReady) {
                    getString(R.string.prototype_sample_loaded, selectedModel.displayName)
                } else {
                    getString(R.string.prototype_sample_missing, selectedModel.displayName)
                },
            )
        }
    }

    private fun restoreDefaultReference(manualTrigger: Boolean) {
        referenceInputState = defaultReferenceState
        renderReferenceState()
        if (manualTrigger) {
            val selectedModel = selectedModelOrNull()
            updateLocalStatus(
                if (referenceInputState.isReady && selectedModel != null) {
                    getString(R.string.prototype_sample_loaded, selectedModel.displayName)
                } else if (selectedModel != null) {
                    getString(R.string.prototype_sample_missing, selectedModel.displayName)
                } else {
                    getString(R.string.prototype_reference_missing)
                },
            )
        }
    }

    private fun handlePickedReferenceAudio(uri: Uri) {
        val picked = runCatching { referenceAudioPicker.copyToAppCache(uri) }.getOrElse { error ->
            updateLocalStatus(getString(R.string.prototype_reference_pick_failed, error.message ?: error.javaClass.simpleName))
            return
        }
        referenceInputState = ReferenceInputState.manualOverride(
            backendAudioPath = picked.cachedFile.absolutePath,
            displayAudioName = picked.displayName,
            referenceText = referenceTextInput.text?.toString().orEmpty(),
            hint = picked.cachedFile.absolutePath,
        )
        renderReferenceState()
        updateLocalStatus(getString(R.string.prototype_reference_selected, picked.displayName))
    }

    private fun renderReferenceState() {
        referenceSourceValue.text = getString(
            when (referenceInputState.source) {
                ReferenceInputState.Source.DEFAULT_SAMPLE -> R.string.prototype_reference_source_default
                ReferenceInputState.Source.MANUAL_OVERRIDE -> R.string.prototype_reference_source_manual
                ReferenceInputState.Source.NONE -> R.string.prototype_reference_source_missing
            },
        )
        referenceAudioValue.text = referenceInputState.displayAudioName.ifBlank {
            getString(R.string.prototype_reference_audio_empty)
        }
        referenceHintView.text = referenceInputState.hint.ifBlank {
            getString(R.string.prototype_reference_hint_empty)
        }
        suppressReferenceTextChanges = true
        referenceTextInput.setText(referenceInputState.referenceText)
        suppressReferenceTextChanges = false
        restoreReferenceButton.isEnabled = defaultReferenceState.isReady &&
            referenceInputState.source == ReferenceInputState.Source.MANUAL_OVERRIDE
        pushActiveDefaults()
    }

    private fun runGeneration() {
        val selectedModel = selectedModelOrNull()
        if (selectedModel == null) {
            updateLocalStatus(getString(R.string.prototype_no_models_found, modelRepository.modelRootPath()))
            return
        }

        val service = backendService
        if (service == null) {
            updateLocalStatus(getString(R.string.prototype_service_not_ready))
            return
        }

        referenceInputState = referenceInputState.withReferenceText(referenceTextInput.text?.toString().orEmpty())
        renderReferenceState()

        if (synthesisTextInput.text.isNullOrBlank()) {
            updateLocalStatus(getString(R.string.prototype_text_required))
            return
        }
        if (!validateSelectedResources(showReady = false)) {
            return
        }
        if (!referenceInputState.isReady) {
            updateLocalStatus(getString(R.string.prototype_reference_missing))
            return
        }

        val request = GenerationRequest(
            characterModel = selectedModel,
            language = selectedLanguage().value,
            promptLanguage = selectedPromptLanguage().value,
            synthesisText = synthesisTextInput.text?.toString().orEmpty(),
            referenceAudioPath = referenceInputState.backendAudioPath,
            referenceText = referenceInputState.referenceText,
            maxDecoderSteps = decoderStepsSlider.value.toInt(),
        )
        if (service.currentState().busy) {
            updateLocalStatus(getString(R.string.prototype_generation_busy), busy = true)
            return
        }
        stopPlaybackIfActive()
        val started = service.synthesize(selectedBackend().backend, request)
        updateLocalStatus(
            message = if (started) {
                getString(R.string.prototype_generation_started)
            } else {
                getString(R.string.prototype_generation_busy)
            },
            busy = started,
        )
    }

    private fun validateSelectedResources(showReady: Boolean): Boolean {
        val selectedModel = selectedModelOrNull()
        if (selectedModel == null) {
            updateLocalStatus(getString(R.string.prototype_no_models_found, modelRepository.modelRootPath()))
            return false
        }

        val modelInspection = modelRepository.inspect(selectedModel.modelFiles)
        if (!modelInspection.isComplete) {
            updateLocalStatus(
                getString(
                    R.string.prototype_model_missing,
                    modelInspection.missingFiles.joinToString(", "),
                ),
            )
            return false
        }

        val runtimeInspection = runtimeRepository.inspect()
        if (!runtimeInspection.isComplete) {
            updateLocalStatus(
                getString(
                    R.string.prototype_runtime_missing,
                    runtimeRepository.runtimeRoot().absolutePath,
                    runtimeInspection.missingFiles.joinToString(", "),
                ),
            )
            return false
        }

        if (!referenceInputState.isReady) {
            if (showReady) {
                updateLocalStatus(getString(R.string.prototype_reference_missing))
            }
            return false
        }

        if (showReady) {
            updateLocalStatus(getString(R.string.prototype_model_and_runtime_ready, selectedModel.displayName))
        }
        return true
    }

    private fun selectedBackend(): BackendOption {
        val selectedLabel = backendSelector.text?.toString().orEmpty()
        return backendOptions.firstOrNull { it.label == selectedLabel && it.enabled }
            ?: BackendCatalog.defaultOption(backendOptions)
    }

    private fun selectedModelOrNull(): CharacterModel? {
        val selectedName = modelSelector.text?.toString().orEmpty()
        return modelOptions.firstOrNull { it.displayName == selectedName } ?: modelOptions.firstOrNull()
    }

    private fun selectedLanguage(): LanguageOption {
        val label = languageSelector.text?.toString().orEmpty()
        return LANGUAGE_OPTIONS.firstOrNull { it.label == label } ?: LANGUAGE_OPTIONS.first()
    }

    private fun selectedPromptLanguage(): LanguageOption {
        val label = promptLanguageSelector.text?.toString().orEmpty()
        return LANGUAGE_OPTIONS.firstOrNull { it.label == label } ?: LANGUAGE_OPTIONS.first()
    }

    private fun startAndBindBackendService() {
        val intent = GenieBackendService.serviceIntent(this)
        try {
            ContextCompat.startForegroundService(this, intent)
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        } catch (error: RuntimeException) {
            updateLocalStatus(
                getString(
                    R.string.prototype_service_start_error,
                    error.message ?: error.javaClass.simpleName,
                ),
            )
        }
    }

    private fun renderServiceState(state: BackendServiceState) {
        lastServiceState = state.copy(
            message = state.message.ifBlank { getString(R.string.prototype_idle_status) },
            requestedBackend = state.requestedBackend,
        )
        latestOutputFilePath = state.latestOutputFilePath
        renderProgress(lastServiceState)
        syncPathViews(outputPath = state.latestOutputFilePath)
        runButton.isEnabled = !state.busy
        backendSelector.isEnabled = !state.busy
        modelSelector.isEnabled = !state.busy
        chooseReferenceButton.isEnabled = !state.busy
        restoreReferenceButton.isEnabled = !state.busy &&
            defaultReferenceState.isReady &&
            referenceInputState.source == ReferenceInputState.Source.MANUAL_OVERRIDE
        playButton.isEnabled = latestOutputFilePath != null
        updatePlayButton()
        renderStatusText()
    }

    private fun renderStatusText() {
        statusView.text = BackendStatusFormatter.format(
            state = lastServiceState,
            isPlaying = mediaPlayer?.isPlaying == true,
        )
    }

    private fun updateLocalStatus(
        message: String,
        busy: Boolean = false,
        stage: GenerationStage = if (busy) GenerationStage.INSPECTING_RESOURCES else lastServiceState.stage,
    ) {
        lastServiceState = lastServiceState.copy(
            stage = stage,
            message = message,
            busy = busy,
            requestedBackend = selectedBackend().backend,
        )
        renderProgress(lastServiceState)
        renderStatusText()
    }

    // These path views are diagnostic only, so keep all refreshes in one place.
    private fun syncPathViews(
        selectedModel: CharacterModel? = selectedModelOrNull(),
        outputPath: String? = latestOutputFilePath,
    ) {
        modelPathView.text = selectedModel?.let {
            "${modelRepository.modelRootPath()}/${it.relativeModelDirectory}"
        } ?: modelRepository.modelRootPath()
        runtimePathView.text = backendService?.runtimeRootPath() ?: runtimeRepository.runtimeRoot().absolutePath
        outputPathView.text = outputPath ?: (backendService?.outputRootPath()
            ?: runtimeRepository.outputRoot().absolutePath)
    }

    private fun pushActiveDefaults() {
        val model = selectedModelOrNull() ?: return
        if (!referenceInputState.isReady) return
        backendService?.setActiveDefaults(
            ActiveDefaults(
                backend = selectedBackend().backend,
                modelId = model.id,
                language = selectedLanguage().value,
                referenceAudioPath = referenceInputState.backendAudioPath,
                referenceText = referenceInputState.referenceText,
                maxDecoderSteps = decoderStepsSlider.value.toInt(),
            ),
        )
    }

    private fun renderProgress(state: BackendServiceState) {
        val progressModel = BackendStatusFormatter.progressModel(state)
        statusProgress.isIndeterminate = progressModel.indeterminate
        if (!progressModel.indeterminate) {
            statusProgress.setProgressCompat(progressModel.percent, true)
        }
    }

    private fun updateDecoderStepsSummary(value: Int) {
        decoderStepsValueView.text = getString(R.string.prototype_decoder_steps_value, value)
    }

    private fun togglePlayback() {
        if (mediaPlayer?.isPlaying == true) {
            stopPlayback()
            return
        }
        val outputPath = latestOutputFilePath
        if (outputPath.isNullOrBlank()) {
            updateLocalStatus(getString(R.string.prototype_playback_missing))
            return
        }
        val outputFile = File(outputPath)
        if (!outputFile.exists()) {
            updateLocalStatus(getString(R.string.prototype_playback_missing))
            return
        }
        releasePlayer()
        mediaPlayer = MediaPlayer().apply {
            setDataSource(outputFile.absolutePath)
            setOnCompletionListener {
                releasePlayer()
                updateLocalStatus(
                    backendService?.currentState()?.message
                        ?: getString(R.string.prototype_playback_completed),
                )
                updatePlayButton()
            }
            prepare()
            start()
        }
        updatePlayButton()
        updateLocalStatus(getString(R.string.prototype_playback_started))
    }

    private fun stopPlayback() {
        stopPlaybackIfActive()
        updateLocalStatus(
            backendService?.currentState()?.message
                ?: getString(R.string.prototype_playback_stopped),
        )
    }

    private fun stopPlaybackIfActive() {
        if (mediaPlayer?.isPlaying == true) {
            mediaPlayer?.stop()
        }
        releasePlayer()
        updatePlayButton()
    }

    private fun releasePlayer() {
        mediaPlayer?.release()
        mediaPlayer = null
    }

    private fun updatePlayButton() {
        playButton.text = getString(
            if (mediaPlayer?.isPlaying == true) {
                R.string.prototype_stop_button
            } else {
                R.string.prototype_play_button
            },
        )
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return
        }
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
            return
        }
        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_POST_NOTIFICATIONS)
        updateLocalStatus(getString(R.string.prototype_notification_permission_hint))
    }

    companion object {
        private const val DEFAULT_MAX_DECODER_STEPS = 500
        private const val REQUEST_POST_NOTIFICATIONS = 1001

        private val LANGUAGE_OPTIONS = listOf(
            LanguageOption(label = "Chinese (zh)", value = "zh"),
            LanguageOption(label = "English (en)", value = "en"),
            LanguageOption(label = "Japanese (ja)", value = "ja"),
            LanguageOption(label = "Korean (ko)", value = "ko"),
            LanguageOption(label = "Chinese-English (hybrid-zh-en)", value = "hybrid-zh-en"),
        )
    }

    private data class LanguageOption(
        val label: String,
        val value: String,
    )
}
