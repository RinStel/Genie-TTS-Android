package dev.rinstel.genie_tts

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ComponentCallbacks2
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import dev.rinstel.genie_tts.api.ActiveDefaults
import dev.rinstel.genie_tts.api.InferResult
import dev.rinstel.genie_tts.api.LocalHttpApiServer
import dev.rinstel.genie_tts.inference.BackendRuntime
import dev.rinstel.genie_tts.inference.BackendRuntimeCallbacks
import dev.rinstel.genie_tts.inference.CharacterModel
import dev.rinstel.genie_tts.inference.ExecutionBackend
import dev.rinstel.genie_tts.inference.GenerationRequest
import dev.rinstel.genie_tts.inference.GenerationPipeline
import dev.rinstel.genie_tts.inference.GenerationStage
import dev.rinstel.genie_tts.inference.InferenceTraceLogger
import dev.rinstel.genie_tts.inference.LogcatInferenceTraceLogger
import dev.rinstel.genie_tts.inference.ModelAssetRepository
import dev.rinstel.genie_tts.inference.OrtSessionRuntime
import dev.rinstel.genie_tts.inference.OrtCpuBackend
import dev.rinstel.genie_tts.inference.OrtRuntimeFeatureExtractor
import dev.rinstel.genie_tts.inference.RuntimeAssetRepository
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class GenieBackendService : Service() {
    private val binder = LocalBinder()
    private val apiClients = linkedSetOf<Messenger>()
    private val listeners = linkedSetOf<BackendServiceListener>()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    private val apiMessenger = Messenger(ApiHandler())
    private val generationGate = GenerationRequestGate()

    private data class WarmupKey(
        val backend: ExecutionBackend,
        val modelId: String,
    )

    private val warmupLock = Any()
    private var queuedWarmupKey: WarmupKey? = null
    private var completedWarmupKey: WarmupKey? = null

    private lateinit var settingsRepository: AppSettingsRepository
    private lateinit var modelRepository: ModelAssetRepository
    private lateinit var runtimeRepository: RuntimeAssetRepository
    private lateinit var localHttpApiServer: LocalHttpApiServer

    private val runtimes = linkedMapOf<ExecutionBackend, BackendRuntime>()
    private var apiEnabledBackends: Set<ExecutionBackend> = setOf(ExecutionBackend.CPU)

    private var serverWakeLock: PowerManager.WakeLock? = null

    @Volatile
    private var resourceReloadPending = false

    @Volatile
    private var cacheClearPending = false

    @Volatile
    private var cacheClearScheduled = false

    @Volatile
    private var activeDefaults: ActiveDefaults? = null

    @Volatile
    private var state: BackendServiceState = BackendServiceState(
        message = "",
    )

    override fun onCreate() {
        super.onCreate()
        settingsRepository = AppSettingsRepository(this)
        modelRepository = ModelAssetRepository(this)
        runtimeRepository = RuntimeAssetRepository(this)
        localHttpApiServer = LocalHttpApiServer(
            object : LocalHttpApiServer.Bridge {
                override fun currentState(): BackendServiceState = this@GenieBackendService.currentState()

                override fun availableModels(): List<CharacterModel> = modelRepository.discoverCharacterModels()

                override fun supportedBackends(): Set<ExecutionBackend> = apiEnabledBackends

                override fun supportsAuxiliaryReferences(model: CharacterModel): Boolean =
                    model.modelFiles.maxAuxiliaryReferenceCount > 0 &&
                        File(
                            modelRepository.modelRootDirectory(),
                            "${model.relativeModelDirectory}/prompt_encoder_multi_fp32.onnx",
                        ).isFile

                override fun activeDefaults(): ActiveDefaults? = this@GenieBackendService.activeDefaults

                override fun infer(backend: ExecutionBackend, request: GenerationRequest): Boolean =
                    synthesize(backend, request)

                override fun inferBlocking(backend: ExecutionBackend, request: GenerationRequest, timeoutMs: Long): InferResult =
                    this@GenieBackendService.inferBlocking(backend, request, timeoutMs)
            },
        )
        val traceLogger = LogcatInferenceTraceLogger()
        runtimes[ExecutionBackend.CPU] = OrtSessionRuntime(
            modelRepository = modelRepository,
            runtimeRepository = runtimeRepository,
            backendEngine = OrtCpuBackend(
                configureSessionOptions = OrtCpuBackend::configureCpuSessionOptions,
                configureRoleSessionOptions = { options, role ->
                    OrtCpuBackend.configureCpuSessionOptionsForRole(
                        options = options,
                        role = role,
                        t2sThreadLimit = settingsRepository.t2sCpuThreads,
                        vocoderThreadLimit = settingsRepository.vocoderCpuThreads,
                    )
                },
                traceLogger = traceLogger,
                useNativeDecoder = true,
            ),
            featureExtractorFactory = { backend ->
                OrtRuntimeFeatureExtractor(
                    context = this,
                    runtimeAssets = runtimeRepository,
                    backend = backend,
                    traceLogger = traceLogger,
                )
            },
            traceLogger = traceLogger,
        )
        if (BuildConfig.QNN_ENABLED) {
            runtimes[ExecutionBackend.QNN] = createOptionalQnnRuntime(traceLogger)
        }
        // A full build may still omit proprietary vendor libraries. Advertise
        // only providers that completed their native-library inspection.
        apiEnabledBackends = runtimes
            .filterValues { it.isAvailable }
            .keys
            .toSet()
        createNotificationChannel()
        startForegroundCompat(NOTIFICATION_ID, buildNotification(state))
        syncLocalHttpApiServer()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_RELOAD_RESOURCES -> requestResourceReload()
            ACTION_RELOAD_INFERENCE_SETTINGS -> requestResourceReload()
            ACTION_CLEAR_CACHE -> requestCacheClear()
        }
        syncLocalHttpApiServer()
        NotificationManagerCompat.from(this).notify(NOTIFICATION_ID, buildNotification(state))
        return START_STICKY
    }

    override fun onTimeout(startId: Int) {
        Log.w(TIMING_LOG_TAG, "Foreground service timed out (dataSync 6h limit). Restarting with specialUse type.")
        startForegroundCompat(NOTIFICATION_ID, buildNotification(state))
    }

    private fun startForegroundCompat(id: Int, notification: android.app.Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            val type = ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC or
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING or
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            startForeground(id, notification, type)
        } else {
            startForeground(id, notification)
        }
    }

    override fun onBind(intent: Intent?): IBinder =
        if (intent?.action == BackendApiContract.ACTION_BIND_API) {
            apiMessenger.binder
        } else {
            binder
        }

    override fun onDestroy() {
        localHttpApiServer.stop()
        releaseServerWakeLock()
        worker.shutdownNow()
        runtimes.values.forEach(BackendRuntime::close)
        runtimes.clear()
        super.onDestroy()
    }

    @Suppress("DEPRECATION")
    override fun onTrimMemory(level: Int) {
        // Hiding the UI is normal for this foreground service and should not
        // evict the warm RoBERTa session. Trim only for actual memory pressure.
        if (level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW ||
            level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
            level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND
        ) {
            runCatching {
                worker.execute {
                    runtimes.values.forEach(BackendRuntime::trimMemory)
                    synchronized(warmupLock) {
                        // Trimming closes graph sessions, so a later idle
                        // period must be allowed to schedule warmup again.
                        completedWarmupKey = null
                        queuedWarmupKey = null
                    }
                }
            }
        }
        super.onTrimMemory(level)
    }

    inner class LocalBinder : Binder() {
        fun getService(): GenieBackendService = this@GenieBackendService
    }

    fun addListener(listener: BackendServiceListener) {
        listeners += listener
        listener.onStateChanged(state)
    }

    fun removeListener(listener: BackendServiceListener) {
        listeners -= listener
    }

    fun currentState(): BackendServiceState = state

    fun supportedBackends(): Set<ExecutionBackend> = apiEnabledBackends

    fun runtimeRootPath(): String = runtimeRepository.runtimeRoot().absolutePath

    fun outputRootPath(): String = runtimeRepository.outputRoot().absolutePath

    fun localHttpApiStatus() = LocalHttpApiServer.currentStatus()

    fun setActiveDefaults(defaults: ActiveDefaults?) {
        val supportedDefaults = defaults?.takeIf { it.backend in apiEnabledBackends }
            ?: defaults?.copy(backend = ExecutionBackend.CPU)
        activeDefaults = supportedDefaults
        companionDefaults = supportedDefaults
        supportedDefaults?.let(::scheduleWarmup)
    }

    fun synthesize(
        backend: ExecutionBackend,
        request: GenerationRequest,
    ): Boolean {
        if (backend !in apiEnabledBackends) return false
        if (state.busy || !generationGate.tryAcquire()) {
            return false
        }
        try {
            worker.execute {
                runSynthesis(backend, request)
            }
        } catch (error: Throwable) {
            generationGate.release()
            throw error
        }
        return true
    }

    fun inferBlocking(
        backend: ExecutionBackend,
        request: GenerationRequest,
        timeoutMs: Long,
    ): InferResult {
        if (backend !in apiEnabledBackends) {
            return InferResult.Error("Backend is not enabled.")
        }
        if (state.busy || !generationGate.tryAcquire()) {
            return InferResult.Busy
        }
        val latch = java.util.concurrent.CountDownLatch(1)
        val resultHolder = java.util.concurrent.atomic.AtomicReference<InferResult>(InferResult.Timeout)
        try {
            worker.execute {
                runSynthesis(backend, request) { res ->
                    resultHolder.set(res)
                    latch.countDown()
                }
            }
        } catch (error: Throwable) {
            generationGate.release()
            throw error
        }
        if (!latch.await(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS)) {
            return InferResult.Timeout
        }
        return resultHolder.get()
    }

    private fun runSynthesis(
        backend: ExecutionBackend,
        request: GenerationRequest,
        onComplete: ((InferResult) -> Unit)? = null,
    ) {
        val runtime = runtimes[backend]
        val startedAtMs = SystemClock.elapsedRealtime()
        try {
            val activeRuntime = requireNotNull(runtime) {
                getString(R.string.prototype_backend_unavailable, backend.label)
            }
            require(activeRuntime.isAvailable) {
                getString(R.string.prototype_backend_unavailable, backend.label)
            }
            val result = withActiveGenerationResources {
                activeRuntime.generate(
                    request = request,
                    callbacks = object : BackendRuntimeCallbacks {
                        override fun onStage(
                            stage: GenerationStage,
                            message: String,
                            progressPercent: Int?,
                            progressLabel: String?,
                        ) {
                            publishState(
                                stage = stage,
                                busy = true,
                                requestedBackend = backend,
                                resolvedBackend = activeRuntime.backend,
                                runtimeLabel = activeRuntime.runtimeLabel,
                                initializedModelId = request.characterModel.id,
                                message = stageMessage(stage, request.characterModel, message),
                                progressPercent = progressPercent,
                                progressLabel = progressLabel,
                                generationDurationMs = null,
                            )
                        }
                    },
                )
            }
            val completedAtMs = System.currentTimeMillis()
            val durationMs = SystemClock.elapsedRealtime() - startedAtMs

            publishState(
                stage = GenerationStage.COMPLETED,
                busy = false,
                requestedBackend = backend,
                resolvedBackend = activeRuntime.backend,
                runtimeLabel = activeRuntime.runtimeLabel,
                initializedModelId = request.characterModel.id,
                latestOutputFilePath = result.file.absolutePath,
                progressPercent = 100,
                progressLabel = null,
                generationDurationMs = durationMs,
                lastCompletedAtMs = completedAtMs,
                message = getString(
                    R.string.prototype_service_completed,
                    formatAudioDuration(result.audioSamples),
                ),
            )
            onComplete?.invoke(InferResult.Success(result.file))
        } catch (error: Throwable) {
            publishState(
                stage = GenerationStage.ERROR,
                busy = false,
                requestedBackend = backend,
                resolvedBackend = runtime?.backend ?: backend,
                runtimeLabel = runtime?.runtimeLabel ?: backend.label,
                initializedModelId = request.characterModel.id,
                progressPercent = null,
                progressLabel = null,
                generationDurationMs = SystemClock.elapsedRealtime() - startedAtMs,
                message = error.message ?: error.javaClass.simpleName,
            )
            onComplete?.invoke(InferResult.Error(error.message ?: error.javaClass.simpleName))
        } finally {
            generationGate.release()
            if (resourceReloadPending) {
                reloadInferenceResources()
            }
            if (cacheClearPending) {
                cacheClearPending = false
                performCacheClear()
            }
        }
    }

    /** Load the full-flavor implementation without linking it into CPU APKs. */
    private fun createOptionalQnnRuntime(traceLogger: InferenceTraceLogger): BackendRuntime {
        val runtimeClass = Class.forName("dev.rinstel.genie_tts.inference.QnnBackendRuntime")
        val constructor = runtimeClass.getConstructor(
            android.content.Context::class.java,
            ModelAssetRepository::class.java,
            RuntimeAssetRepository::class.java,
            InferenceTraceLogger::class.java,
        )
        return constructor.newInstance(
            this,
            modelRepository,
            runtimeRepository,
            traceLogger,
        ) as BackendRuntime
    }

    private fun requestResourceReload() {
        if (state.busy || generationGate.isClaimed()) {
            resourceReloadPending = true
        } else {
            // Resource/session teardown must share the inference worker with
            // warmup and generation; doing it on the service main thread can
            // race the queued warmup and close an ORT session underneath it.
            worker.execute(::reloadInferenceResources)
        }
    }

    private fun reloadInferenceResources() {
        if (generationGate.isClaimed()) {
            resourceReloadPending = true
            return
        }
        runtimes.values.forEach(BackendRuntime::close)
        resourceReloadPending = false
        synchronized(warmupLock) {
            completedWarmupKey = null
            queuedWarmupKey = null
        }
        activeDefaults?.let(::scheduleWarmup)
    }

    private fun requestCacheClear() {
        if (cacheClearPending || cacheClearScheduled) return
        if (state.busy || generationGate.isClaimed()) {
            cacheClearPending = true
            return
        }
        cacheClearScheduled = true
        worker.execute {
            try {
                performCacheClear()
            } finally {
                cacheClearScheduled = false
            }
        }
    }

    private fun performCacheClear() {
        if (generationGate.isClaimed()) {
            cacheClearPending = true
            return
        }
        try {
            // Closing the runtime releases ORT sessions and all conditioning
            // caches before deleting files that can be regenerated.
            runtimes.values.forEach(BackendRuntime::close)
            synchronized(warmupLock) {
                completedWarmupKey = null
                queuedWarmupKey = null
            }
            val result = runtimeRepository.clearCaches()
            publishState(
                stage = GenerationStage.IDLE,
                busy = false,
                requestedBackend = state.requestedBackend,
                resolvedBackend = state.resolvedBackend,
                runtimeLabel = state.runtimeLabel,
                initializedModelId = null,
                latestOutputFilePath = null,
                progressPercent = null,
                progressLabel = null,
                generationDurationMs = null,
                lastCompletedAtMs = null,
                message = getString(
                    R.string.prototype_cache_cleared,
                    result.generatedAudioFiles,
                    result.derivedRuntimeFiles,
                ),
            )
        } catch (error: Throwable) {
            publishState(
                stage = GenerationStage.ERROR,
                busy = false,
                requestedBackend = state.requestedBackend,
                resolvedBackend = state.resolvedBackend,
                runtimeLabel = state.runtimeLabel,
                progressPercent = null,
                progressLabel = null,
                generationDurationMs = null,
                message = getString(
                    R.string.prototype_cache_clear_failed,
                    error.message ?: error.javaClass.simpleName,
                ),
            )
        }
    }

    private fun formatAudioDuration(audioSamples: Int): String = String.format(
        Locale.US,
        "%.2fs",
        audioSamples.toDouble() / GenerationPipeline.OUTPUT_SAMPLE_RATE,
    )

    // One state snapshot fans out to notification, bound UI listeners, and local API clients.
    private fun publishState(
        stage: GenerationStage,
        busy: Boolean,
        requestedBackend: ExecutionBackend,
        resolvedBackend: ExecutionBackend,
        runtimeLabel: String,
        message: String,
        initializedModelId: String? = state.initializedModelId,
        latestOutputFilePath: String? = state.latestOutputFilePath,
        progressPercent: Int? = null,
        progressLabel: String? = null,
        generationDurationMs: Long? = state.generationDurationMs,
        lastCompletedAtMs: Long? = state.lastCompletedAtMs,
    ) {
        val snapshot = BackendServiceState(
            stage = stage,
            message = message,
            busy = busy,
            requestedBackend = requestedBackend,
            resolvedBackend = resolvedBackend,
            runtimeLabel = runtimeLabel,
            initializedModelId = initializedModelId,
            latestOutputFilePath = latestOutputFilePath,
            progressPercent = progressPercent,
            progressLabel = progressLabel,
            generationDurationMs = generationDurationMs,
            lastCompletedAtMs = lastCompletedAtMs,
        )
        state = snapshot
        NotificationManagerCompat.from(this).notify(NOTIFICATION_ID, buildNotification(snapshot))
        mainHandler.post {
            listeners.toList().forEach { it.onStateChanged(snapshot) }
            apiClients.toList().forEach { client ->
                runCatching {
                    client.send(
                        Message.obtain(null, BackendApiContract.MSG_STATE_CHANGED).apply {
                            data = snapshot.toBundle()
                        },
                    )
                }.onFailure {
                    apiClients -= client
                }
            }
        }
    }

    // Foreground service mode alone does not keep the CPU awake or boost this worker thread priority.
    private inline fun <T> withActiveGenerationResources(block: () -> T): T {
        val originalPriority = runCatching {
            Process.getThreadPriority(Process.myTid())
        }.getOrDefault(Process.THREAD_PRIORITY_DEFAULT)
        val wakeLock = acquireGenerationWakeLock()
        logPowerState()
        runCatching {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        }
        try {
            return block()
        } finally {
            runCatching {
                Process.setThreadPriority(originalPriority)
            }
            if (wakeLock?.isHeld == true) {
                runCatching { wakeLock.release() }
            }
        }
    }

    private fun acquireGenerationWakeLock(): PowerManager.WakeLock? =
        runCatching {
            getSystemService(PowerManager::class.java)
                ?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:Generation")
                ?.apply {
                    setReferenceCounted(false)
                    acquire(GENERATION_WAKE_LOCK_TIMEOUT_MS)
                }
        }.getOrNull()

    private fun logPowerState() {
        val powerManager = getSystemService(PowerManager::class.java)
        val powerSaveMode = powerManager?.isPowerSaveMode ?: false
        val idleMode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            powerManager?.isDeviceIdleMode ?: false
        } else {
            false
        }
        val ignoringBatteryOptimizations = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            powerManager?.isIgnoringBatteryOptimizations(packageName) ?: false
        } else {
            true
        }
        Log.i(
            TIMING_LOG_TAG,
            "power_state: powerSave=$powerSaveMode deviceIdle=$idleMode ignoringBatteryOptimizations=$ignoringBatteryOptimizations",
        )
    }

    private fun scheduleWarmup(defaults: ActiveDefaults) {
        val key = WarmupKey(defaults.backend, defaults.modelId)
        synchronized(warmupLock) {
            if (completedWarmupKey == key || queuedWarmupKey == key) return
            queuedWarmupKey = key
        }
        runCatching {
            worker.execute {
                try {
                    if (activeDefaults?.let { WarmupKey(it.backend, it.modelId) } != key) return@execute
                    val model = modelRepository.discoverCharacterModels()
                        .firstOrNull { it.id == key.modelId }
                        ?: return@execute
                    val startedAt = SystemClock.elapsedRealtime()
                    withActiveGenerationResources {
                        runtimes[key.backend]?.warmup(defaults.toWarmupRequest(model))
                    }
                    synchronized(warmupLock) {
                        completedWarmupKey = key
                    }
                    Log.i(
                        TIMING_LOG_TAG,
                        "warmup completed backend=${key.backend.label} model=${key.modelId} elapsed_ms=${SystemClock.elapsedRealtime() - startedAt}",
                    )
                } catch (error: Throwable) {
                    Log.i(TIMING_LOG_TAG, "warmup skipped: ${error.message}")
                } finally {
                    synchronized(warmupLock) {
                        if (queuedWarmupKey == key) queuedWarmupKey = null
                    }
                }
            }
        }.onFailure {
            synchronized(warmupLock) {
                if (queuedWarmupKey == key) queuedWarmupKey = null
            }
        }
    }

    private fun ActiveDefaults.toWarmupRequest(model: CharacterModel): GenerationRequest =
        GenerationRequest(
            characterModel = model,
            language = language,
            promptLanguage = promptLanguage,
            synthesisText = if (language.startsWith("zh", ignoreCase = true)) "。" else ".",
            referenceAudioPath = referenceAudioPath,
            referenceText = referenceText,
            maxDecoderSteps = maxDecoderSteps,
            auxiliaryReferenceAudioPaths = auxiliaryReferenceAudioPaths,
        )

    private fun syncLocalHttpApiServer() {
        if (settingsRepository.apiEnabled) {
            acquireServerWakeLock()
            localHttpApiServer.start(settingsRepository.apiPort)
        } else {
            localHttpApiServer.stop()
            releaseServerWakeLock()
        }
    }

    private fun acquireServerWakeLock() {
        if (serverWakeLock?.isHeld == true) return
        serverWakeLock = runCatching {
            getSystemService(PowerManager::class.java)
                ?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:HttpServer")
                ?.apply {
                    setReferenceCounted(false)
                    acquire()
                }
        }.getOrNull()
    }

    private fun releaseServerWakeLock() {
        if (serverWakeLock?.isHeld == true) {
            runCatching { serverWakeLock?.release() }
        }
        serverWakeLock = null
    }

    private fun stageMessage(
        stage: GenerationStage,
        characterModel: CharacterModel,
        messageKey: String,
    ): String = when (stage) {
        GenerationStage.INSPECTING_RESOURCES -> getString(R.string.prototype_stage_inspecting_resources)
        GenerationStage.INSTALLING_MODEL -> getString(R.string.prototype_stage_installing_model)
        GenerationStage.INITIALIZING_BACKEND -> getString(
            R.string.prototype_stage_initializing_backend,
            characterModel.displayName,
        )
        GenerationStage.PREPARING_FEATURES -> {
            if (messageKey == "preparing_runtime") {
                getString(R.string.prototype_stage_preparing_runtime)
            } else {
                getString(R.string.prototype_stage_preparing_features)
            }
        }
        GenerationStage.RUNNING_INFERENCE -> getString(R.string.prototype_stage_running_inference)
        GenerationStage.WRITING_OUTPUT -> getString(R.string.prototype_stage_writing_output)
        GenerationStage.COMPLETED -> getString(R.string.prototype_idle_status)
        GenerationStage.ERROR -> getString(R.string.prototype_generation_error, messageKey)
        GenerationStage.IDLE -> getString(R.string.prototype_idle_status)
    }

    private fun buildNotification(state: BackendServiceState) =
        NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(getString(R.string.prototype_service_notification_title))
            .setContentText(state.message.ifBlank { getString(R.string.prototype_idle_status) })
            .setStyle(
                NotificationCompat.BigTextStyle().bigText(
                    state.message.ifBlank { getString(R.string.prototype_idle_status) },
                ),
            )
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(
                PendingIntent.getActivity(
                    this,
                    0,
                    Intent(this, MainActivity::class.java).apply {
                        flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
                    },
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
                ),
            )
            .build()

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val manager = getSystemService(NotificationManager::class.java)
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            getString(R.string.prototype_service_notification_title),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.prototype_service_notification_description)
        }
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val NOTIFICATION_CHANNEL_ID = "genie_backend_service"
        private const val NOTIFICATION_ID = 2001
        private const val TIMING_LOG_TAG = "GenieTtsTiming"
        private const val GENERATION_WAKE_LOCK_TIMEOUT_MS = 30L * 60L * 1_000L
        private const val KEY_PROGRESS_PERCENT = "progress_percent"
        private const val KEY_PROGRESS_LABEL = "progress_label"
        private const val KEY_GENERATION_DURATION_MS = "generation_duration_ms"
        private const val KEY_LAST_COMPLETED_AT_MS = "last_completed_at_ms"
        private const val ACTION_RELOAD_RESOURCES = "dev.rinstel.genie_tts.action.RELOAD_RESOURCES"
        internal const val ACTION_RELOAD_INFERENCE_SETTINGS =
            "dev.rinstel.genie_tts.action.RELOAD_INFERENCE_SETTINGS"
        private const val ACTION_CLEAR_CACHE = "dev.rinstel.genie_tts.action.CLEAR_CACHE"

        @Volatile
        var companionDefaults: ActiveDefaults? = null
            private set

        fun serviceIntent(context: android.content.Context): Intent =
            Intent(context, GenieBackendService::class.java)

        fun reloadResourcesIntent(context: android.content.Context): Intent =
            serviceIntent(context).setAction(ACTION_RELOAD_RESOURCES)

        fun reloadInferenceSettingsIntent(context: android.content.Context): Intent =
            serviceIntent(context).setAction(ACTION_RELOAD_INFERENCE_SETTINGS)

        fun clearCacheIntent(context: android.content.Context): Intent =
            serviceIntent(context).setAction(ACTION_CLEAR_CACHE)
    }

    private inner class ApiHandler : Handler(Looper.getMainLooper()) {
        override fun handleMessage(message: Message) {
            when (message.what) {
                BackendApiContract.MSG_REGISTER_CLIENT -> {
                    message.replyTo?.let(apiClients::add)
                    sendStateTo(message.replyTo)
                }
                BackendApiContract.MSG_UNREGISTER_CLIENT -> {
                    message.replyTo?.let(apiClients::remove)
                }
                BackendApiContract.MSG_GET_STATUS -> {
                    sendStateTo(message.replyTo)
                }
                BackendApiContract.MSG_SYNTHESIZE -> {
                    val request = message.data.toGenerationRequest() ?: run {
                        sendErrorTo(message.replyTo, getString(R.string.prototype_service_api_invalid_request))
                        return
                    }
                    val backend = message.data.getString(BackendApiContract.KEY_BACKEND)
                        ?.let { value -> ExecutionBackend.entries.firstOrNull { it.name == value } }
                        ?: ExecutionBackend.CPU
                    val started = synthesize(backend, request)
                    if (!started) {
                        sendErrorTo(message.replyTo, getString(R.string.prototype_generation_busy))
                    }
                }
                else -> super.handleMessage(message)
            }
        }
    }

    private fun sendStateTo(target: Messenger?) {
        if (target == null) {
            return
        }
        runCatching {
            target.send(
                Message.obtain(null, BackendApiContract.MSG_STATE_CHANGED).apply {
                    data = state.toBundle()
                },
            )
        }
    }

    private fun sendErrorTo(target: Messenger?, message: String) {
        if (target == null) {
            return
        }
        runCatching {
            target.send(
                Message.obtain(null, BackendApiContract.MSG_STATE_CHANGED).apply {
                    data = state.copy(
                        stage = GenerationStage.ERROR,
                        message = message,
                        busy = false,
                    ).toBundle()
                },
            )
        }
    }

    private fun Bundle.toGenerationRequest(): GenerationRequest? {
        val modelId = getString(BackendApiContract.KEY_MODEL_ID) ?: return null
        val characterModel = modelRepository.discoverCharacterModels()
            .firstOrNull { it.id == modelId }
            ?: return null
        return GenerationRequest(
            characterModel = characterModel,
            language = getString(BackendApiContract.KEY_LANGUAGE).orEmpty(),
            promptLanguage = getString(BackendApiContract.KEY_PROMPT_LANGUAGE)
                ?.takeIf(String::isNotBlank)
                ?: getString(BackendApiContract.KEY_LANGUAGE).orEmpty(),
            synthesisText = getString(BackendApiContract.KEY_SYNTHESIS_TEXT).orEmpty(),
            referenceAudioPath = getString(BackendApiContract.KEY_REFERENCE_AUDIO_PATH).orEmpty(),
            referenceText = getString(BackendApiContract.KEY_REFERENCE_TEXT).orEmpty(),
            maxDecoderSteps = getInt(BackendApiContract.KEY_MAX_DECODER_STEPS, 500),
            auxiliaryReferenceAudioPaths = getStringArrayList(
                BackendApiContract.KEY_AUXILIARY_REFERENCE_AUDIO_PATHS,
            )?.toList().orEmpty(),
        )
    }

    private fun BackendServiceState.toBundle(): Bundle =
        Bundle().apply {
            putString(BackendApiContract.KEY_STAGE, stage.name)
            putString(BackendApiContract.KEY_MESSAGE, message)
            putBoolean(BackendApiContract.KEY_BUSY, busy)
            putString(BackendApiContract.KEY_REQUESTED_BACKEND, requestedBackend.name)
            putString(BackendApiContract.KEY_RESOLVED_BACKEND, resolvedBackend.name)
            putString(BackendApiContract.KEY_RUNTIME_LABEL, runtimeLabel)
            putString(BackendApiContract.KEY_INITIALIZED_MODEL_ID, initializedModelId)
            putString(BackendApiContract.KEY_LATEST_OUTPUT_FILE_PATH, latestOutputFilePath)
            progressPercent?.let { putInt(KEY_PROGRESS_PERCENT, it) }
            putString(KEY_PROGRESS_LABEL, progressLabel)
            generationDurationMs?.let { putLong(KEY_GENERATION_DURATION_MS, it) }
            lastCompletedAtMs?.let { putLong(KEY_LAST_COMPLETED_AT_MS, it) }
        }
}
