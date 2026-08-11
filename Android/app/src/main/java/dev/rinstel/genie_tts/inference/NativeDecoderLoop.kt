package dev.rinstel.genie_tts.inference

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.lang.reflect.Field

/** Optional JNI decoder shared by CPU and full flavors. */
object NativeDecoderLoop {
    private var loaded = false
    private var environmentHandleField: Field? = null
    private var sessionHandleField: Field? = null

    fun interface DecoderProgressCallback {
        fun onProgress(generatedSteps: Int, maxSteps: Int)
    }

    init {
        try {
            System.loadLibrary("native_decoder_loop")
            loaded = true
        } catch (error: UnsatisfiedLinkError) {
            loaded = false
        }
    }

    fun isAvailable(): Boolean = loaded

    /** JNI is useful only when QNN can keep the KV cache on device memory. */
    fun hasQnnDeviceMemory(environment: OrtEnvironment): Boolean {
        if (!loaded) return false
        return runCatching {
            nativeHasQnnDeviceMemory(getEnvironmentNativeHandle(environment))
        }.getOrDefault(false)
    }

    private fun getSessionNativeHandle(session: OrtSession): Long {
        val field = sessionHandleField ?: run {
            val declared = OrtSession::class.java.getDeclaredField("nativeHandle")
            declared.isAccessible = true
            sessionHandleField = declared
            declared
        }
        return field.getLong(session)
    }

    private fun getEnvironmentNativeHandle(environment: OrtEnvironment): Long {
        val field = environmentHandleField ?: run {
            val declared = OrtEnvironment::class.java.getDeclaredField("nativeHandle")
            declared.isAccessible = true
            environmentHandleField = declared
            declared
        }
        return field.getLong(environment)
    }

    @JvmStatic
    external fun nativeHasQnnDeviceMemory(environmentPtr: Long): Boolean

    @JvmStatic
    external fun nativeDecoderLoop(
        environmentPtr: Long,
        sessionPtr: Long,
        inputNames: Array<String>,
        outputNames: Array<String>,
        yData: LongArray,
        yShape: LongArray,
        yEmbData: FloatArray,
        yEmbShape: LongArray,
        kvData: FloatArray,
        kvShape: LongArray,
        numKvTensors: Int,
        outputMapping: IntArray,
        useQnnIoBinding: Boolean,
        maxSteps: Int,
        progressCallback: DecoderProgressCallback?,
    ): LongArray?

    fun runDecoderLoop(
        environment: OrtEnvironment,
        session: OrtSession,
        inputNames: List<String>,
        outputNames: List<String>,
        initialY: LongArray,
        yShape: LongArray,
        initialYEmb: FloatArray,
        yEmbShape: LongArray,
        initialKv: FloatArray,
        kvShape: LongArray,
        numKvTensors: Int,
        outputMapping: IntArray,
        useQnnIoBinding: Boolean,
        maxSteps: Int,
        progressCallback: DecoderProgressCallback?,
    ): LongArray {
        val environmentPtr = getEnvironmentNativeHandle(environment)
        val sessionPtr = getSessionNativeHandle(session)
        return requireNotNull(
            nativeDecoderLoop(
                environmentPtr,
                sessionPtr,
                inputNames.toTypedArray(),
                outputNames.toTypedArray(),
                initialY,
                yShape,
                initialYEmb,
                yEmbShape,
                initialKv,
                kvShape,
                numKvTensors,
                outputMapping,
                useQnnIoBinding,
                maxSteps,
                progressCallback,
            ),
        ) {
            "Native decoder loop failed; no semantic tokens were produced."
        }
    }
}
