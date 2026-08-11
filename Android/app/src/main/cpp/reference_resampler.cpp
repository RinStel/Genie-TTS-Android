#include "reference_resampler.h"
#include "soxr.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(__ANDROID__)
#include <jni.h>
#endif

namespace genie_tts {

std::vector<float> ResampleReferenceAudio(
    const std::vector<float>& input,
    int32_t source_rate,
    int32_t target_rate) {
    if (source_rate <= 0 || target_rate <= 0) {
        throw std::invalid_argument("Sample rates must be positive.");
    }
    if (input.empty() || source_rate == target_rate) return input;

    const auto output_size = static_cast<size_t>(std::max<long long>(
        1LL,
        std::llround(
            static_cast<double>(input.size()) * target_rate / source_rate)));
    std::vector<float> output(output_size);

    const soxr_quality_spec_t quality = soxr_quality_spec(SOXR_HQ, 0);
    size_t input_done = 0;
    size_t output_done = 0;
    const soxr_error_t error = soxr_oneshot(
        static_cast<double>(source_rate),
        static_cast<double>(target_rate),
        1,
        input.data(),
        input.size(),
        &input_done,
        output.data(),
        output.size(),
        &output_done,
        nullptr,
        &quality,
        nullptr);
    if (error) {
        throw std::runtime_error(
            std::string("Reference audio resampling failed: ") + soxr_strerror(error));
    }
    if (input_done != input.size() || output_done != output.size()) {
        throw std::runtime_error("Reference audio resampler returned an incomplete block.");
    }
    return output;
}

}  // namespace genie_tts

#if defined(__ANDROID__)
extern "C" JNIEXPORT jfloatArray JNICALL
Java_dev_rinstel_genie_1tts_inference_ReferenceAudioResampler_nativeResample(
    JNIEnv* env,
    jclass,
    jfloatArray input_array,
    jint source_rate,
    jint target_rate) {
    if (!input_array) return env->NewFloatArray(0);
    const jsize input_size = env->GetArrayLength(input_array);
    jfloat* input_data = env->GetFloatArrayElements(input_array, nullptr);
    if (!input_data && input_size != 0) return env->NewFloatArray(0);

    std::vector<float> input;
    if (input_size > 0) input.assign(input_data, input_data + input_size);
    if (input_data) env->ReleaseFloatArrayElements(input_array, input_data, JNI_ABORT);

    try {
        const std::vector<float> output = genie_tts::ResampleReferenceAudio(
            input, source_rate, target_rate);
        jfloatArray output_array = env->NewFloatArray(static_cast<jsize>(output.size()));
        if (output_array && !output.empty()) {
            env->SetFloatArrayRegion(
                output_array, 0, static_cast<jsize>(output.size()), output.data());
        }
        return output_array;
    } catch (const std::invalid_argument&) {
        jclass exception_class = env->FindClass("java/lang/IllegalArgumentException");
        env->ThrowNew(exception_class, "Sample rates must be positive.");
        return nullptr;
    }
}
#endif
