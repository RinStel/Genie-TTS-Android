#include "reference_resampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(__ANDROID__)
#include <jni.h>
#endif

namespace genie_tts {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kHalfWindow = 24;

double sinc(double value) {
    if (std::abs(value) < 1e-12) return 1.0;
    return std::sin(kPi * value) / (kPi * value);
}

}  // namespace

std::vector<float> ResampleReferenceAudio(
    const std::vector<float>& input,
    int32_t source_rate,
    int32_t target_rate) {
    if (source_rate <= 0 || target_rate <= 0) {
        throw std::invalid_argument("Sample rates must be positive.");
    }
    if (input.empty() || source_rate == target_rate) return input;

    const double source_to_target = static_cast<double>(source_rate) / target_rate;
    // Keep the transition band below Nyquist before applying the finite window.
    const double cutoff = 0.95 * std::min(1.0, 1.0 / source_to_target);
    const auto output_size = static_cast<size_t>(std::max<long long>(
        1LL,
        std::llround(static_cast<double>(input.size()) * target_rate / source_rate)));
    std::vector<float> output(output_size);

    for (size_t output_index = 0; output_index < output_size; ++output_index) {
        const double source_position = output_index * source_to_target;
        const int64_t center = static_cast<int64_t>(std::floor(source_position));
        double value = 0.0;

        for (int64_t tap = center - kHalfWindow + 1;
             tap <= center + kHalfWindow;
             ++tap) {
            const double distance = static_cast<double>(tap) - source_position;
            const double window_position = std::abs(distance) / kHalfWindow;
            if (window_position >= 1.0) continue;

            const double coefficient = cutoff * sinc(cutoff * distance) *
                (0.5 + 0.5 * std::cos(kPi * window_position));
            if (tap >= 0 && tap < static_cast<int64_t>(input.size())) {
                value += coefficient * input[static_cast<size_t>(tap)];
            }
        }

        // Keep zero-padding at the signal edges to match soxr's boundary behavior.
        output[output_index] = static_cast<float>(value);
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
