#include "reference_resampler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

void testOutputDurationAndConstantSignal() {
    const std::vector<float> input(1001, 0.25f);
    const std::vector<float> output = genie_tts::ResampleReferenceAudio(input, 32'000, 16'000);
    assert(output.size() == 501);
    assert(std::abs(output.front() - 0.18480448f) < 2e-4f);
    assert(std::abs(output[250] - 0.25f) < 1e-5f);
    assert(std::abs(output.back() - 0.18480448f) < 2e-4f);
}

void testImpulseRemainsBounded() {
    std::vector<float> input(257, 0.0f);
    input[0] = 1.0f;
    const std::vector<float> output = genie_tts::ResampleReferenceAudio(input, 32'000, 16'000);
    assert(output.size() == 129);
    for (const float sample : output) assert(std::isfinite(sample));
}

void assertAnchor(float actual, float expected, float tolerance) {
    assert(std::abs(actual - expected) <= tolerance);
}

void testSoxrHqFixtureAnchors() {
    // Anchors are generated from tests/parity/fixtures/reference_resampler_soxr_hq.json.
    // They cover startup, steady-state, and tail behavior without embedding large fixtures.
    std::vector<float> sweep(513);
    for (size_t index = 0; index < sweep.size(); ++index) {
        sweep[index] = static_cast<float>(std::sin(
            40.0 * kPi * static_cast<double>(index) / (sweep.size() - 1)));
    }
    const std::vector<float> sweep_output = genie_tts::ResampleReferenceAudio(
        sweep, 32'000, 16'000);
    assert(sweep_output.size() == 257);
    assertAnchor(sweep_output[0], 0.042268168f, 0.001f);
    assertAnchor(sweep_output[1], 0.462894917f, 0.001f);
    assertAnchor(sweep_output[128], 0.0f, 0.0001f);
    assertAnchor(sweep_output[256], -0.042268217f, 0.001f);

    std::vector<float> speech_like(401);
    for (size_t index = 0; index < speech_like.size(); ++index) {
        const double position = static_cast<double>(index) / (speech_like.size() - 1);
        speech_like[index] = static_cast<float>(
            0.55 * std::sin(13.0 * kPi * position) +
            0.2 * std::sin(71.0 * kPi * position));
    }
    const std::vector<float> speech_output = genie_tts::ResampleReferenceAudio(
        speech_like, 32'000, 16'000);
    assert(speech_output.size() == 201);
    assertAnchor(speech_output[0], 0.029738743f, 0.001f);
    assertAnchor(speech_output[1], 0.284872353f, 0.001f);
    assertAnchor(speech_output[100], 0.350000024f, 0.001f);
    assertAnchor(speech_output[200], 0.029738756f, 0.001f);

    std::vector<float> edge_length(1001);
    for (size_t index = 0; index < edge_length.size(); ++index) {
        edge_length[index] = static_cast<float>(index) / edge_length.size();
    }
    const std::vector<float> edge_output = genie_tts::ResampleReferenceAudio(
        edge_length, 32'000, 16'000);
    assert(edge_output.size() == 501);
    assertAnchor(edge_output[0], 0.000170194f, 0.0001f);
    assertAnchor(edge_output[250], 0.499500513f, 0.0001f);
    assertAnchor(edge_output[500], 0.738309324f, 0.002f);

    std::vector<float> cancellation(257);
    std::fill(cancellation.begin(), cancellation.end(), 0.0f);
    const std::vector<float> cancellation_output = genie_tts::ResampleReferenceAudio(
        cancellation, 32'000, 16'000);
    assert(std::all_of(cancellation_output.begin(), cancellation_output.end(), [](float value) {
        return value == 0.0f;
    }));
}

}  // namespace

int main() {
    testOutputDurationAndConstantSignal();
    testImpulseRemainsBounded();
    testSoxrHqFixtureAnchors();
    return 0;
}
