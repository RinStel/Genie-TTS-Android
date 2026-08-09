#pragma once

#include <cstdint>
#include <vector>

namespace genie_tts {

// Bounded-memory band-limited resampling for reference conditioning audio.
std::vector<float> ResampleReferenceAudio(
    const std::vector<float>& input,
    int32_t source_rate,
    int32_t target_rate);

}  // namespace genie_tts
