#pragma once

#include <cstdint>
#include <vector>

std::vector<int64_t> SliceSemanticTokens(
    const std::vector<int64_t>& tokens,
    int64_t completed_count,
    int64_t terminal_token);

// Compatibility entry point for the existing native unit test.
namespace genie_tts {
std::vector<int64_t> sliceSemanticTokens(
    const std::vector<int64_t>& tokens,
    int64_t completed_count,
    int64_t terminal_token);
}
