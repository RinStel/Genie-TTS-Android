#include "native_semantic_slicing.h"

#include <algorithm>

std::vector<int64_t> SliceSemanticTokens(
    const std::vector<int64_t>& tokens,
    int64_t completed_count,
    int64_t terminal_token) {
    const int64_t count = std::clamp<int64_t>(
        completed_count, 0, static_cast<int64_t>(tokens.size()));
    std::vector<int64_t> result(tokens.begin(), tokens.begin() + count);
    if (!result.empty() && result.back() == terminal_token) {
        result.pop_back();
    }
    return result;
}

namespace genie_tts {

std::vector<int64_t> sliceSemanticTokens(
    const std::vector<int64_t>& tokens,
    int64_t completed_count,
    int64_t terminal_token) {
    if (tokens.empty()) return {};
    const int64_t count = std::clamp<int64_t>(
        completed_count, 0, static_cast<int64_t>(tokens.size()));
    if (count <= 1) return {};
    std::vector<int64_t> semantic(tokens.begin() + 1, tokens.begin() + count);
    if (!semantic.empty() && semantic.back() == terminal_token) {
        semantic.pop_back();
    }
    return semantic;
}

}  // namespace genie_tts
