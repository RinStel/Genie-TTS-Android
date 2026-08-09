#include "native_semantic_slicing.h"

#include <algorithm>

std::vector<int64_t> SliceSemanticTokens(
    const std::vector<int64_t>& tokens,
    int64_t completed_count,
    int64_t terminal_token) {
    const int64_t count = std::clamp<int64_t>(
        completed_count, 0, static_cast<int64_t>(tokens.size()));
    if (count == 0) return {};

    const auto first = tokens.end() - count;
    std::vector<int64_t> result(first, tokens.end());
    const auto first_special = std::find_if(
        result.begin(), result.end(), [](int64_t token) { return token >= 1024; });
    result.erase(first_special, result.end());
    if (!result.empty() && result.back() == terminal_token) {
        result.pop_back();
    }
    return result;
}
