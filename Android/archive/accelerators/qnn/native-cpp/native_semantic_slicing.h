#pragma once

#include <cstdint>
#include <vector>

std::vector<int64_t> SliceSemanticTokens(
    const std::vector<int64_t>& tokens,
    int64_t completed_count,
    int64_t terminal_token);
