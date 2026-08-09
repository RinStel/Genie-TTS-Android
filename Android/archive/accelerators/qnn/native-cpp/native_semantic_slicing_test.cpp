#include "native_semantic_slicing.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

void expectEqual(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected) {
    assert(actual == expected);
}

void testOneStepEos() {
    const std::vector<int64_t> outputs = {99, 1024};
    expectEqual(SliceSemanticTokens(outputs, 1, 0), {});
}

void testMaxStepsKeepsEveryGeneratedSemanticToken() {
    const std::vector<int64_t> outputs = {99, 11, 12, 13};
    expectEqual(SliceSemanticTokens(outputs, 3, 0), {11, 12, 13});
}

void testNormalEos() {
    const std::vector<int64_t> outputs = {99, 21, 22, 1024};
    expectEqual(SliceSemanticTokens(outputs, 3, 0), {21, 22});
}

void testTerminalReplacement() {
    const std::vector<int64_t> outputs = {99, 31, 32, 0};
    expectEqual(SliceSemanticTokens(outputs, 3, 0), {31, 32});
}

void testCompletedCountIsClampedToAvailableOutputs() {
    const std::vector<int64_t> outputs = {99, 41, 42, 1024};
    expectEqual(SliceSemanticTokens(outputs, 99, 0), {99, 41, 42});
    expectEqual(SliceSemanticTokens(outputs, -1, 0), {});
}

void testCustomTerminalReplacementIsExcluded() {
    const std::vector<int64_t> outputs = {99, 51, 52, -7};
    expectEqual(SliceSemanticTokens(outputs, 3, -7), {51, 52});
}

void testSpecialTokenBeforeBoundaryIsTruncated() {
    const std::vector<int64_t> outputs = {99, 61, 1024, 0};
    expectEqual(SliceSemanticTokens(outputs, 3, 0), {61});
}

}  // namespace

int main() {
    testOneStepEos();
    testMaxStepsKeepsEveryGeneratedSemanticToken();
    testNormalEos();
    testTerminalReplacement();
    testCompletedCountIsClampedToAvailableOutputs();
    testCustomTerminalReplacementIsExcluded();
    testSpecialTokenBeforeBoundaryIsTruncated();
    return 0;
}
