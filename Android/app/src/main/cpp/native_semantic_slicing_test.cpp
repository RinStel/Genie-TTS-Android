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
    expectEqual(genie_tts::sliceSemanticTokens(outputs, 1, 0), {});
}

void testNormalEos() {
    const std::vector<int64_t> outputs = {99, 11, 12, 1024};
    expectEqual(genie_tts::sliceSemanticTokens(outputs, 3, 0), {11, 12});
}

void testMaxSteps() {
    const std::vector<int64_t> outputs = {99, 21, 22, 23};
    expectEqual(genie_tts::sliceSemanticTokens(outputs, 3, 0), {21, 22});
}

void testTerminalReplacement() {
    const std::vector<int64_t> outputs = {99, 31, 32, 0};
    expectEqual(genie_tts::sliceSemanticTokens(outputs, 3, 0), {31, 32});
}

void testCompletedCountIsClampedToAvailableOutputs() {
    const std::vector<int64_t> outputs = {99, 41, 42, 1024};
    expectEqual(genie_tts::sliceSemanticTokens(outputs, 99, 0), {99, 41, 42});
    expectEqual(genie_tts::sliceSemanticTokens(outputs, -1, 0), {});
}

void testCustomTerminalReplacementIsExcluded() {
    const std::vector<int64_t> outputs = {99, 51, 52, -7};
    expectEqual(genie_tts::sliceSemanticTokens(outputs, 3, -7), {51, 52});
}

}  // namespace

int main() {
    testOneStepEos();
    testNormalEos();
    testMaxSteps();
    testTerminalReplacement();
    testCompletedCountIsClampedToAvailableOutputs();
    testCustomTerminalReplacementIsExcluded();
    return 0;
}
