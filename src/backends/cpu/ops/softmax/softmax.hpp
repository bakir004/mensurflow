// CPU softmax kernels over contiguous float32 buffers. The input is seen as [outer, length, inner]:
// each of the outer * inner lines of `length` elements (stride `inner`) is normalized on its own,
// so one line is always handled by one thread and the result does not depend on the thread count.
// The lines are split across threads with MENSURFLOW_PARALLEL_FOR (threads.hpp).
//
// Stable: every line subtracts its maximum before exp, and sums in double. A line whose elements
// are all -inf (fully masked) gives zeros from softmax and -inf from log_softmax instead of NaN.
// A NaN anywhere in a line makes the whole line NaN.
//
// Callers (src/ops) have already validated the shapes, so kernels do no checking of their own.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// out = exp(x - max) / sum(exp(x - max)) along each line.
void softmax_f32(const float* x, float* out, std::int64_t outer, std::int64_t length,
                 std::int64_t inner);

// out = x - max - log(sum(exp(x - max))) along each line.
void log_softmax_f32(const float* x, float* out, std::int64_t outer, std::int64_t length,
                     std::int64_t inner);

// Gradient of softmax from its output y: out = y * (gradient - sum(gradient * y)) along each line.
void softmax_backward_f32(const float* y, const float* gradient, float* out, std::int64_t outer,
                          std::int64_t length, std::int64_t inner);

// Gradient of log_softmax from its output y: out = gradient - exp(y) * sum(gradient) per line.
void log_softmax_backward_f32(const float* y, const float* gradient, float* out, std::int64_t outer,
                              std::int64_t length, std::int64_t inner);

} // namespace mensurflow::cpu
