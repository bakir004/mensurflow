// CPU BLAS kernels over contiguous float32 buffers.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// sum(x[i] * y[i]) for i in [0, n). Products are accumulated in double, so the result is the
// float rounding of a near-exact sum, whatever n is. Parallel above the cutoff;
// the result then depends slightly on the thread count, never on scheduling. n <= 0 gives 0.
float dot_f32(const float* x, const float* y, std::int64_t n);

} // namespace mensurflow::cpu
