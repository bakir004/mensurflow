// Plain sequential versions of the CPU kernels: one loop each, no OpenMP and no parallel_for.
// They are the baseline the parallel kernels in elementwise.hpp and blas.hpp are compared with,
// and they follow the same contract (argument checks, in-place use, double accumulation in dot),
// so a result from either implementation can be compared with the other.
#pragma once

#include <cstdint>

namespace mensurflow::cpu::sequential {

// out[i] = a[i] + b[i], and likewise for sub, mul, and div (IEEE 754: x / 0 is inf or NaN).
// out may be the same buffer as a or b. n == 0 does nothing; a negative n, or a null pointer with
// n > 0, throws invalid_argument.
void add_f32(const float* a, const float* b, float* out, std::int64_t n);
void sub_f32(const float* a, const float* b, float* out, std::int64_t n);
void mul_f32(const float* a, const float* b, float* out, std::int64_t n);
void div_f32(const float* a, const float* b, float* out, std::int64_t n);

// sum(x[i] * y[i]) accumulated in double and rounded to float once at the end, exactly what the
// parallel dot_f32 does with one thread. n <= 0 gives 0.
float dot_f32(const float* x, const float* y, std::int64_t n);

} // namespace mensurflow::cpu::sequential
