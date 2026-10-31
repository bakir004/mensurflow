// Plain sequential versions of the CPU kernels: one loop each, no OpenMP and no parallel_for.
// They are the baseline the parallel CPU kernels are compared with,
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

// out[i] = -x[i]. Same length, null-pointer, and in-place rules as the binary kernels above.
void neg_f32(const float* x, float* out, std::int64_t n);

// sum(x[i] * y[i]) accumulated in double and rounded to float once at the end, exactly what the
// parallel dot_f32 does with one thread. n <= 0 gives 0.
float dot_f32(const float* x, const float* y, std::int64_t n);

// Row-major C[m,n] = A[m,k] * B[k,n] in the plain i-k-j loop order: for each row of A, every
// entry A[i,p] scales row p of B and is added into row i of C. C is overwritten, not accumulated
// into, and must not overlap A or B. k == 0 fills C with zeros; m == 0 or n == 0 does nothing.
// A negative dimension, or a null pointer where memory would be touched, throws invalid_argument.
void matmul_f32(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                std::int64_t k);

} // namespace mensurflow::cpu::sequential
