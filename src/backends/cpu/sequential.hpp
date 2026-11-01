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

// Fused float32 alpha*x + y, with one rounding; out may alias x or y.
void axpy_f32(float alpha, const float* x, const float* y, float* out, std::int64_t n);

// Add bias[features] to every row of x[n / features, features]. out may alias x but must
// not overlap bias. Nonnegative n/features are required; n == 0 permits features == 0.
// For n > 0, features must be positive and divide n exactly, and all pointers must be nonnull.
void bias_add_f32(const float* x, const float* bias, float* out, std::int64_t n,
                  std::int64_t features);

// Unary operations follow the same length, null-pointer, and in-place rules as above.
void neg_f32(const float* x, float* out, std::int64_t n);
void abs_f32(const float* x, float* out, std::int64_t n);
void square_f32(const float* x, float* out, std::int64_t n);
void relu_f32(const float* x, float* out, std::int64_t n);
void sqrt_f32(const float* x, float* out, std::int64_t n);
void exp_f32(const float* x, float* out, std::int64_t n);
void sigmoid_f32(const float* x, float* out, std::int64_t n);

// sum(x[i] * y[i]) accumulated in double and rounded to float once at the end, exactly what the
// parallel dot_f32 does with one thread. n <= 0 gives 0.
float dot_f32(const float* x, const float* y, std::int64_t n);

// Row-major y[m] = A[m,k] * x[k], accumulated in double. k == 0 fills y with zeros.
void matvec_f32(const float* a, const float* x, float* y, std::int64_t m, std::int64_t k);

// Row-major C[m,n] = A[m,k] * B[k,n] in the plain i-k-j loop order: for each row of A, every
// entry A[i,p] scales row p of B and is added into row i of C. C is overwritten, not accumulated
// into, and must not overlap A or B. k == 0 fills C with zeros; m == 0 or n == 0 does nothing.
// A negative dimension, or a null pointer where memory would be touched, throws invalid_argument.
void matmul_f32(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                std::int64_t k);

} // namespace mensurflow::cpu::sequential
