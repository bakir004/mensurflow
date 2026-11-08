// CPU elementwise kernels over contiguous float32 buffers. Each is one loop marked with
// MENSURFLOW_PARALLEL_FOR: with OpenMP, inputs larger than the operation's cutoff are split across
// threads; without OpenMP, or for small inputs, the loop runs on the calling thread. Every element
// is computed the same way either way, so results do not depend on the thread count.
//
// Callers (src/ops) have already validated shapes, so kernels do no checking of their own: n is
// non-negative and pointers are valid whenever n > 0. out may be the same buffer as an input
// (in place), but must not partially overlap one.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// F32 comparisons write canonical Bool bytes (0 or 1). Output must not overlap either input.
void eq_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n);
void ne_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n);
void lt_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n);
void le_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n);
void gt_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n);
void ge_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n);

// Nonzero condition selects x, zero selects y. out may equal x or y exactly, but must not
// overlap condition or partially overlap a value input. Selected bits are copied unchanged.
void where_f32(const std::uint8_t* condition, const float* x, const float* y, float* out,
               std::int64_t n);

// Binary operations: out[i] = f(a[i], b[i]).

// out[i] = a[i] + b[i]
void add_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] - b[i]
void sub_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] * b[i]
void mul_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] / b[i] (IEEE 754: x / 0 is inf or NaN)
void div_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] raised to b[i]
void pow_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = min(a[i], b[i]); a NaN in either input propagates
void minimum_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = max(a[i], b[i]); a NaN in either input propagates
void maximum_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = alpha * x[i] + y[i] with one rounding (std::fma).
void axpy_f32(float alpha, const float* x, const float* y, float* out, std::int64_t n);

// Adds bias[features] to every row of x[n / features, features]. out may be x but must not
// overlap bias. features may be 0 only when n is 0.
void bias_add_f32(const float* x, const float* bias, float* out, std::int64_t n,
                  std::int64_t features);

// Unary operations: out[i] = f(x[i]).

// out[i] = -x[i]; flips only the sign bit
void neg_f32(const float* x, float* out, std::int64_t n);

// out[i] = |x[i]|; clears only the sign bit, including for zeros and NaNs
void abs_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] * x[i]
void square_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] < 0 ? 0 : x[i]; preserves NaNs and signed zeros
void relu_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] < 0 ? negative_slope * x[i] : x[i]; preserves NaNs and signed zeros
void leaky_relu_f32(const float* x, float* out, std::int64_t n, float negative_slope);

// GELU (Gaussian CDF or tanh approximation) and SiLU use their limits at infinity.
void gelu_f32(const float* x, float* out, std::int64_t n, bool approximate = false);
void silu_f32(const float* x, float* out, std::int64_t n);

// Negative values become alpha * expm1(x).
void elu_f32(const float* x, float* out, std::int64_t n, float alpha);

// Stable softplus; caller validates positive finite beta and finite threshold.
void softplus_f32(const float* x, float* out, std::int64_t n, float beta, float threshold);

// out[i] = sqrt(x[i]); negatives give NaN
void sqrt_f32(const float* x, float* out, std::int64_t n);

// out[i] = exp(x[i])
void exp_f32(const float* x, float* out, std::int64_t n);

// out[i] = 1 / (1 + exp(-x[i])), computed without overflow for either sign
void sigmoid_f32(const float* x, float* out, std::int64_t n);

// out[i] = natural logarithm
void log_f32(const float* x, float* out, std::int64_t n);

// out[i] = 1 / sqrt(x[i])
void rsqrt_f32(const float* x, float* out, std::int64_t n);

// out[i] = tanh(x[i])
void tanh_f32(const float* x, float* out, std::int64_t n);

// out[i] = sin(x[i])
void sin_f32(const float* x, float* out, std::int64_t n);

// out[i] = cos(x[i])
void cos_f32(const float* x, float* out, std::int64_t n);

// out[i] = 1 / x[i]
void reciprocal_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] limited to [lower, upper]; NaNs pass through.
void clamp_f32(const float* x, float* out, std::int64_t n, float lower, float upper);

} // namespace mensurflow::cpu
