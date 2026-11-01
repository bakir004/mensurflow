// CPU elementwise kernels over contiguous float32 buffers. out may be the same buffer as a or b
// (in-place), but partial overlap is not supported. n == 0 does nothing; a negative n, or a null
// pointer with n > 0, throws invalid_argument. Division follows IEEE 754 (x/0 is inf or NaN).
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// out[i] = a[i] + b[i]
void add_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] - b[i]
void sub_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] * b[i]
void mul_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = a[i] / b[i]
void div_f32(const float* a, const float* b, float* out, std::int64_t n);

// out[i] = fma(alpha, x[i], y[i]), with a single float32 rounding. out may alias x or y.
void axpy_f32(float alpha, const float* x, const float* y, float* out, std::int64_t n);

// Add compact bias[features] to each row of x[n / features, features]. out may alias x;
// bias must not overlap out. n and features must be nonnegative. n == 0 does nothing,
// including features == 0; otherwise features must be positive and divide n exactly.
void bias_add_f32(const float* x, const float* bias, float* out, std::int64_t n,
                  std::int64_t features);

// Unary kernels allow out == x and follow the same length and pointer rules as binary kernels.
// out[i] = -x[i]
void neg_f32(const float* x, float* out, std::int64_t n);

// out[i] = |x[i]|; clears the sign bit, including for signed zeros and NaNs.
void abs_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] * x[i], with float32 rounding, overflow, and underflow.
void square_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] < 0 ? 0 : x[i]; preserves NaNs and signed zeros.
void relu_f32(const float* x, float* out, std::int64_t n);

// sqrt(x[i]); negative values produce NaN, and signed zeros are preserved.
void sqrt_f32(const float* x, float* out, std::int64_t n);

// exp(x[i]); overflow produces +inf, underflow can produce +0.
void exp_f32(const float* x, float* out, std::int64_t n);

// Numerically stable logistic function; -inf -> 0, +inf -> 1, NaN -> NaN.
void sigmoid_f32(const float* x, float* out, std::int64_t n);
void log_f32(const float* x, float* out, std::int64_t n);
void rsqrt_f32(const float* x, float* out, std::int64_t n);
void tanh_f32(const float* x, float* out, std::int64_t n);
void minimum_f32(const float* a, const float* b, float* out, std::int64_t n);
void maximum_f32(const float* a, const float* b, float* out, std::int64_t n);
void clamp_f32(const float* x, float* out, std::int64_t n, float lower, float upper);

} // namespace mensurflow::cpu
