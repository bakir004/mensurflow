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

// Unary kernels allow out == x and follow the same length and pointer rules as binary kernels.
// out[i] = -x[i]
void neg_f32(const float* x, float* out, std::int64_t n);

// out[i] = |x[i]|; clears the sign bit, including for signed zeros and NaNs.
void abs_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] * x[i], with float32 rounding, overflow, and underflow.
void square_f32(const float* x, float* out, std::int64_t n);

// out[i] = x[i] < 0 ? 0 : x[i]; preserves NaNs and signed zeros.
void relu_f32(const float* x, float* out, std::int64_t n);

} // namespace mensurflow::cpu
