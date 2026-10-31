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

// out[i] = -x[i]. out may be the same buffer as x; the same length and pointer rules apply.
void neg_f32(const float* x, float* out, std::int64_t n);

} // namespace mensurflow::cpu
