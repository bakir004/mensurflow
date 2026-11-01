// Scalar versions of the sequential CPU kernels: the same code as sequential.cpp, compiled with
// auto-vectorization switched off. They are the slowest baseline, and exist so a benchmark can
// separate what SIMD gives from what threads give:
//
//   scalar      one element at a time, no SIMD, no threads
//   sequential  the same loop, with whatever SIMD the compiler finds
//   parallel    the same loop split across threads
//
// There is no scalar.cpp of its own to maintain: scalar.cpp includes sequential.cpp with a
// different namespace and per-file compiler flags (see CMakeLists.txt). Results are identical to
// the sequential ones; tests/ops/scalar_test.cpp checks that.
//
// When a kernel is added to sequential.hpp, copy its declaration here too if a benchmark needs a
// scalar version of it. The list below is the benchmarked set, not a contract.
#pragma once

#include <cstdint>

namespace mensurflow::cpu::scalar {

void add_f32(const float* a, const float* b, float* out, std::int64_t n);
void sub_f32(const float* a, const float* b, float* out, std::int64_t n);
void mul_f32(const float* a, const float* b, float* out, std::int64_t n);
void div_f32(const float* a, const float* b, float* out, std::int64_t n);
void axpy_f32(float alpha, const float* x, const float* y, float* out, std::int64_t n);
void bias_add_f32(const float* x, const float* bias, float* out, std::int64_t n,
                  std::int64_t features);
void neg_f32(const float* x, float* out, std::int64_t n);
void abs_f32(const float* x, float* out, std::int64_t n);
void square_f32(const float* x, float* out, std::int64_t n);
void relu_f32(const float* x, float* out, std::int64_t n);
void sqrt_f32(const float* x, float* out, std::int64_t n);
void exp_f32(const float* x, float* out, std::int64_t n);
void sigmoid_f32(const float* x, float* out, std::int64_t n);
float dot_f32(const float* x, const float* y, std::int64_t n);
void matmul_f32(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                std::int64_t k);

} // namespace mensurflow::cpu::scalar
