// CPU reduction kernels over contiguous float32 buffers. Each is one loop marked with
// MENSURFLOW_PARALLEL_FOR, which splits it across threads in an OpenMP build; max and min alone
// have a separate sequential and parallel version.
//
// Callers (src/ops) have already validated shapes, so kernels do no checking of their own: n is
// non-negative, mean/max/min/argmax never see an empty input, and pointers are valid wherever a
// size is non-zero. Sums and products are accumulated in double and rounded to float at the end.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// Whole-tensor reductions of x[n] to one value. A NaN anywhere gives NaN.

// sum(x[i]). n == 0 gives 0.
float sum_f32(const float* x, std::int64_t n);

// sum(x[i]) / n. n > 0.
float mean_f32(const float* x, std::int64_t n);

// The largest x[i]. n > 0. When the largest values are zeros, the result is +0.
float max_f32_sequential(const float* x, std::int64_t n);
float max_f32_parallel(const float* x, std::int64_t n);

// The smallest x[i]. n > 0.
float min_f32_sequential(const float* x, std::int64_t n);
float min_f32_parallel(const float* x, std::int64_t n);

// The product of all x[i]. n == 0 gives 1.
float prod_f32(const float* x, std::int64_t n);

// Reductions along one axis of a row-major tensor x with `rank` dimensions given by `shape`.
// out has the input shape without that axis. An empty axis gives 0 for sum and 1 for prod; mean,
// max and min never see one.
void sum_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis);

void mean_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                   std::int64_t axis);

void max_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis);

void min_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis);

void prod_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                   std::int64_t axis);

// For each of `rows` rows of x[rows, width], the index of the largest value. Ties pick the first
// index; a NaN picks the first NaN. width > 0.
void argmax_f32(const float* x, std::int64_t* out, std::int64_t rows, std::int64_t width);

} // namespace mensurflow::cpu
