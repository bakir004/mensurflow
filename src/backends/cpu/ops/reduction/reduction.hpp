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

// For each of `rows` rows of x[rows, width], the index of the smallest value, with argmax_f32's
// rules: ties pick the first index; a NaN picks the first NaN. width > 0. Sequential: the
// baseline for a later parallel version.
void argmin_f32_sequential(const float* x, std::int64_t* out, std::int64_t rows,
                           std::int64_t width);

// Variance along one axis (shape, rank and axis as in sum_axis_f32), in one pass with Welford's
// update (mean and M2 kept in double): out = M2 / (count - correction), where count is the axis
// size. count - correction <= 0 (including an empty axis) gives NaN; a NaN or infinity in the
// reduced elements gives NaN. Sequential: the baseline for a later parallel version.
void var_axis_f32_sequential(const float* x, float* out, const std::int64_t* shape,
                             std::int64_t rank, std::int64_t axis, std::int64_t correction);

// grad_x = grad * 2 * (x - mean) / (count - correction), with grad the [outer, inner] gradient of
// var_axis_f32_sequential's output (keepdim does not change its layout) and the mean recomputed
// with the same Welford pass. NaN where count - correction <= 0. Overwrites grad_x (x's shape).
void var_backward_f32_sequential(const float* x, const float* grad, float* grad_x,
                                 const std::int64_t* shape, std::int64_t rank, std::int64_t axis,
                                 std::int64_t correction);

// Standard deviation along one axis: sqrt of var_axis_f32_sequential's result (taken in double),
// with the same correction and NaN rules.
void std_axis_f32_sequential(const float* x, float* out, const std::int64_t* shape,
                             std::int64_t rank, std::int64_t axis, std::int64_t correction);

// grad_x = grad * (x - mean) / ((count - correction) * std), with mean and std recomputed by the
// same Welford pass. Where std is 0 (a constant row) the gradient is 0, as in PyTorch; NaN where
// count - correction <= 0. Overwrites grad_x (x's shape).
void std_backward_f32_sequential(const float* x, const float* grad, float* grad_x,
                                 const std::int64_t* shape, std::int64_t rank, std::int64_t axis,
                                 std::int64_t correction);

} // namespace mensurflow::cpu
