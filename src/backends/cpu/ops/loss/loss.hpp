// CPU loss kernels over contiguous float32 buffers: cross-entropy with integer labels, mean
// squared error and binary cross-entropy with logits, forward and backward.
//
// `reduction` is 0 (none: one loss per row or element), 1 (sum) or 2 (mean); with sum or mean the
// output is a single float. Reduced losses add fixed-size chunks in parallel and then the chunk
// sums in order, so the result is the same at every thread count. Sums are kept in double. The
// mean of no elements is NaN, their sum zero.
//
// Backward kernels take the upstream gradient as given for that reduction: one value per row or
// element for none, a single value for sum and mean (mean also divides by the count).
//
// Callers (src/ops) have already validated shapes and labels (0 <= label < classes).
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// Per row of [rows, classes] logits: logsumexp(row) - row[label]. Fused: no softmax is stored.
void cross_entropy_f32(const float* logits, const std::int64_t* labels, float* out,
                       std::int64_t rows, std::int64_t classes, int reduction);

// d loss / d logits = (softmax(row) - one_hot(label)) * upstream, row by row.
void cross_entropy_backward_f32(const float* logits, const std::int64_t* labels,
                                const float* gradient, float* out, std::int64_t rows,
                                std::int64_t classes, int reduction);

// (x - target)^2 per element.
void mse_loss_f32(const float* x, const float* target, float* out, std::int64_t n, int reduction);

// d loss / d x = 2 * (x - target) * upstream.
void mse_loss_backward_f32(const float* x, const float* target, const float* gradient, float* out,
                           std::int64_t n, int reduction);

// max(x, 0) - x * target + log1p(exp(-|x|)) per element: binary cross-entropy on sigmoid(x)
// without computing sigmoid, so large |x| neither overflows nor loses the loss to rounding.
void bce_with_logits_f32(const float* x, const float* target, float* out, std::int64_t n,
                         int reduction);

// d loss / d x = (sigmoid(x) - target) * upstream.
void bce_with_logits_backward_f32(const float* x, const float* target, const float* gradient,
                                  float* out, std::int64_t n, int reduction);

} // namespace mensurflow::cpu
