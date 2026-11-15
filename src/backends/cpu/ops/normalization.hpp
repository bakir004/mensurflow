// CPU layer_norm, rms_norm and batch_norm, forward and backward, float32.
//
// Layer and RMS norm see the input as `rows` independent lines of `width` contiguous elements.
// Batch norm sees it as [n, c, s]: n samples, c channels, s contiguous spatial elements per
// channel (s == 1 for a [n, c] input).
//
// Every kernel gives the same bits at every thread count: one thread owns a whole line (or a whole
// plane), and the sums that cross lines (weight and bias gradients, per-channel statistics) are
// added in fixed blocks and combined in block order, never per thread.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// y = (x - mean) / sqrt(var + eps) * w + b, with the biased variance of each line.
void layer_norm_f32(const float* x, const float* w, const float* b, float* y, std::int64_t rows,
                    std::int64_t width, float eps);

// dx per line; dw and db sum over all lines. The statistics are recomputed from x.
void layer_norm_backward_f32(const float* x, const float* dy, const float* w, float* dx, float* dw,
                             float* db, std::int64_t rows, std::int64_t width, float eps);

// y = x / sqrt(mean(x^2) + eps) * w.
void rms_norm_f32(const float* x, const float* w, float* y, std::int64_t rows, std::int64_t width,
                  float eps);

void rms_norm_backward_f32(const float* x, const float* dy, const float* w, float* dx, float* dw,
                           std::int64_t rows, std::int64_t width, float eps);

// Training: normalizes with the batch's mean and biased variance and moves the running statistics
// toward them by `momentum` (the running variance gets the unbiased estimate, as PyTorch does).
// Evaluation: normalizes with the running statistics and leaves them alone.
void batch_norm_f32(const float* x, const float* w, const float* b, float* running_mean,
                    float* running_var, float* y, std::int64_t n, std::int64_t c, std::int64_t s,
                    bool training, float momentum, float eps);

// Training backward recomputes the batch statistics from x; evaluation backward uses the running
// statistics (a constant scale per channel).
void batch_norm_backward_f32(const float* x, const float* dy, const float* w,
                             const float* running_mean, const float* running_var, float* dx,
                             float* dw, float* db, std::int64_t n, std::int64_t c, std::int64_t s,
                             bool training, float eps);

} // namespace mensurflow::cpu
