// CPU 2-D pooling kernels, float32, NCHW layout. These are the sequential versions: they run on
// the calling thread and are the baseline the parallel versions are tested and benchmarked against.
//
// Callers (src/ops) have validated the shapes, so kernels do no checking of their own. All
// buffers are contiguous row-major. Padding is at most half the kernel in each dimension, so every
// window holds at least one input cell.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// The sizes of one pooling: input x[n, c, h, w], output y[n, c, h_out, w_out] with
// h_out = (h + 2 * pad_h - kh) / stride_h + 1 (rounded down) and w_out likewise. Window (oh, ow)
// covers input rows oh * stride_h - pad_h .. + kh - 1 and the matching columns; cells outside the
// input are padding.
struct Pool2dShape {
    std::int64_t n = 0, c = 0, h = 0, w = 0;
    std::int64_t kh = 0, kw = 0;
    std::int64_t stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0;
    std::int64_t h_out = 0, w_out = 0;
};

// y = the largest input cell of each window; indices = its position ih * w + iw within its
// [h, w] plane. Padding never wins. Ties go to the first cell in row-major order; a NaN wins over
// every number and the first NaN wins (as in argmax and max).
void max_pool2d_f32_sequential(const float* x, float* y, std::int64_t* indices,
                               const Pool2dShape& s);

// grad_x[n, c, h, w]: every grad_y element added to the cell its index names (cells chosen by
// several windows add up). Overwrites grad_x.
void max_pool2d_backward_f32_sequential(const float* grad_y, const std::int64_t* indices,
                                        float* grad_x, const Pool2dShape& s);

// y = the sum of the input cells of each window (in double) divided by kh * kw: padded cells
// count in the divisor as zeros (PyTorch's count_include_pad = true). Without ceil_mode no window
// reaches past the padding, so the divisor is always kh * kw.
void avg_pool2d_f32_sequential(const float* x, float* y, const Pool2dShape& s);

// grad_x[n, c, h, w]: every grad_y element divided by kh * kw and added to each input cell of its
// window (overlapping windows add up; padding cells take no gradient). Overwrites grad_x.
void avg_pool2d_backward_f32_sequential(const float* grad_y, float* grad_x, const Pool2dShape& s);

} // namespace mensurflow::cpu
