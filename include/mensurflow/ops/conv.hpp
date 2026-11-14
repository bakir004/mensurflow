// Convolution and pooling over images. The last argument says where the op runs (see
// execution.hpp); the result is on the inputs' device. Only float32, and only CPU kernels so far.
//
// Layout and conventions for conv2d:
// * NCHW: input x[N, C_in, H, W], weight[C_out, C_in, kH, kW], optional bias[C_out], output
//   y[N, C_out, H_out, W_out]. Inputs may be views; they are packed before the kernel runs.
// * Separate stride {stride_h, stride_w} >= 1 and zero padding {pad_h, pad_w} >= 0 per dimension.
//   Padding is added on both sides and reads as zero.
// * H_out = floor((H + 2 * pad_h - kH) / stride_h) + 1 and W_out likewise. A kernel larger than the
//   padded input (an output size below 1) is rejected, as is a kernel dimension of 0.
// * Only dilation {1, 1} and groups 1 are supported; other values are rejected with
//   invalid_argument rather than silently ignored.
// * y[n, co, oh, ow] = bias[co] + sum over ci, i, j of
//   weight[co, ci, i, j] * x[n, ci, oh * stride_h - pad_h + i, ow * stride_w - pad_w + j].
//   This is cross-correlation, as in PyTorch and every other framework (the kernel is not flipped).
// * Computed as im2col + the packed GEMM, one image at a time. Products accumulate in float32, so
//   results can differ from a direct loop in the last bits.
// * Empty batch or C_out == 0 gives an empty output; C_in == 0 gives the bias (or zeros).
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <array>
#include <cstdint>

namespace mensurflow {

// A pair of sizes for the two spatial dimensions, in the order {height, width}.
using Size2d = std::array<std::int64_t, 2>;

// 2-D convolution (see the conventions above). bias may be nullptr. Records autograd when x,
// weight or bias requires gradients; the backward pass uses the three functions below, and the
// weight and bias gradients sum over the batch.
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr,
              Size2d stride = {1, 1}, Size2d padding = {0, 0}, Size2d dilation = {1, 1},
              std::int64_t groups = 1, Execution execution = Execution::Auto);

// The gradient of conv2d with respect to its input x of shape input_shape [N, C_in, H, W], given
// the gradient of its output grad_output [N, C_out, H_out, W_out] and the weight. Computed as
// weight^T * grad_output per image, then col2im (overlapping windows add up).
Tensor conv2d_backward_data(const Tensor& grad_output, const Tensor& weight,
                            const Shape& input_shape, Size2d stride = {1, 1},
                            Size2d padding = {0, 0}, Execution execution = Execution::Auto);

// The gradient of conv2d with respect to a weight of shape weight_shape [C_out, C_in, kH, kW],
// given grad_output and the forward input x. Summed over the batch; an empty batch gives zeros.
Tensor conv2d_backward_weight(const Tensor& grad_output, const Tensor& x,
                              const Shape& weight_shape, Size2d stride = {1, 1},
                              Size2d padding = {0, 0}, Execution execution = Execution::Auto);

// The gradient of conv2d with respect to its bias: grad_output [N, C_out, H_out, W_out] summed over
// N, H_out and W_out (in double), giving [C_out].
Tensor conv2d_backward_bias(const Tensor& grad_output, Execution execution = Execution::Auto);

} // namespace mensurflow
