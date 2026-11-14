// Convolution and pooling over images. The last argument says where the op runs (see
// execution.hpp); the result is on the inputs' device. Only float32, and only CPU kernels so far.
//
// Layout and conventions for conv2d (pooling's are further down):
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

// ---- Pooling -------------------------------------------------------------------------------
//
// Conventions shared by max_pool2d and avg_pool2d:
// * NCHW input x[N, C, H, W] with H, W >= 1; each channel is pooled on its own; output
//   [N, C, H_out, W_out]. Inputs may be views.
// * kernel_size {kH, kW}, each at least 1. stride {stride_h, stride_w}, each at least 1, or
//   {0, 0} (the default) for stride = kernel_size. Zero padding {pad_h, pad_w} with
//   0 <= pad <= kernel / 2 in each dimension (PyTorch's rule), so every window holds at least one
//   input cell.
// * H_out = floor((H + 2 * pad_h - kH) / stride_h) + 1 and W_out likewise (ceil_mode = false). An
//   output size below 1 is rejected. Dilation is always 1.

struct MaxPool2dResult {
    Tensor output;  // F32 [N, C, H_out, W_out]
    Tensor indices; // I64 [N, C, H_out, W_out]
};

// Max pooling. indices holds, for every output element, the position ih * W + iw of the chosen
// cell within its own [H, W] plane (as PyTorch). Padded cells are never chosen. Ties go to the
// first cell of the window in row-major order. NaN follows argmax and max: a NaN beats every
// number, the first NaN of the window is chosen and the output is NaN. Records autograd for x
// with a private copy of the indices (changing the returned indices does not change the
// gradient); the backward pass is max_pool2d_backward.
MaxPool2dResult max_pool2d(const Tensor& x, Size2d kernel_size, Size2d stride = {0, 0},
                           Size2d padding = {0, 0}, Execution execution = Execution::Auto);

// The gradient of max_pool2d with respect to an input of shape input_shape [N, C, H, W]: every
// element of grad_output is added to the input cell its index names (a cell chosen by several
// overlapping windows gets the sum); all other cells are 0. indices must be I64 with
// grad_output's shape and values in [0, H * W).
Tensor max_pool2d_backward(const Tensor& grad_output, const Tensor& indices,
                           const Shape& input_shape, Execution execution = Execution::Auto);

} // namespace mensurflow
