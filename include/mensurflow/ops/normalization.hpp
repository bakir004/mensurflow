// Layer, RMS and batch normalization, forward and backward, on the CPU in float32. The last
// argument says where the op runs (see execution.hpp); the result is on the input's device.
//
// layer_norm / rms_norm normalize over the last `normalized_shape.size()` dimensions of x
// (normalized_shape must equal those dimensions); weight and bias, when given, have exactly that
// shape and are optional (null means scale 1 and bias 0).
//
//   layer_norm  y = (x - mean) / sqrt(var + epsilon) * weight + bias
//   rms_norm    y = x / sqrt(mean(x^2) + epsilon) * weight       (no centering, no bias)
//
// with the biased (divide by N) variance, per line of the normalized dimensions.
//
// batch_norm normalizes each channel of x = [N, C, ...] (channels in dimension 1, any number of
// trailing spatial dimensions, NCHW for images, [N, C] for features) over everything but the
// channel; weight and bias have shape [C] and are optional.
//   training = true   uses the batch's mean and biased variance, and moves the running statistics
//                     toward them in place: running = (1 - momentum) * running + momentum * batch,
//                     where the running variance receives the unbiased estimate (divide by N - 1),
//                     exactly as PyTorch does. Needs at least two values per channel.
//   training = false  uses running_mean and running_variance (read only).
// running_mean and running_variance are contiguous F32 tensors of shape [C].
//
// The *_backward functions return the gradient of the input (and of the weight and bias; rms_norm
// has no bias, so its `bias` member is left undefined). They recompute the statistics from x
// instead of keeping them from the forward pass. batch_norm_backward in training mode ignores the
// running statistics and in evaluation mode treats them as constants.
//
// The three forward functions record autograd for x, weight and bias when any of them requires a
// gradient (CPU float32). Every kernel gives the same bits at every thread count: sums across
// lines or samples are added in blocks whose layout depends only on the shape.
//
// Inputs must be F32 CPU tensors (views and strided inputs are packed first); epsilon must be
// finite and positive. Mismatched shapes, wrong dtypes or devices and invalid arguments throw
// std::invalid_argument before anything is computed or updated.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow {

struct NormalizationBackward {
    Tensor input;  // gradient of x
    Tensor weight; // gradient of the scale, shape normalized_shape or [C]
    Tensor bias;   // gradient of the bias; undefined for rms_norm
};

Tensor layer_norm(const Tensor& x, const Shape& normalized_shape, const Tensor* weight = nullptr,
                  const Tensor* bias = nullptr, float epsilon = 1e-5f,
                  Execution execution = Execution::Auto);
NormalizationBackward layer_norm_backward(const Tensor& x, const Tensor& gradient,
                                          const Shape& normalized_shape,
                                          const Tensor* weight = nullptr, float epsilon = 1e-5f,
                                          Execution execution = Execution::Auto);

Tensor rms_norm(const Tensor& x, const Shape& normalized_shape, const Tensor* weight = nullptr,
                float epsilon = 1e-5f, Execution execution = Execution::Auto);
NormalizationBackward rms_norm_backward(const Tensor& x, const Tensor& gradient,
                                        const Shape& normalized_shape,
                                        const Tensor* weight = nullptr, float epsilon = 1e-5f,
                                        Execution execution = Execution::Auto);

Tensor batch_norm(const Tensor& x, Tensor& running_mean, Tensor& running_variance,
                  const Tensor* weight = nullptr, const Tensor* bias = nullptr,
                  bool training = true, float momentum = 0.1f, float epsilon = 1e-5f,
                  Execution execution = Execution::Auto);
NormalizationBackward batch_norm_backward(const Tensor& x, const Tensor& gradient,
                                          const Tensor& running_mean,
                                          const Tensor& running_variance,
                                          const Tensor* weight = nullptr, bool training = true,
                                          float epsilon = 1e-5f,
                                          Execution execution = Execution::Auto);

} // namespace mensurflow
