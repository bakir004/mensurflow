#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <vector>

namespace mensurflow {

struct NormalizationBackward {
    Tensor input;
    Tensor weight;
    Tensor bias;
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
                                        const Shape& normalized_shape, const Tensor* weight = nullptr,
                                        float epsilon = 1e-5f,
                                        Execution execution = Execution::Auto);

Tensor batch_norm(const Tensor& x, Tensor& running_mean, Tensor& running_variance,
                  const Tensor* weight = nullptr, const Tensor* bias = nullptr,
                  bool training = true, float momentum = 0.1f, float epsilon = 1e-5f,
                  Execution execution = Execution::Auto);
NormalizationBackward batch_norm_backward(const Tensor& x, const Tensor& gradient,
                                          const Tensor& running_mean,
                                          const Tensor& running_variance,
                                          const Tensor* weight = nullptr, bool training = true,
                                          float momentum = 0.1f, float epsilon = 1e-5f,
                                          Execution execution = Execution::Auto);

} // namespace mensurflow
