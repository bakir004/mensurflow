#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <cstdint>

namespace mensurflow {

// In-place CPU optimizer updates. State tensors must be F32, contiguous, and parameter-shaped.
void sgd(Tensor& parameter, const Tensor& gradient, float learning_rate, float momentum = 0.0f,
         float weight_decay = 0.0f, Tensor* momentum_buffer = nullptr,
         Execution execution = Execution::Auto);

void adam(Tensor& parameter, const Tensor& gradient, Tensor& first_moment, Tensor& second_moment,
          float learning_rate, std::int64_t step, float beta1 = 0.9f, float beta2 = 0.999f,
          float epsilon = 1e-8f, float weight_decay = 0.0f,
          Execution execution = Execution::Auto);

void adamw(Tensor& parameter, const Tensor& gradient, Tensor& first_moment, Tensor& second_moment,
           float learning_rate, std::int64_t step, float beta1 = 0.9f, float beta2 = 0.999f,
           float epsilon = 1e-8f, float weight_decay = 0.01f,
           Execution execution = Execution::Auto);

} // namespace mensurflow
