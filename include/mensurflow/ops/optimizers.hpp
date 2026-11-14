// Optimizer updates: SGD (with momentum and weight decay), Adam and AdamW. Each one updates the
// parameter and its state in place, in one fused pass over the elements (no temporaries), on the
// CPU, in float32. Every tensor argument must be a contiguous F32 CPU tensor of the parameter's
// shape (the gradient may be any view: it is packed first); the parameter and its state tensors
// must be distinct tensors that do not share storage. Nothing is recorded for autograd: these are
// the step that follows backward().
//
// Hyperparameters follow PyTorch's conventions, so the same numbers give the same trajectory:
//
//   sgd    g' = g + weight_decay * p;  v = momentum * v + g';  p -= learning_rate * v
//          (without momentum: p -= learning_rate * g'). The momentum buffer starts as zeros, which
//          makes the first step a plain gradient step.
//   adam   g' = g + weight_decay * p (L2 regularization folded into the gradient);
//          m = beta1 * m + (1 - beta1) * g';  v = beta2 * v + (1 - beta2) * g'^2;
//          p -= learning_rate / (1 - beta1^step) * m / (sqrt(v) / sqrt(1 - beta2^step) + epsilon)
//   adamw  the same, but the weight decay is decoupled from the moments: p *= 1 - lr * weight_decay
//          first, and the gradient is used as it is.
//
// `step` counts from 1 (the number of this update, for the bias correction); the caller owns it.
// The result is the same on every thread count. Invalid hyperparameters and mismatched tensors
// throw std::invalid_argument before anything is changed.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <cstdint>

namespace mensurflow {

// momentum_buffer is required when momentum > 0 (zeros on the first step) and ignored otherwise
// (it may be null; if given it is still validated).
void sgd(Tensor& parameter, const Tensor& gradient, float learning_rate, float momentum = 0.0f,
         float weight_decay = 0.0f, Tensor* momentum_buffer = nullptr,
         Execution execution = Execution::Auto);

// first_moment and second_moment start as zeros.
void adam(Tensor& parameter, const Tensor& gradient, Tensor& first_moment, Tensor& second_moment,
          float learning_rate, std::int64_t step, float beta1 = 0.9f, float beta2 = 0.999f,
          float epsilon = 1e-8f, float weight_decay = 0.0f, Execution execution = Execution::Auto);

void adamw(Tensor& parameter, const Tensor& gradient, Tensor& first_moment, Tensor& second_moment,
           float learning_rate, std::int64_t step, float beta1 = 0.9f, float beta2 = 0.999f,
           float epsilon = 1e-8f, float weight_decay = 0.01f,
           Execution execution = Execution::Auto);

} // namespace mensurflow
