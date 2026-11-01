#pragma once

#include "mensurflow/tensor.hpp"

namespace mensurflow {

Tensor mse_loss(const Tensor& prediction, const Tensor& target);
Tensor log_softmax(const Tensor& x, int axis = -1);
Tensor cross_entropy(const Tensor& logits, const Tensor& targets, int axis = -1);

} // namespace mensurflow
