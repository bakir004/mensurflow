// Integer labels and trainable lookup tables. The result stays on the inputs' device; CPU
// kernels are available so far, following the same execution policy as the other public ops.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <cstdint>

namespace mensurflow {

// Appends a dimension of `num_classes` to an I64 index tensor. The output is I64, with one at
// each selected class and zero elsewhere. Indices may be scalar, empty or strided; classes must
// be positive and every index must be in [0, num_classes). This integer op is not differentiable.
Tensor one_hot(const Tensor& indices, std::int64_t num_classes,
               Execution execution = Execution::Auto);

// Looks up rows of a float32 [num_embeddings, embedding_dim] table with I64 indices of any rank.
// The output shape is indices.shape() + [embedding_dim]. Views are accepted; negative and
// out-of-range indices are errors. Autograd accumulates gradients only into the weight table.
Tensor embedding(const Tensor& weight, const Tensor& indices,
                 Execution execution = Execution::Auto);

// Backward of embedding: gradient.shape() must be indices.shape() + [embedding_dim]. Returns
// float32 [num_embeddings, embedding_dim], summing gradients for repeated indices in input order.
// All indices must be in [0, num_embeddings); unused rows receive zero. Not differentiable itself.
Tensor embedding_backward(const Tensor& gradient, const Tensor& indices,
                          std::int64_t num_embeddings, Execution execution = Execution::Auto);

} // namespace mensurflow
