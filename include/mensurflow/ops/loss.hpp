// Loss functions, forward and backward. The last argument says where the op runs (see
// execution.hpp); the result is on the inputs' device. Only float32 values (and int64 labels),
// and only on the CPU so far.
//
// Every loss takes a Reduction: None returns one loss per row (cross_entropy) or per element,
// Sum and Mean return a scalar (0-d) tensor. Mean of an empty input is NaN, as in PyTorch; Sum of
// one is zero. Reduced losses give the same bits at every thread count. Strided and transposed
// inputs are packed first. The forward ops record autograd for their float inputs (labels have no
// gradient). Throw invalid_argument for undefined inputs, wrong dtypes or shapes, mixed devices,
// and labels outside [0, classes).
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow {

enum class Reduction { None, Sum, Mean };

// Multiclass cross-entropy with logits and integer labels: for logits [N, C] and labels [N]
// (int64, each in [0, C)), the loss of row i is logsumexp(logits[i]) - logits[i][labels[i]],
// i.e. -log(softmax(logits[i])[labels[i]]), computed in one stable pass without storing softmax
// (like torch.nn.functional.cross_entropy). None gives an [N] tensor.
Tensor cross_entropy(const Tensor& logits, const Tensor& labels,
                     Reduction reduction = Reduction::Mean, Execution execution = Execution::Auto);

// Backward pass of cross_entropy: (softmax(logits) - one_hot(labels)) * gradient, an [N, C]
// tensor. gradient is the upstream gradient of the loss: [N] for None, a scalar for Sum and Mean
// (Mean also divides by N).
Tensor cross_entropy_backward(const Tensor& logits, const Tensor& labels, const Tensor& gradient,
                              Reduction reduction = Reduction::Mean,
                              Execution execution = Execution::Auto);

// Mean squared error: (x - target)^2 per element, reduced as asked. x and target must have the
// same shape (no broadcasting).
Tensor mse_loss(const Tensor& x, const Tensor& target, Reduction reduction = Reduction::Mean,
                Execution execution = Execution::Auto);

// Backward pass of mse_loss with respect to x: 2 * (x - target) * gradient (divided by the element
// count for Mean). The gradient for target is its negation.
Tensor mse_loss_backward(const Tensor& x, const Tensor& target, const Tensor& gradient,
                         Reduction reduction = Reduction::Mean,
                         Execution execution = Execution::Auto);

// Binary cross-entropy on logits: for each element, -(t * log(sigmoid(x)) + (1 - t) *
// log(1 - sigmoid(x))), computed as max(x, 0) - x * t + log1p(exp(-|x|)) so that large |x| neither
// overflows nor rounds the loss to zero (like torch.nn.functional.binary_cross_entropy_with_logits).
// target holds probabilities in [0, 1] (not checked) and must have x's shape.
Tensor binary_cross_entropy_with_logits(const Tensor& x, const Tensor& target,
                                        Reduction reduction = Reduction::Mean,
                                        Execution execution = Execution::Auto);

// Backward pass of binary_cross_entropy_with_logits with respect to x:
// (sigmoid(x) - target) * gradient (divided by the element count for Mean).
Tensor binary_cross_entropy_with_logits_backward(const Tensor& x, const Tensor& target,
                                                 const Tensor& gradient,
                                                 Reduction reduction = Reduction::Mean,
                                                 Execution execution = Execution::Auto);

} // namespace mensurflow
