// Softmax and log-softmax along one axis, forward and backward. The last argument says where the
// op runs (see execution.hpp); the result is on the input's device. Only float32, and only on the
// CPU so far.
//
// Both are numerically stable: each line subtracts its maximum before exp and sums in double, so
// logits of 1000 or -1000 give the same result as 0. A line whose elements are all -inf (every
// position masked) gives zeros from softmax and -inf from log_softmax instead of NaN; a NaN
// anywhere in a line makes the whole line NaN. Strided and transposed inputs are packed first.
// softmax and log_softmax record autograd. Throw invalid_argument for an undefined or non-F32
// input, an axis out of range, or (backward) mismatched shapes.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow {

// Normalized exponentials along `axis` (default the last): exp(x - max) / sum(exp(x - max)), so
// every line is positive and sums to one. The result has x's shape.
Tensor softmax(const Tensor& x, int axis = -1, Execution execution = Execution::Auto);

// log(softmax(x)) along `axis`, computed as x - logsumexp(x) without forming softmax, so very
// negative log-probabilities stay exact instead of rounding to log(0) = -inf.
Tensor log_softmax(const Tensor& x, int axis = -1, Execution execution = Execution::Auto);

// Backward pass of softmax from its output y = softmax(x): y * (gradient - sum(gradient * y))
// along `axis`. output and gradient must have the same shape.
Tensor softmax_backward(const Tensor& output, const Tensor& gradient, int axis = -1,
                        Execution execution = Execution::Auto);

// Backward pass of log_softmax from its output y = log_softmax(x):
// gradient - exp(y) * sum(gradient) along `axis`.
Tensor log_softmax_backward(const Tensor& output, const Tensor& gradient, int axis = -1,
                            Execution execution = Execution::Auto);

} // namespace mensurflow
