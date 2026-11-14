// Reductions. The last argument says where the op runs (see execution.hpp); the result is on the
// inputs' device. Only float32, and only CPU kernels so far.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow {

// Returns the sum of all elements as a scalar tensor. An empty tensor sums to zero.
// NaN inputs propagate to the result.
Tensor sum(const Tensor& x, Execution execution = Execution::Auto);

// Returns the arithmetic mean of all elements as a scalar tensor. Empty tensors are invalid.
// NaN inputs propagate to the result.
Tensor mean(const Tensor& x, Execution execution = Execution::Auto);

// Returns the maximum element as a scalar tensor. Empty tensors are invalid. NaN inputs
// propagate to the result; positive zero wins when the maximum values are both zero.
Tensor max(const Tensor& x, Execution execution = Execution::Auto);
// Returns the minimum element as a scalar tensor. Empty tensors are invalid. NaN inputs propagate
// to the result.
Tensor min(const Tensor& x, Execution execution = Execution::Auto);
// Returns the product of all elements as a scalar tensor. An empty tensor gives one.
Tensor prod(const Tensor& x, Execution execution = Execution::Auto);

// Reduces one axis and optionally retains it as a size-one dimension. Empty reduction axes are
// invalid for mean and max; sum over an empty axis returns zeros.

// Sums along one axis: a [R, C] input reduced over axis 1 gives the [R] row sums.
Tensor sum(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
// Mean along one axis.
Tensor mean(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
// Largest value along one axis.
Tensor max(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
// Smallest value along one axis.
Tensor min(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
// Product along one axis.
Tensor prod(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);

// Returns the index of the largest element independently along the last axis. Ties select the
// first index; a NaN selects its first index and propagates the NaN choice within that row.
// The result has the input shape with its last dimension removed and dtype int64. Empty axes are
// invalid. Only float32 input and CPU execution are supported so far.
Tensor argmax(const Tensor& x, int axis = -1, Execution execution = Execution::Auto);

// Returns the index of the smallest element independently along the last axis, with argmax's
// rules: only the last axis (axis -1 or dim - 1; others are invalid), ties select the first index,
// and a NaN counts as the extreme value, so the first NaN of a row is selected. The result has the
// input shape with its last dimension removed and dtype int64. Empty axes and scalars are invalid.
// Not differentiable. Only float32 input and CPU execution are supported so far.
Tensor argmin(const Tensor& x, int axis = -1, Execution execution = Execution::Auto);

// Variance along one axis (any axis; negative counts from the end), optionally keeping it as a
// size-one dimension:
//   var = sum((x - mean)^2) / (count - correction)
// with count the size of the axis. correction = 1 (the default, as in PyTorch) is the unbiased
// sample variance, correction = 0 the population variance; it must be >= 0. Computed in one pass
// with Welford's algorithm (mean and M2 in double), so a large mean with a small spread does not
// lose precision as sum(x^2) - count * mean^2 would.
// When count - correction <= 0 (for example one element with correction 1, or an empty axis) the
// result is NaN. A NaN or infinity among the reduced elements gives NaN. Records autograd for x;
// the backward pass is var_backward.
Tensor var(const Tensor& x, int axis, bool keepdim = false, std::int64_t correction = 1,
           Execution execution = Execution::Auto);

// The gradient of var(x, axis, keepdim, correction) with respect to x, given the gradient of its
// output (with var's output shape): grad * 2 * (x - mean) / (count - correction); NaN where
// count - correction <= 0.
Tensor var_backward(const Tensor& grad_output, const Tensor& x, int axis, bool keepdim = false,
                    std::int64_t correction = 1, Execution execution = Execution::Auto);

} // namespace mensurflow
