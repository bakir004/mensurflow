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
Tensor min(const Tensor& x, Execution execution = Execution::Auto);
Tensor prod(const Tensor& x, Execution execution = Execution::Auto);

// Reduces one axis and optionally retains it as a size-one dimension. Empty reduction axes are
// invalid for mean and max; sum over an empty axis returns zeros.
Tensor sum(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
Tensor mean(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
Tensor max(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
Tensor min(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);
Tensor prod(const Tensor& x, int axis, bool keepdim = false, Execution execution = Execution::Auto);

// Returns the index of the largest element independently along the last axis. Ties select the
// first index; a NaN selects its first index and propagates the NaN choice within that row.
// The result has the input shape with its last dimension removed and dtype int64. Empty axes are
// invalid. Only float32 input and CPU execution are supported so far.
Tensor argmax(const Tensor& x, int axis = -1, Execution execution = Execution::Auto);

} // namespace mensurflow
