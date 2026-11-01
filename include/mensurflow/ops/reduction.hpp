// Full-tensor reductions. The initial implementation supports float32 on the CPU.
#pragma once

#include "mensurflow/tensor.hpp"

namespace mensurflow {

// Returns the sum of all elements as a scalar tensor. An empty tensor sums to zero.
// NaN inputs propagate to the result.
Tensor sum(const Tensor& x);

// Returns the arithmetic mean of all elements as a scalar tensor. Empty tensors are invalid.
// NaN inputs propagate to the result.
Tensor mean(const Tensor& x);

// Returns the maximum element as a scalar tensor. Empty tensors are invalid. NaN inputs
// propagate to the result; equal values follow std::fmax semantics.
Tensor max(const Tensor& x);

} // namespace mensurflow
