// Numerically stable softmax operations.
#pragma once

#include "mensurflow/tensor.hpp"

namespace mensurflow {

// Applies softmax independently along the selected axis. Negative axes count from the end.
// NaNs propagate through their row. A row with
// positive infinity assigns equal probability to its positive-infinity entries and zero elsewhere.
// Only float32, and only on the CPU so far.
Tensor softmax(const Tensor& x, int axis = -1);

} // namespace mensurflow
