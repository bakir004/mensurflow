// Numerically stable softmax operations.
#pragma once

#include "mensurflow/tensor.hpp"

namespace mensurflow {

// Applies softmax independently along the last axis. Axis -1 or the normalized last-axis index
// is accepted; other axes are not implemented yet. NaNs propagate through their row. A row with
// positive infinity assigns equal probability to its positive-infinity entries and zero elsewhere.
// Only float32, and only on the CPU so far.
Tensor softmax(const Tensor& x, int axis = -1);

} // namespace mensurflow
