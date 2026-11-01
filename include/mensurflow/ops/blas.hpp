// BLAS-style operations. Each one runs on the device its tensors live on, so the same call
// works on CPU, CUDA, ROCm, and Metal builds. The call may return before a GPU finishes; reading
// the result (item(), to_vector(), copy_to_host()) waits for it.
#pragma once

#include "mensurflow/tensor.hpp"

namespace mensurflow {

// Inner product of two 1-D tensors of the same length, dtype, and device. Returns a scalar
// (0-d) tensor on that device; use result.item<float>() to read it. Empty inputs give 0.
// Strided views (slices with a step, expand) are accepted and made contiguous first.
// Only float32 so far.
// Throws invalid_argument for wrong rank, length, dtype, or mixed devices, and NotImplemented
// (a runtime_error) if the device has no kernel for the dtype.
Tensor dot(const Tensor& a, const Tensor& b);

// Matrix-vector product of a [M, K] and a [K] tensor, giving a new contiguous [M] tensor on the
// same device. Both inputs must have the same dtype and device. An empty M gives an empty result;
// K == 0 gives zeros. Only float32, and only on the CPU so far.
Tensor matvec(const Tensor& a, const Tensor& x);

// Matrix product of a [M, K] and a [K, N] tensor, giving a new contiguous [M, N] tensor on the same
// device. Both inputs must have the same dtype and device. Transposed or sliced views are accepted
// and made contiguous first. An empty M or N gives an empty result; K == 0 gives zeros.
// Only float32, and only on the CPU so far: a GPU device reports NotImplemented.
// Throws invalid_argument for non-2-D inputs, mismatched inner dimensions, dtypes or devices.
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace mensurflow
