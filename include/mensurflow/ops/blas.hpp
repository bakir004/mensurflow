// BLAS-style operations. The last argument says where the op runs (see execution.hpp); the result
// is on the inputs' device. The call may return before a GPU finishes; reading the result
// (item(), to_vector(), copy_to_host()) waits for it.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow {

// Inner product of two 1-D tensors of the same length, dtype, and device. Returns a scalar
// (0-d) tensor on that device; use result.item<float>() to read it. Empty inputs give 0.
// Strided views (slices with a step, expand) are accepted and made contiguous first.
// Only float32 so far.
// Throws invalid_argument for wrong rank, length, dtype, or mixed devices, and NotImplemented
// (a runtime_error) if the device has no kernel for the dtype.
Tensor dot(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);

// Scales every element of a vector by alpha. The result is a new contiguous tensor.
Tensor scal(float alpha, const Tensor& x, Execution execution = Execution::Auto);

// Sum of absolute values of a vector. Empty inputs give zero.
Tensor asum(const Tensor& x, Execution execution = Execution::Auto);

// Euclidean norm of a vector. Empty inputs give zero.
Tensor nrm2(const Tensor& x, Execution execution = Execution::Auto);

// Copies x into a new contiguous tensor of the same shape, dtype and device (a strided view is
// packed). The result never shares storage with x.
Tensor copy(const Tensor& x, Execution execution = Execution::Auto);

// Matrix-vector product of a [M, K] and a [K] tensor, giving a new contiguous [M] tensor on the
// same device. Both inputs must have the same dtype and device. An empty M gives an empty result;
// K == 0 gives zeros. Only float32, and only on the CPU so far.
Tensor matvec(const Tensor& a, const Tensor& x, Execution execution = Execution::Auto);

// General matrix-vector multiply: alpha * op(a) * x + beta * y, where a is [M, N], op(a) is a or
// its transpose (transpose_a), and y is optional (zeros when null; beta needs a y). Gives a new
// contiguous vector of op(a)'s row count. Only float32, and only on the CPU so far.
Tensor gemv(const Tensor& a, const Tensor& x, float alpha = 1.0f, const Tensor* y = nullptr,
            float beta = 0.0f, bool transpose_a = false, Execution execution = Execution::Auto);

// Rank-1 update: alpha * outer(x, y) + a, for an [M] x and an [N] y, giving a new contiguous [M, N]
// tensor. a is optional (zeros when null) and must be [M, N]. All tensors must share dtype and
// device. Only float32, and only on the CPU so far. Not differentiable yet.
// Throws invalid_argument for wrong ranks or shapes, mixed dtypes or devices.
Tensor ger(float alpha, const Tensor& x, const Tensor& y, const Tensor* a = nullptr,
           Execution execution = Execution::Auto);

// Matrix product of a [M, K] and a [K, N] tensor, giving a new contiguous [M, N] tensor on the same
// device. Both inputs must have the same dtype and device. Transposed or sliced views are accepted
// and made contiguous first. An empty M or N gives an empty result; K == 0 gives zeros.
// Only float32, and only on the CPU so far: a GPU device reports NotImplemented.
// Throws invalid_argument for non-2-D inputs, mismatched inner dimensions, dtypes or devices.
Tensor matmul(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);

// General matrix multiply. Computes alpha * op(a) * op(b) + beta * c, where c is optional and
// defaults to zero. The result is always a new contiguous tensor.
Tensor gemm(const Tensor& a, const Tensor& b, float alpha = 1.0f, const Tensor* c = nullptr,
            float beta = 0.0f, bool transpose_a = false, bool transpose_b = false,
            Execution execution = Execution::Auto);

// Batched gemm: alpha * op(a[i]) * op(b[i]) + beta * c[i] for every matrix i of a batch. a is
// [..., M, K] and b is [..., K, N] (or [..., K, M] and [..., N, K] with the transpose flags); the
// leading batch dimensions broadcast like NumPy's matmul, so a [B, M, K] times a [K, N] multiplies
// every matrix by the same right-hand side without copying it. The result is a new contiguous
// [batch..., M, N] tensor. c is optional and must broadcast to the result's shape; beta needs a c.
// Transposed, sliced and expanded inputs are read in place. With the defaults this is a batched
// matmul (torch.matmul / torch.bmm). Parallel over the batch or within each matrix, whichever
// suits the shapes. Only float32, and only on the CPU so far.
// Not differentiable yet.
// Throws invalid_argument for inputs below 2-D, mismatched inner dimensions, batch shapes that do
// not broadcast, a c that does not broadcast to the result, beta without c, or mixed dtypes or
// devices.
Tensor gemm_strided_batched(const Tensor& a, const Tensor& b, float alpha = 1.0f,
                            const Tensor* c = nullptr, float beta = 0.0f, bool transpose_a = false,
                            bool transpose_b = false, Execution execution = Execution::Auto);

} // namespace mensurflow
