// Elementwise arithmetic. Each op runs on the device its tensors live on; backend coverage varies
// by operation. The call may return before a GPU finishes; reading the
// result (item(), to_vector(), copy_to_host()) waits for it.
//
// Binary inputs must have the same dtype and device. Their shapes may differ if NumPy broadcasting
// applies (equal sizes, or a size of 1, aligned from the right): a scalar, a row added to every
// row of a matrix, and so on. The result has the broadcast shape and is a new contiguous tensor;
// inputs are never modified. Broadcast or strided inputs are expanded into a packed copy before
// the kernel runs, so they cost extra memory traffic compared with two same-shape tensors.
// Only float32 so far.
//
// Throw invalid_argument for undefined tensors, mismatched dtypes or devices, and shapes that do
// not broadcast. NotImplemented (a runtime_error) means the device has no kernel for that op and
// dtype. Division follows IEEE 754: x / 0 is inf or NaN.
#pragma once

#include "mensurflow/tensor.hpp"

namespace mensurflow {

Tensor add(const Tensor& a, const Tensor& b); // a + b
Tensor sub(const Tensor& a, const Tensor& b); // a - b
Tensor mul(const Tensor& a, const Tensor& b); // a * b, elementwise (not matrix multiplication)
Tensor div(const Tensor& a, const Tensor& b); // a / b

// Elementwise -x. Returns a new contiguous tensor with the same shape, dtype, and device;
// strided and broadcast views are supported. Currently CPU float32 only. Undefined input throws
// invalid_argument; unsupported devices or dtypes throw NotImplemented, including empty inputs.
Tensor neg(const Tensor& x);

} // namespace mensurflow
