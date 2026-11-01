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

// alpha*x + y, using a fused float32 multiply-add (one rounding, as std::fma).
// Shapes broadcast like add; alpha is a float scalar. Returns a new contiguous tensor.
// CPU float32 only; unsupported devices/dtypes throw NotImplemented, including empty inputs.
Tensor axpy(float alpha, const Tensor& x, const Tensor& y);

// Add a [features] bias to every row of a [batch, features] matrix.
// Both inputs must have the same dtype and device. Views are packed as needed without
// expanding the bias to matrix size. Returns a new contiguous tensor; inputs are unchanged.
// Wrong rank or bias length throws invalid_argument. CPU float32 only; unsupported
// devices/dtypes throw NotImplemented, including empty matrices.
Tensor bias_add(const Tensor& x, const Tensor& bias);

// Unary operations return a new contiguous tensor with the same shape, dtype, and device;
// strided and broadcast views are supported. Currently CPU float32 only. Undefined input throws
// invalid_argument; unsupported devices or dtypes throw NotImplemented, including empty inputs.
Tensor neg(const Tensor& x);    // -x
Tensor abs(const Tensor& x);    // |x|, including -0 -> +0 and clearing a NaN's sign
Tensor square(const Tensor& x); // x * x with float32 rounding and overflow
Tensor relu(const Tensor& x);   // x < 0 ? +0 : x; preserves NaNs and signed zeros
Tensor sqrt(const Tensor& x);   // sqrt(x); negatives -> NaN, preserves signed zeros
Tensor exp(const Tensor& x);    // exp(x); overflow -> +inf, underflow may produce +0
Tensor sigmoid(const Tensor& x); // stable logistic function; -inf -> 0, +inf -> 1
Tensor log(const Tensor& x);    // natural logarithm
Tensor rsqrt(const Tensor& x);  // reciprocal square root
Tensor tanh(const Tensor& x);   // hyperbolic tangent

Tensor minimum(const Tensor& a, const Tensor& b); // elementwise minimum
Tensor maximum(const Tensor& a, const Tensor& b); // elementwise maximum
Tensor clamp(const Tensor& x, float lower, float upper);

} // namespace mensurflow
