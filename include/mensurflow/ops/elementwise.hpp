// Elementwise arithmetic. The last argument says where the op runs (see execution.hpp); the result
// is on the inputs' device. Backend coverage varies by operation. The call may return before a
// GPU finishes; reading the result (item(), to_vector(), copy_to_host()) waits for it.
//
// Binary inputs must have the same dtype and device. Their shapes may differ if NumPy broadcasting
// applies (equal sizes, or a size of 1, aligned from the right): a scalar, a row added to every
// row of a matrix, and so on. The result has the broadcast shape and is a new contiguous tensor;
// inputs are never modified. Broadcast or strided inputs are expanded into a packed copy before
// the kernel runs, so they cost extra memory traffic compared with two same-shape tensors.
// Only float32 so far.
//
// Throw invalid_argument for undefined tensors, mismatched dtypes or devices, and shapes that do
// not broadcast. NotImplemented (a runtime_error) means the execution device has no kernel for
// that op and dtype. Division follows IEEE 754: x / 0 is inf or NaN.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow {

// a + b
Tensor add(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a - b
Tensor sub(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a * b, elementwise (not matrix multiplication)
Tensor mul(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a / b
Tensor div(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a raised to b
Tensor pow(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);

// alpha*x + y, using a fused float32 multiply-add (one rounding, as std::fma).
// Shapes broadcast like add; alpha is a float scalar. Returns a new contiguous tensor.
// CPU float32 only; unsupported devices/dtypes throw NotImplemented, including empty inputs.
Tensor axpy(float alpha, const Tensor& x, const Tensor& y, Execution execution = Execution::Auto);

// Add a [features] bias to every row of a [batch, features] matrix.
// Both inputs must have the same dtype and device. Views are packed as needed without
// expanding the bias to matrix size. Returns a new contiguous tensor; inputs are unchanged.
// Wrong rank or bias length throws invalid_argument. CPU float32 only; unsupported
// devices/dtypes throw NotImplemented, including empty matrices.
Tensor bias_add(const Tensor& x, const Tensor& bias, Execution execution = Execution::Auto);

// Unary operations return a new contiguous tensor with the same shape, dtype, and device;
// strided and broadcast views are supported. Currently CPU float32 only. Undefined input throws
// invalid_argument; unsupported devices or dtypes throw NotImplemented, including empty inputs.
// -x
Tensor neg(const Tensor& x, Execution execution = Execution::Auto);
// |x|, including -0 -> +0 and clearing a NaN's sign
Tensor abs(const Tensor& x, Execution execution = Execution::Auto);
// x * x with float32 rounding and overflow
Tensor square(const Tensor& x, Execution execution = Execution::Auto);
// x < 0 ? +0 : x; preserves NaNs and signed zeros
Tensor relu(const Tensor& x, Execution execution = Execution::Auto);
// sqrt(x); negatives -> NaN, preserves signed zeros
Tensor sqrt(const Tensor& x, Execution execution = Execution::Auto);
// exp(x); overflow -> +inf, underflow may produce +0
Tensor exp(const Tensor& x, Execution execution = Execution::Auto);
// stable logistic function; -inf -> 0, +inf -> 1
Tensor sigmoid(const Tensor& x, Execution execution = Execution::Auto);
// natural logarithm
Tensor log(const Tensor& x, Execution execution = Execution::Auto);
// reciprocal square root
Tensor rsqrt(const Tensor& x, Execution execution = Execution::Auto);
// hyperbolic tangent
Tensor tanh(const Tensor& x, Execution execution = Execution::Auto);
Tensor sin(const Tensor& x, Execution execution = Execution::Auto);
Tensor cos(const Tensor& x, Execution execution = Execution::Auto);
Tensor reciprocal(const Tensor& x, Execution execution = Execution::Auto);

// elementwise minimum
Tensor minimum(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// elementwise maximum
Tensor maximum(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
Tensor clamp(const Tensor& x, float lower, float upper, Execution execution = Execution::Auto);

} // namespace mensurflow
