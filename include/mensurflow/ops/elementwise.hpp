// Elementwise arithmetic. The last argument says where the op runs (see execution.hpp); the result
// is on the inputs' device. Backend coverage varies by operation. The call may return before a
// GPU finishes; reading the result (item(), to_vector(), copy_to_host()) waits for it.
//
// Binary inputs must have the same dtype and device. Their shapes may differ if NumPy broadcasting
// applies (equal sizes, or a size of 1, aligned from the right): a scalar, a row added to every
// row of a matrix, and so on. The result has the broadcast shape and is a new contiguous tensor;
// inputs are never modified. Broadcast or strided inputs are expanded into a packed copy before
// the kernel runs, so they cost extra memory traffic compared with two same-shape tensors.
// Arithmetic is float32; comparisons return Bool and cast supports the dtypes documented below.
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
// Negative values are multiplied by negative_slope.
Tensor leaky_relu(const Tensor& x, float negative_slope = 0.01f,
                  Execution execution = Execution::Auto);
// Gaussian error linear unit. approximate=true selects the tanh approximation.
// GELU and SiLU use their limiting value -0 at -infinity, and +infinity at +infinity.
Tensor gelu(const Tensor& x, bool approximate = false, Execution execution = Execution::Auto);
// x * sigmoid(x), computed without exponential overflow.
Tensor silu(const Tensor& x, Execution execution = Execution::Auto);
// Negative values become alpha * expm1(x).
Tensor elu(const Tensor& x, float alpha = 1.0f, Execution execution = Execution::Auto);
// log(1 + exp(beta*x)) / beta; returns x when beta*x > threshold.
// beta must be positive and finite; threshold must be finite.
Tensor softplus(const Tensor& x, float beta = 1.0f, float threshold = 20.0f,
                Execution execution = Execution::Auto);

// Activation derivatives. `x` is the forward input and `gradient` is the upstream gradient.
// Both tensors must have the same shape, dtype, and device. CPU float32 is supported.

// Backward pass of relu: gradient where x > 0, otherwise 0.
Tensor relu_backward(const Tensor& x, const Tensor& gradient,
                     Execution execution = Execution::Auto);
// Backward pass of sigmoid: gradient * s * (1 - s) with s = sigmoid(x).
Tensor sigmoid_backward(const Tensor& x, const Tensor& gradient,
                        Execution execution = Execution::Auto);
// Backward pass of tanh: gradient * (1 - tanh(x)^2).
Tensor tanh_backward(const Tensor& x, const Tensor& gradient,
                     Execution execution = Execution::Auto);
// Backward pass of gelu: gradient * gelu'(x), exact or tanh-approximate to match the forward.
Tensor gelu_backward(const Tensor& x, const Tensor& gradient, bool approximate = false,
                     Execution execution = Execution::Auto);
// Backward pass of silu: gradient * s * (1 + x * (1 - s)) with s = sigmoid(x).
Tensor silu_backward(const Tensor& x, const Tensor& gradient,
                     Execution execution = Execution::Auto);
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
// sine of x (radians)
Tensor sin(const Tensor& x, Execution execution = Execution::Auto);
// cosine of x (radians)
Tensor cos(const Tensor& x, Execution execution = Execution::Auto);
// 1 / x; 0 -> inf with the sign of the zero
Tensor reciprocal(const Tensor& x, Execution execution = Execution::Auto);

// elementwise minimum
Tensor minimum(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// elementwise maximum
Tensor maximum(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// Limits every element to [lower, upper]: values below lower become lower, above upper become
// upper. lower must not exceed upper.
Tensor clamp(const Tensor& x, float lower, float upper, Execution execution = Execution::Auto);

// Elementwise F32 comparisons. Inputs must have the same dtype/device and broadcast normally.
// Return a new contiguous Bool tensor on the inputs' device, with byte values 0 or 1.
// NaN follows ordinary floating-point comparisons: only ne is true for a NaN operand.
// CPU F32 only; unsupported dtype/backend throws NotImplemented, including empty inputs.
// Comparisons do not track gradients. Read Bool values with copy_to_host into uint8_t storage.

// a == b, elementwise, as a Bool tensor
Tensor eq(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a != b, elementwise, as a Bool tensor (true when either is NaN)
Tensor ne(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a < b, elementwise, as a Bool tensor
Tensor lt(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a <= b, elementwise, as a Bool tensor
Tensor le(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a > b, elementwise, as a Bool tensor
Tensor gt(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);
// a >= b, elementwise, as a Bool tensor
Tensor ge(const Tensor& a, const Tensor& b, Execution execution = Execution::Auto);

// Select x where condition is nonzero, otherwise y. Condition must be Bool; x/y must be F32
// with the same dtype/device. All three shapes broadcast together. Returns a new contiguous F32
// tensor on their device and preserves the selected value's bits, including signed zeros/NaNs.
// CPU forward only: no gradient graph is recorded. Unsupported dtype/backend throws
// NotImplemented even for empty results; invalid condition, devices or shapes throw invalid_argument.
Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y,
             Execution execution = Execution::Auto);

// Convert between F32, I32, F16 and BF16, returning a fresh contiguous tensor of the same
// shape and device, even when the dtype is unchanged. Views are packed before conversion.
// Floating results round to nearest, ties to even. Floating-to-I32 truncates toward zero;
// NaN, infinity or an out-of-range value produces INT32_MIN. Same-dtype casts preserve bits.
// CPU forward only: no gradient graph is recorded. Unsupported source/target dtype or backend
// throws NotImplemented, including empty inputs. Undefined input/invalid dtype throws invalid_argument.
Tensor cast(const Tensor& x, DType dtype, Execution execution = Execution::Auto);

} // namespace mensurflow
