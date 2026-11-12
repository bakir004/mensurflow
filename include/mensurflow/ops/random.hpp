// Seeded CPU random generation and dropout. All random values come from Philox4x32-10.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <cstdint>

namespace mensurflow {

// Factories return new contiguous F32 tensors on the CPU. Shape {} produces one scalar;
// zero dimensions produce an empty tensor. There is no global RNG state: an identical seed,
// offset and distribution produces identical values on every CPU thread count. offset counts
// output elements (not Philox blocks), so adjoining calls can continue a stream with offset+n.
// Values are reproducible within a build; normal's transcendental rounding may differ across
// platforms. Unsupported execution devices throw, including for empty tensors.
// offset + numel(shape) - 1 must fit in uint64_t for a nonempty result.

// Uniform values in [low, high). Finite low <= high is required; low == high gives a constant.
Tensor uniform(Shape shape, float low = 0.0f, float high = 1.0f, std::uint64_t seed = 0,
               std::uint64_t offset = 0, Execution execution = Execution::Auto);

// Gaussian values, using Box-Muller. mean and stddev must be finite, stddev >= 0.
// stddev == 0 gives mean. Very large finite parameters can overflow F32 outputs.
Tensor normal(Shape shape, float mean = 0.0f, float stddev = 1.0f, std::uint64_t seed = 0,
              std::uint64_t offset = 0, Execution execution = Execution::Auto);

// F32 values 0 or 1, with probability of 1 equal to probability (finite, in [0, 1]).
Tensor bernoulli(Shape shape, float probability = 0.5f, std::uint64_t seed = 0,
                 std::uint64_t offset = 0, Execution execution = Execution::Auto);

struct DropoutResult {
    Tensor output;
    Tensor mask;
};

// Inverted dropout: training keeps each element with probability 1-probability and scales
// kept values by 1/(1-probability). Dropped values are +0, even for NaN/Inf inputs. At p=1
// the output is all zero. Evaluation and p=0 copy x and return an all-true mask.
// Returns a fresh F32 output and Bool mask of x.shape() on x.device(); views are accepted.
// Autograd saves a private copy of the mask, so changes to the returned mask do not affect it.
// CPU F32 only. Undefined tensors or invalid probability throw invalid_argument;
// unsupported dtype/backend throws NotImplemented, including empty inputs.
DropoutResult dropout(const Tensor& x, float probability = 0.5f, bool training = true,
                      std::uint64_t seed = 0, std::uint64_t offset = 0,
                      Execution execution = Execution::Auto);

// Applies the forward mask and scaling to an upstream F32 gradient. mask must be Bool,
// with the same shape and device. Use the same probability and training as in forward.
Tensor dropout_backward(const Tensor& gradient, const Tensor& mask, float probability = 0.5f,
                        bool training = true, Execution execution = Execution::Auto);

} // namespace mensurflow
