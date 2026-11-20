// Helpers for backward tests: deterministic test data, comparisons with a tolerance that report the
// first mismatch, central-difference gradients, and gradients through autograd.
//
// Every check uses the scalar loss L = sum(f(x) * weights) for a fixed random `weights` of f's
// output shape, so dL/df = weights and dL/dx is f's backward applied to weights.
#pragma once

#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace test {

// Values uniformly spread over [low, high) from a fixed linear congruential generator, so the data
// is the same with every compiler and standard library.
inline std::vector<float> random_values(std::int64_t n, std::uint32_t seed, float low = -1.0f,
                                        float high = 1.0f) {
    std::vector<float> values(static_cast<std::size_t>(n));
    std::uint32_t state = seed * 2654435761u + 12345u;
    for (float& value : values) {
        state = state * 1664525u + 1013904223u;
        const double unit = static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
        value = static_cast<float>(low + (high - low) * unit);
    }
    return values;
}

// |got[i] - want[i]| <= abs_tol + rel_tol * |want[i]| for every i (NaNs must match NaNs); on
// failure names the first index that is off.
template <typename Want>
void require_close(const std::vector<float>& got, const std::vector<Want>& want, double abs_tol,
                   double rel_tol, const std::string& label) {
    require(got.size() == want.size(), label + ": got " + std::to_string(got.size()) +
                                           " values, expected " + std::to_string(want.size()));
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double w = static_cast<double>(want[i]);
        if (std::isnan(g) || std::isnan(w)) {
            require(std::isnan(g) && std::isnan(w),
                    label + ": NaN mismatch at " + std::to_string(i) + " (got " +
                        std::to_string(g) + ", expected " + std::to_string(w) + ")");
            continue;
        }
        if (std::abs(g - w) > abs_tol + rel_tol * std::abs(w)) {
            require(false, label + ": index " + std::to_string(i) + " is " + std::to_string(g) +
                               ", expected " + std::to_string(w));
        }
    }
}

// sum(values * weights), in double.
inline double weighted_sum(const std::vector<float>& values, const std::vector<float>& weights) {
    require(values.size() == weights.size(), "weighted_sum: sizes differ");
    double sum = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        sum += static_cast<double>(values[i]) * weights[i];
    }
    return sum;
}

// Central differences of L(x) = sum(forward(x) * weights) for each element of x, where forward maps
// the host values of x to the host values of its output. Each element is moved by +-eps.
template <typename Forward>
std::vector<double> numeric_gradient(Forward&& forward, std::vector<float> x,
                                     const std::vector<float>& weights, float eps) {
    std::vector<double> gradient(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float saved = x[i];
        x[i] = saved + eps;
        const double plus = weighted_sum(forward(x), weights);
        x[i] = saved - eps;
        const double minus = weighted_sum(forward(x), weights);
        x[i] = saved;
        gradient[i] = (plus - minus) / (2.0 * static_cast<double>(eps));
    }
    return gradient;
}

// L = sum(output * weights) as a scalar tensor that autograd can differentiate.
inline mensurflow::Tensor weighted_loss(const mensurflow::Tensor& output,
                                        const std::vector<float>& weights) {
    const mensurflow::Tensor w =
        mensurflow::Tensor::from_vector(weights, output.shape(), output.device());
    return mensurflow::sum(mensurflow::mul(output, w));
}

} // namespace test
