// Independent single-threaded references. Basic arithmetic is checked exactly; transcendental
// kernels allow float32 rounding against float64 references, while sequential/parallel results
// must still match bit for bit.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

template <typename Op>
void binary(const float* a, const float* b, float* out, std::int64_t n, Op op) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = op(a[i], b[i]);
    }
}

inline void add(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x + y; });
}

inline void sub(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x - y; });
}

inline void mul(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x * y; });
}

inline void div(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x / y; });
}

inline float axpy_value(float alpha, float x, float y) {
    return static_cast<float>(
        std::fma(static_cast<double>(alpha), static_cast<double>(x), static_cast<double>(y)));
}

inline void axpy(float alpha, const float* x, const float* y, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = axpy_value(alpha, x[i], y[i]);
    }
}

inline void bias_add(const float* x, const float* bias, float* out, std::int64_t n,
                     std::int64_t features) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) + bias[i % features]);
    }
}

// Compute in float64, then round back to the float32 kernel's output type.
inline void neg(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(-static_cast<double>(x[i]));
    }
}

inline void abs(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(std::fabs(static_cast<double>(x[i])));
    }
}

inline void square(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        const double value = static_cast<double>(x[i]);
        out[i] = static_cast<float>(value * value);
    }
}

inline void relu(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        const double value = static_cast<double>(x[i]);
        out[i] = static_cast<float>(value < 0.0 ? 0.0 : value);
    }
}

inline void sqrt(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(std::sqrt(static_cast<double>(x[i])));
    }
}

inline void exp(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(std::exp(static_cast<double>(x[i])));
    }
}

inline void sigmoid(const float* x, float* out, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        const double value = static_cast<double>(x[i]);
        const double z = std::exp(-std::fabs(value));
        out[i] = static_cast<float>(value < 0.0 ? z / (1.0 + z) : 1.0 / (1.0 + z));
    }
}

// Activation references evaluate the mathematical definitions in float64.
inline double leaky_relu_value(double x, double slope = 0.01f) {
    return x < 0.0 ? slope * x : x;
}

inline double gelu_value(double x, bool approximate = false) {
    if (std::isinf(x)) {
        return x < 0.0 ? -0.0 : x;
    }
    if (!approximate) {
        return x * std::erfc(-x / std::sqrt(2.0)) / 2.0;
    }
    const double u = std::sqrt(2.0 / std::acos(-1.0)) * (x + 0.044715 * x * x * x);
    // Use the tanh definition in its well-conditioned range and the equivalent tail outside it.
    if (std::fabs(u) < 10.0) {
        return 0.5 * x * (1.0 + std::tanh(u));
    }
    return u < 0.0 ? x * std::exp(2.0 * u) / (1.0 + std::exp(2.0 * u))
                   : x / (1.0 + std::exp(-2.0 * u));
}

inline double silu_value(double x) {
    if (std::isinf(x) && x < 0.0) {
        return -0.0;
    }
    return x < 0.0 ? x * std::exp(x) / (1.0 + std::exp(x)) : x / (1.0 + std::exp(-x));
}

inline double elu_value(double x, double alpha = 1.0) {
    return x < 0.0 ? alpha * std::expm1(x) : x;
}

inline double softplus_value(double x, double beta = 1.0, double threshold = 20.0) {
    const double z = beta * x;
    if (z > threshold) {
        return x;
    }
    return z < 0.0 ? std::log1p(std::exp(z)) / beta
                   : (z + std::log1p(std::exp(-z))) / beta;
}

template <typename Function>
void activation(const float* x, float* out, std::int64_t n, Function apply) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(apply(static_cast<double>(x[i])));
    }
}

inline void leaky_relu(const float* x, float* out, std::int64_t n) {
    activation(x, out, n, [](double v) { return leaky_relu_value(v); });
}

inline void gelu(const float* x, float* out, std::int64_t n) {
    activation(x, out, n, [](double v) { return gelu_value(v); });
}

inline void gelu_tanh(const float* x, float* out, std::int64_t n) {
    activation(x, out, n, [](double v) { return gelu_value(v, true); });
}

inline void silu(const float* x, float* out, std::int64_t n) {
    activation(x, out, n, [](double v) { return silu_value(v); });
}

inline void elu(const float* x, float* out, std::int64_t n) {
    activation(x, out, n, [](double v) { return elu_value(v); });
}

inline void softplus(const float* x, float* out, std::int64_t n) {
    activation(x, out, n, [](double v) { return softplus_value(v); });
}

} // namespace reference

// Broadcasting binary op written from scratch, so tests do not rely on the library's own
// broadcast_shapes/expand. Shapes are NumPy-compatible; values are row-major.
namespace reference {

struct Broadcasted {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

template <typename Op>
Broadcasted broadcast_binary(const std::vector<float>& a, const std::vector<std::int64_t>& a_shape,
                             const std::vector<float>& b, const std::vector<std::int64_t>& b_shape,
                             Op op) {
    const std::size_t nd = std::max(a_shape.size(), b_shape.size());
    // Size of dimension d (counted from the left of the output) for one operand, 1 if missing.
    auto dim_of = [nd](const std::vector<std::int64_t>& s, std::size_t d) {
        const std::size_t lead = nd - s.size();
        return d < lead ? std::int64_t{1} : s[d - lead];
    };
    Broadcasted out;
    std::int64_t total = 1;
    for (std::size_t d = 0; d < nd; ++d) {
        // A size-1 dimension stretches to the other operand's size, which may be 0.
        const std::int64_t da = dim_of(a_shape, d), db = dim_of(b_shape, d);
        out.shape.push_back(da == 1 ? db : da);
        total *= out.shape.back();
    }
    for (std::int64_t flat = 0; flat < total; ++flat) {
        // Unravel the flat output index, then re-ravel it into each operand, pinning size-1 dims.
        std::int64_t rest = flat, ia = 0, ib = 0;
        std::int64_t stride_a = 1, stride_b = 1;
        for (std::size_t d = nd; d-- > 0;) {
            const std::int64_t index = rest % out.shape[d];
            rest /= out.shape[d];
            ia += (dim_of(a_shape, d) == 1 ? 0 : index) * stride_a;
            ib += (dim_of(b_shape, d) == 1 ? 0 : index) * stride_b;
            stride_a *= dim_of(a_shape, d);
            stride_b *= dim_of(b_shape, d);
        }
        out.values.push_back(op(a[static_cast<std::size_t>(ia)], b[static_cast<std::size_t>(ib)]));
    }
    return out;
}

} // namespace reference
