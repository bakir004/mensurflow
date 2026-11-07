// CPU elementwise kernels: one loop per operation. MENSURFLOW_PARALLEL_FOR (threads.hpp) splits
// the loop across threads when the build has OpenMP; its `if` keeps small inputs on one thread.
#include "backends/cpu/ops/elementwise/elementwise.hpp"

#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cmath>

namespace mensurflow::cpu {

namespace {

// Activation kernels share traversal and the in-place contract, with a cutoff per operation.
template <ElementwiseOp Op, typename Function>
void activation_f32(const float* x, float* out, std::int64_t n, Function apply) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(Op)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = apply(x[i]);
    }
}

double stable_sigmoid(double value) {
    const double z = std::exp(-std::fabs(value));
    return value < 0.0 ? z / (1.0 + z) : 1.0 / (1.0 + z);
}

template <ElementwiseOp Op, typename Compare>
void comparison_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n,
                     Compare compare) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(Op)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<std::uint8_t>(compare(a[i], b[i]));
    }
}

} // namespace

void eq_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison_f32<ElementwiseOp::eq>(a, b, out, n, [](float x, float y) { return x == y; });
}

void ne_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison_f32<ElementwiseOp::ne>(a, b, out, n, [](float x, float y) { return x != y; });
}

void lt_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison_f32<ElementwiseOp::lt>(a, b, out, n, [](float x, float y) { return x < y; });
}

void le_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison_f32<ElementwiseOp::le>(a, b, out, n, [](float x, float y) { return x <= y; });
}

void gt_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison_f32<ElementwiseOp::gt>(a, b, out, n, [](float x, float y) { return x > y; });
}

void ge_f32(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison_f32<ElementwiseOp::ge>(a, b, out, n, [](float x, float y) { return x >= y; });
}

void where_f32(const std::uint8_t* condition, const float* x, const float* y, float* out,
               std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::where)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = condition[i] != 0 ? x[i] : y[i];
    }
}

void add_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::add)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::sub)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::mul)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void div_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::div)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

void pow_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::pow)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::pow(a[i], b[i]);
    }
}

void minimum_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::minimum)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::isnan(a[i]) ? a[i] : (std::isnan(b[i]) ? b[i] : std::fmin(a[i], b[i]));
    }
}

void maximum_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::maximum)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::isnan(a[i]) ? a[i] : (std::isnan(b[i]) ? b[i] : std::fmax(a[i], b[i]));
    }
}

void axpy_f32(float alpha, const float* x, const float* y, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::axpy)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::fma(alpha, x[i], y[i]);
    }
}

void bias_add_f32(const float* x, const float* bias, float* out, std::int64_t n,
                  std::int64_t features) {
    const std::int64_t rows = features == 0 ? 0 : n / features;
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::bias_add)))
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t j = 0; j < features; ++j) {
            out[row * features + j] = x[row * features + j] + bias[j];
        }
    }
}

void neg_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::neg)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = -x[i];
    }
}

void abs_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::abs)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::fabs(x[i]);
    }
}

void square_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::square)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = x[i] * x[i];
    }
}

void relu_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::relu)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = x[i] < 0.0f ? 0.0f : x[i];
    }
}

void leaky_relu_f32(const float* x, float* out, std::int64_t n, float negative_slope) {
    activation_f32<ElementwiseOp::leaky_relu>(x, out, n, [negative_slope](float value) {
        return value < 0.0f ? negative_slope * value : value;
    });
}

void gelu_f32(const float* x, float* out, std::int64_t n, bool approximate) {
    // Compute the CDF in double to retain the negative tail before rounding to F32.
    // sigmoid(2*u) is algebraically (1+tanh(u))/2 without negative-tail cancellation.
    if (approximate) {
        activation_f32<ElementwiseOp::gelu_tanh>(x, out, n, [](float value) {
            if (std::isinf(value)) {
                return value < 0.0f ? -0.0f : value;
            }
            const double v = value;
            const double u = 0.79788456080286535588 * v * (1.0 + 0.044715 * v * v);
            return static_cast<float>(v * stable_sigmoid(2.0 * u));
        });
    } else {
        activation_f32<ElementwiseOp::gelu>(x, out, n, [](float value) {
            if (std::isinf(value)) {
                return value < 0.0f ? -0.0f : value;
            }
            const double v = value;
            return static_cast<float>(0.5 * v * std::erfc(-v * 0.70710678118654752440));
        });
    }
}

void silu_f32(const float* x, float* out, std::int64_t n) {
    activation_f32<ElementwiseOp::silu>(x, out, n, [](float value) {
        if (std::isinf(value) && value < 0.0f) {
            return -0.0f;
        }
        // Keep the sigmoid tail in double until after multiplying by x.
        return static_cast<float>(static_cast<double>(value) * stable_sigmoid(value));
    });
}

void elu_f32(const float* x, float* out, std::int64_t n, float alpha) {
    activation_f32<ElementwiseOp::elu>(x, out, n, [alpha](float value) {
        return value < 0.0f ? alpha * std::expm1(value) : value;
    });
}

void softplus_f32(const float* x, float* out, std::int64_t n, float beta, float threshold) {
    activation_f32<ElementwiseOp::softplus>(x, out, n, [beta, threshold](float value) {
        const double scaled = static_cast<double>(beta) * value;
        if (scaled > threshold) {
            return value;
        }
        return static_cast<float>(
            (std::max(scaled, 0.0) + std::log1p(std::exp(-std::fabs(scaled)))) / beta);
    });
}

void sqrt_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::sqrt)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::sqrt(x[i]);
    }
}

void exp_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::exp)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::exp(x[i]);
    }
}

void sigmoid_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::sigmoid)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = x[i] >= 0.0f ? 1.0f / (1.0f + std::exp(-x[i]))
                              : std::exp(x[i]) / (1.0f + std::exp(x[i]));
    }
}

void log_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::log)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::log(x[i]);
    }
}

void rsqrt_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::rsqrt)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = 1.0f / std::sqrt(x[i]);
    }
}

void tanh_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::tanh)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::tanh(x[i]);
    }
}

void sin_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::sin)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::sin(x[i]);
    }
}

void cos_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::cos)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::cos(x[i]);
    }
}

void reciprocal_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::reciprocal)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = 1.0f / x[i];
    }
}

void clamp_f32(const float* x, float* out, std::int64_t n, float lower, float upper) {
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::clamp)))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::isnan(x[i]) ? x[i] : std::fmin(std::fmax(x[i], lower), upper);
    }
}

} // namespace mensurflow::cpu
