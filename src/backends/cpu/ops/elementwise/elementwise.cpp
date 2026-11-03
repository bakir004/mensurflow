// CPU elementwise kernels: one loop per operation. MENSURFLOW_PARALLEL_FOR (threads.hpp) splits
// the loop across threads when the build has OpenMP; its `if` keeps small inputs on one thread.
#include "backends/cpu/ops/elementwise/elementwise.hpp"

#include "backends/cpu/threads.hpp"

#include <cmath>

namespace mensurflow::cpu {

void add_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void div_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

void pow_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::pow(a[i], b[i]);
    }
}

void minimum_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::isnan(a[i]) ? a[i] : (std::isnan(b[i]) ? b[i] : std::fmin(a[i], b[i]));
    }
}

void maximum_f32(const float* a, const float* b, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::isnan(a[i]) ? a[i] : (std::isnan(b[i]) ? b[i] : std::fmax(a[i], b[i]));
    }
}

void axpy_f32(float alpha, const float* x, const float* y, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::fma(alpha, x[i], y[i]);
    }
}

void bias_add_f32(const float* x, const float* bias, float* out, std::int64_t n,
                  std::int64_t features) {
    const std::int64_t rows = features == 0 ? 0 : n / features;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t j = 0; j < features; ++j) {
            out[row * features + j] = x[row * features + j] + bias[j];
        }
    }
}

void neg_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = -x[i];
    }
}

void abs_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::fabs(x[i]);
    }
}

void square_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = x[i] * x[i];
    }
}

void relu_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = x[i] < 0.0f ? 0.0f : x[i];
    }
}

void sqrt_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::sqrt(x[i]);
    }
}

void exp_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::exp(x[i]);
    }
}

void sigmoid_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = x[i] >= 0.0f ? 1.0f / (1.0f + std::exp(-x[i]))
                              : std::exp(x[i]) / (1.0f + std::exp(x[i]));
    }
}

void log_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::log(x[i]);
    }
}

void rsqrt_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = 1.0f / std::sqrt(x[i]);
    }
}

void tanh_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::tanh(x[i]);
    }
}

void sin_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::sin(x[i]);
    }
}

void cos_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::cos(x[i]);
    }
}

void reciprocal_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = 1.0f / x[i];
    }
}

void clamp_f32(const float* x, float* out, std::int64_t n, float lower, float upper) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = std::isnan(x[i]) ? x[i] : std::fmin(std::fmax(x[i], lower), upper);
    }
}

} // namespace mensurflow::cpu
