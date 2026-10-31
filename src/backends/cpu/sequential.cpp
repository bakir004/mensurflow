// Sequential CPU kernels: the baseline for measuring what parallelism buys.
#include "backends/cpu/sequential.hpp"

#include <cmath>
#include <stdexcept>

namespace mensurflow::cpu::sequential {

namespace {

template <typename Op>
void unary(const float* x, float* out, std::int64_t n, Op op) {
    if (n < 0) {
        throw std::invalid_argument("elementwise: negative length");
    }
    if (n == 0) {
        return;
    }
    if (x == nullptr || out == nullptr) {
        throw std::invalid_argument("elementwise: null pointer");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = op(x[i]);
    }
}

template <typename Op>
void binary(const float* a, const float* b, float* out, std::int64_t n, Op op) {
    if (n < 0) {
        throw std::invalid_argument("elementwise: negative length");
    }
    if (n == 0) {
        return;
    }
    if (a == nullptr || b == nullptr || out == nullptr) {
        throw std::invalid_argument("elementwise: null pointer");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = op(a[i], b[i]);
    }
}

} // namespace

void add_f32(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x + y; });
}

void sub_f32(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x - y; });
}

void mul_f32(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x * y; });
}

void div_f32(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x / y; });
}

void axpy_f32(float alpha, const float* x, const float* y, float* out, std::int64_t n) {
    binary(x, y, out, n, [alpha](float a, float b) { return std::fma(alpha, a, b); });
}

void bias_add_f32(const float* x, const float* bias, float* out, std::int64_t n,
                  std::int64_t features) {
    if (n < 0 || features < 0) {
        throw std::invalid_argument("bias_add: negative dimension");
    }
    if (n == 0) {
        return;
    }
    if (features == 0 || n % features != 0) {
        throw std::invalid_argument("bias_add: features must be positive and divide length");
    }
    if (x == nullptr || bias == nullptr || out == nullptr) {
        throw std::invalid_argument("bias_add: null pointer");
    }
    if (features == 1) {
        const float value = bias[0];
        unary(x, out, n, [value](float a) { return a + value; });
        return;
    }
    for (std::int64_t row = 0; row < n; row += features) {
        for (std::int64_t feature = 0; feature < features; ++feature) {
            out[row + feature] = x[row + feature] + bias[feature];
        }
    }
}

void neg_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return -value; });
}

void abs_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return std::fabs(value); });
}

void square_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return value * value; });
}

void relu_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return value < 0.0f ? 0.0f : value; });
}

void sqrt_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return std::sqrt(value); });
}

void exp_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return std::exp(value); });
}

void sigmoid_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) {
        if (value >= 0.0f) {
            return 1.0f / (1.0f + std::exp(-value));
        }
        const float z = std::exp(value);
        return z / (1.0f + z);
    });
}

float dot_f32(const float* x, const float* y, std::int64_t n) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
    }
    return static_cast<float>(sum);
}

void matmul_f32(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                std::int64_t k) {
    if (m < 0 || n < 0 || k < 0) {
        throw std::invalid_argument("matmul: negative dimension");
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (c == nullptr || (k > 0 && (a == nullptr || b == nullptr))) {
        throw std::invalid_argument("matmul: null pointer");
    }
    for (std::int64_t i = 0; i < m; ++i) {
        float* ci = c + i * n;
        for (std::int64_t j = 0; j < n; ++j) {
            ci[j] = 0.0f;
        }
        for (std::int64_t p = 0; p < k; ++p) {
            const float aip = a[i * k + p];
            const float* bp = b + p * n;
            for (std::int64_t j = 0; j < n; ++j) {
                ci[j] += aip * bp[j];
            }
        }
    }
}

} // namespace mensurflow::cpu::sequential
