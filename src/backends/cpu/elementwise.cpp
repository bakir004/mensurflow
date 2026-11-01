// Elementwise kernels: shared unary and binary loops, parallelized by parallel_for.
#include "backends/cpu/elementwise.hpp"

#include "backends/cpu/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mensurflow::cpu {

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
    parallel_for(n, [=](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            out[i] = op(x[i]);
        }
    });
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
    parallel_for(n, [=](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            out[i] = op(a[i], b[i]);
        }
    });
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

void pow_f32(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return std::pow(x, y); });
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
    parallel_for(n, [=](std::int64_t begin, std::int64_t end) {
        // A thread's block can begin inside a row. Use contiguous spans so the inner loop
        // can vectorize, rather than taking a remainder for every element.
        std::int64_t feature = begin % features;
        for (std::int64_t i = begin; i < end;) {
            const std::int64_t count = std::min(end - i, features - feature);
            for (std::int64_t j = 0; j < count; ++j) {
                out[i + j] = x[i + j] + bias[feature + j];
            }
            i += count;
            feature = 0;
        }
    });
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

void log_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return std::log(value); });
}

void rsqrt_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return 1.0f / std::sqrt(value); });
}

void tanh_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return std::tanh(value); });
}

void sin_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return std::sin(value); });
}

void cos_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return std::cos(value); });
}

void reciprocal_f32(const float* x, float* out, std::int64_t n) {
    unary(x, out, n, [](float value) { return 1.0f / value; });
}

void minimum_f32(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) {
        return std::isnan(x) ? x : (std::isnan(y) ? y : std::fmin(x, y));
    });
}

void maximum_f32(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) {
        return std::isnan(x) ? x : (std::isnan(y) ? y : std::fmax(x, y));
    });
}

void clamp_f32(const float* x, float* out, std::int64_t n, float lower, float upper) {
    unary(x, out, n, [lower, upper](float value) {
        return std::isnan(value) ? value : std::fmin(std::fmax(value, lower), upper);
    });
}

} // namespace mensurflow::cpu
