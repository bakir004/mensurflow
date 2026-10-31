// Elementwise kernels: shared unary and binary loops, parallelized by parallel_for.
#include "backends/cpu/elementwise.hpp"

#include "backends/cpu/parallel.hpp"

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

} // namespace mensurflow::cpu
