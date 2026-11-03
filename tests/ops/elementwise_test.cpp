// Tests the CPU elementwise kernels against the reference loops, across sizes around
// the parallel cutoff and several thread counts. Transcendental references allow float32 rounding;
// exact sequential/parallel agreement is checked in sequential_test.cpp.
#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/threads.hpp"
#include "reference/elementwise.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace mensurflow::cpu;

namespace {

using Kernel = void (*)(const float*, const float*, float*, std::int64_t);

struct Op {
    const char* name;
    Kernel kernel;
    Kernel reference;
};

const Op ops[] = {{"add", add_f32, reference::add},
                  {"sub", sub_f32, reference::sub},
                  {"mul", mul_f32, reference::mul},
                  {"div", div_f32, reference::div}};

using UnaryKernel = void (*)(const float*, float*, std::int64_t);

struct UnaryOp {
    const char* name;
    UnaryKernel kernel;
    UnaryKernel reference;
    float reference_tolerance = 0.0f;
};

const UnaryOp unary_ops[] = {{"neg", neg_f32, reference::neg},
                             {"abs", abs_f32, reference::abs},
                             {"square", square_f32, reference::square},
                             {"relu", relu_f32, reference::relu},
                             {"sqrt", sqrt_f32, reference::sqrt, 4e-7f},
                             {"exp", exp_f32, reference::exp, 4e-7f},
                             {"sigmoid", sigmoid_f32, reference::sigmoid, 4e-7f}};

// Deterministic values in roughly [-4, 4], with some exact zeros so div hits inf and NaN.
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (std::int64_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        const float value = static_cast<float>(static_cast<int>(state >> 8) % 2001 - 1000) / 250;
        v[static_cast<std::size_t>(i)] = state % 11 == 0 ? 0.0f : value;
    }
    return v;
}

bool same_bits(const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() &&
           (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0);
}

void test_matches_reference() {
    const int original_threads = max_threads();
    // Sizes straddle the cutoff and the 16-float block edge so uneven chunks are exercised.
    const std::int64_t sizes[] = {0,
                                  1,
                                  15,
                                  16,
                                  17,
                                  1000,
                                  parallel_threshold - 1,
                                  parallel_threshold,
                                  parallel_threshold + 1,
                                  75000,
                                  100003,
                                  (1 << 20) + 7};
    for (int threads : {1, 2, 3, original_threads}) {
        set_max_threads(threads);
        for (std::int64_t n : sizes) {
            const auto a = make_input(n, 1);
            const auto b = make_input(n, 2);
            for (const Op& op : ops) {
                std::vector<float> expected(a.size()), actual(a.size());
                op.reference(a.data(), b.data(), expected.data(), n);
                op.kernel(a.data(), b.data(), actual.data(), n);
                test::require(same_bits(expected, actual),
                              std::string(op.name) + " n=" + std::to_string(n) +
                                  " threads=" + std::to_string(threads));
            }
            for (const UnaryOp& op : unary_ops) {
                std::vector<float> expected(a.size()), actual(a.size());
                op.reference(a.data(), expected.data(), n);
                op.kernel(a.data(), actual.data(), n);
                test::require(test::matches_f32(actual, expected, op.reference_tolerance),
                              std::string(op.name) + " n=" + std::to_string(n) +
                                  " threads=" + std::to_string(threads));
            }
        }
    }
    set_max_threads(original_threads);
}

void test_in_place() {
    const std::int64_t n = 100003;
    for (const Op& op : ops) {
        const auto a = make_input(n, 3);
        const auto b = make_input(n, 4);
        std::vector<float> expected(a.size());
        op.reference(a.data(), b.data(), expected.data(), n);

        auto into_a = a;
        op.kernel(into_a.data(), b.data(), into_a.data(), n);
        test::require(same_bits(expected, into_a), std::string(op.name) + " out aliases a");

        auto into_b = b;
        op.kernel(a.data(), into_b.data(), into_b.data(), n);
        test::require(same_bits(expected, into_b), std::string(op.name) + " out aliases b");
    }
    for (const UnaryOp& op : unary_ops) {
        const auto input = make_input(n, 3);
        std::vector<float> expected(input.size());
        op.reference(input.data(), expected.data(), n);
        auto actual = input;
        op.kernel(actual.data(), actual.data(), n);
        test::require(test::matches_f32(actual, expected, op.reference_tolerance),
                      std::string(op.name) + " out aliases input");
    }
}

void test_axpy_and_bias_contracts() {
    float x = 1.0f, y = 2.0f, out = 0.0f;
    // n == 0 never touches memory.
    axpy_f32(2.0f, nullptr, nullptr, nullptr, 0);
    bias_add_f32(nullptr, nullptr, nullptr, 0, 0);
    bias_add_f32(nullptr, nullptr, nullptr, 0, 7);

    // Fused multiply-add retains the product's low bits and avoids intermediate overflow.
    x = 0x1.fffffcp-1f;
    y = -1.0f;
    axpy_f32(0x1.000002p0f, &x, &y, &out, 1);
    test::require(out == -0x1p-46f, "axpy uses one fused rounding");
    x = std::numeric_limits<float>::max();
    y = -x;
    axpy_f32(2.0f, &x, &y, &out, 1);
    test::require(out == x, "axpy avoids intermediate product overflow");
}

// Kernels do not validate (src/ops does); they must only leave memory alone when n == 0.
void test_empty() {
    add_f32(nullptr, nullptr, nullptr, 0);
    for (const UnaryOp& op : unary_ops) {
        op.kernel(nullptr, nullptr, 0);
    }
}

} // namespace

int main() {
    return test::run("elementwise_test", [] {
        test_matches_reference();
        test_in_place();
        test_axpy_and_bias_contracts();
        test_empty();
    });
}
