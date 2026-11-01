// Tests the scalar baseline (sequential.cpp compiled without auto-vectorization): it must give
// exactly the same results as the sequential kernels, since the benchmarks compare their speed and
// a baseline that computes something different would be meaningless.
#include "backends/cpu/scalar.hpp"
#include "backends/cpu/sequential.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace seq = mensurflow::cpu::sequential;
namespace scalar = mensurflow::cpu::scalar;

namespace {

using Binary = void (*)(const float*, const float*, float*, std::int64_t);
using Unary = void (*)(const float*, float*, std::int64_t);

// Values in [-4, 4] with some exact zeros, so division and square roots meet inf and NaN.
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = state % 11 == 0 ? 0.0f
                            : static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 4;
    }
    return v;
}

bool same_bits(const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() &&
           (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0);
}

void test_elementwise() {
    const struct {
        const char* name;
        Binary scalar_fn;
        Binary sequential_fn;
    } binary[] = {{"add", scalar::add_f32, seq::add_f32},
                  {"sub", scalar::sub_f32, seq::sub_f32},
                  {"mul", scalar::mul_f32, seq::mul_f32},
                  {"div", scalar::div_f32, seq::div_f32}};

    const struct {
        const char* name;
        Unary scalar_fn;
        Unary sequential_fn;
    } unary[] = {{"neg", scalar::neg_f32, seq::neg_f32},
                 {"abs", scalar::abs_f32, seq::abs_f32},
                 {"square", scalar::square_f32, seq::square_f32},
                 {"relu", scalar::relu_f32, seq::relu_f32},
                 {"sqrt", scalar::sqrt_f32, seq::sqrt_f32},
                 {"exp", scalar::exp_f32, seq::exp_f32},
                 {"sigmoid", scalar::sigmoid_f32, seq::sigmoid_f32}};

    for (std::int64_t n : {0, 1, 15, 16, 17, 1000, 100003}) {
        const auto a = make_input(n, 1);
        const auto b = make_input(n, 2);
        for (const auto& op : binary) {
            std::vector<float> want(a.size()), got(a.size());
            op.sequential_fn(a.data(), b.data(), want.data(), n);
            op.scalar_fn(a.data(), b.data(), got.data(), n);
            test::require(same_bits(want, got),
                          std::string("scalar ") + op.name + " n=" + std::to_string(n));
        }
        for (const auto& op : unary) {
            std::vector<float> want(a.size()), got(a.size());
            op.sequential_fn(a.data(), want.data(), n);
            op.scalar_fn(a.data(), got.data(), n);
            test::require(same_bits(want, got),
                          std::string("scalar ") + op.name + " n=" + std::to_string(n));
        }
    }
}

void test_axpy_bias_dot_matmul() {
    const std::int64_t n = 4096;
    const auto x = make_input(n, 3);
    const auto y = make_input(n, 4);
    std::vector<float> want(x.size()), got(x.size());

    seq::axpy_f32(1.25f, x.data(), y.data(), want.data(), n);
    scalar::axpy_f32(1.25f, x.data(), y.data(), got.data(), n);
    test::require(same_bits(want, got), "scalar axpy");

    const auto bias = make_input(64, 5);
    seq::bias_add_f32(x.data(), bias.data(), want.data(), n, 64);
    scalar::bias_add_f32(x.data(), bias.data(), got.data(), n, 64);
    test::require(same_bits(want, got), "scalar bias_add");

    const float expected_dot = seq::dot_f32(x.data(), y.data(), n);
    const float scalar_dot = scalar::dot_f32(x.data(), y.data(), n);
    test::require(std::memcmp(&expected_dot, &scalar_dot, sizeof(float)) == 0, "scalar dot");

    const std::int64_t m = 17, k = 31, cols = 23;
    const auto a = make_input(m * k, 6);
    const auto b = make_input(k * cols, 7);
    std::vector<float> c_seq(static_cast<std::size_t>(m * cols)), c_scalar(c_seq.size());
    seq::matmul_f32(a.data(), b.data(), c_seq.data(), m, cols, k);
    scalar::matmul_f32(a.data(), b.data(), c_scalar.data(), m, cols, k);
    test::require(same_bits(c_seq, c_scalar), "scalar matmul");
}

} // namespace

int main() {
    return test::run("scalar_test", [] {
        test_elementwise();
        test_axpy_bias_dot_matmul();
    });
}
