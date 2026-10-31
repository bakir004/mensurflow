// Tests the sequential CPU kernels against the float64/float references, and the parallel kernels
// against the sequential ones: the two implementations must agree, which is what makes timing
// them against each other meaningful.
#include "backends/cpu/blas.hpp"
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential.hpp"
#include "reference/blas.hpp"
#include "reference/elementwise.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace seq = mensurflow::cpu::sequential;

namespace {

using Kernel = void (*)(const float*, const float*, float*, std::int64_t);

struct Op {
    const char* name;
    Kernel sequential;
    Kernel parallel;
    Kernel reference;
};

const Op ops[] = {{"add", seq::add_f32, cpu::add_f32, reference::add},
                  {"sub", seq::sub_f32, cpu::sub_f32, reference::sub},
                  {"mul", seq::mul_f32, cpu::mul_f32, reference::mul},
                  {"div", seq::div_f32, cpu::div_f32, reference::div}};

using UnaryKernel = void (*)(const float*, float*, std::int64_t);

struct UnaryOp {
    const char* name;
    UnaryKernel sequential;
    UnaryKernel parallel;
    UnaryKernel reference;
    float reference_tolerance = 0.0f;
};

const UnaryOp unary_ops[] = {{"neg", seq::neg_f32, cpu::neg_f32, reference::neg},
                             {"abs", seq::abs_f32, cpu::abs_f32, reference::abs},
                             {"square", seq::square_f32, cpu::square_f32, reference::square},
                             {"relu", seq::relu_f32, cpu::relu_f32, reference::relu},
                             {"sqrt", seq::sqrt_f32, cpu::sqrt_f32, reference::sqrt, 4e-7f},
                             {"exp", seq::exp_f32, cpu::exp_f32, reference::exp, 4e-7f},
                             {"sigmoid", seq::sigmoid_f32, cpu::sigmoid_f32, reference::sigmoid,
                              4e-7f}};

// Multiples of 0.25 in [-4, 4], with zeros so division produces inf and NaN.
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 4;
    }
    return v;
}

bool same_bits(const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() &&
           (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0);
}

// Sizes around the parallel cutoff and the 16-float block edge.
const std::int64_t sizes[] = {
    0, 1, 15, 16, 17, 1000, cpu::default_grain - 1, cpu::default_grain,
    cpu::default_grain + 1, 75000, 100003, (1 << 20) + 7};

void test_elementwise() {
    const int original_threads = cpu::max_threads();
    for (std::int64_t n : sizes) {
        const auto a = make_input(n, 1);
        const auto b = make_input(n, 2);
        for (const Op& op : ops) {
            std::vector<float> want(a.size()), seq_out(a.size());
            op.reference(a.data(), b.data(), want.data(), n);
            op.sequential(a.data(), b.data(), seq_out.data(), n);
            test::require(same_bits(want, seq_out), std::string("sequential ") + op.name +
                                                        " vs reference, n=" + std::to_string(n));
            for (int threads : {1, 2, 3, original_threads}) {
                cpu::set_max_threads(threads);
                std::vector<float> par_out(a.size());
                op.parallel(a.data(), b.data(), par_out.data(), n);
                test::require(same_bits(seq_out, par_out),
                              std::string("parallel ") + op.name + " vs sequential, n=" +
                                  std::to_string(n) + ", threads=" + std::to_string(threads));
            }
        }
        for (const UnaryOp& op : unary_ops) {
            std::vector<float> want(a.size()), seq_out(a.size());
            op.reference(a.data(), want.data(), n);
            op.sequential(a.data(), seq_out.data(), n);
            test::require(test::matches_f32(seq_out, want, op.reference_tolerance),
                          std::string("sequential ") + op.name + " vs float64 reference");
            auto in_place = a;
            op.sequential(in_place.data(), in_place.data(), n);
            test::require(same_bits(seq_out, in_place),
                          std::string(op.name) + " sequential in place");
            for (int threads : {1, 2, 3, original_threads}) {
                cpu::set_max_threads(threads);
                std::vector<float> par_out(a.size());
                op.parallel(a.data(), par_out.data(), n);
                const std::string where = std::string(op.name) + " n=" + std::to_string(n) +
                                          " threads=" + std::to_string(threads);
                test::require(same_bits(seq_out, par_out), where + ": parallel vs sequential");
                in_place = a;
                op.parallel(in_place.data(), in_place.data(), n);
                test::require(same_bits(seq_out, in_place), where + ": parallel in place");
            }
        }
    }
    cpu::set_max_threads(original_threads);
}

void test_in_place() {
    const std::int64_t n = 100003;
    for (const Op& op : ops) {
        const auto a = make_input(n, 3);
        const auto b = make_input(n, 4);
        std::vector<float> want(a.size());
        op.reference(a.data(), b.data(), want.data(), n);
        auto into_a = a;
        op.sequential(into_a.data(), b.data(), into_a.data(), n);
        test::require(same_bits(want, into_a), std::string("sequential ") + op.name + " into a");
        auto into_b = b;
        op.sequential(a.data(), into_b.data(), into_b.data(), n);
        test::require(same_bits(want, into_b), std::string("sequential ") + op.name + " into b");
    }
}

void test_axpy_and_bias() {
    const int original_threads = cpu::max_threads();
    for (std::int64_t n : sizes) {
        const auto x = make_input(n, 13), y = make_input(n, 14);
        for (float alpha : {0.0f, -1.25f, 0.75f, 2.0f}) {
            std::vector<float> want(x.size()), baseline(x.size()), actual(x.size());
            reference::axpy(alpha, x.data(), y.data(), want.data(), n);
            seq::axpy_f32(alpha, x.data(), y.data(), baseline.data(), n);
            test::require(same_bits(want, baseline), "sequential axpy vs float64 reference");
            auto into_x = x, into_y = y;
            seq::axpy_f32(alpha, into_x.data(), y.data(), into_x.data(), n);
            seq::axpy_f32(alpha, x.data(), into_y.data(), into_y.data(), n);
            test::require(same_bits(want, into_x) && same_bits(want, into_y),
                          "sequential axpy permits either input as output");
            for (int threads : {1, 2, 3, original_threads}) {
                cpu::set_max_threads(threads);
                cpu::axpy_f32(alpha, x.data(), y.data(), actual.data(), n);
                test::require(same_bits(baseline, actual), "axpy parallel vs sequential");
                into_x = x;
                into_y = y;
                cpu::axpy_f32(alpha, into_x.data(), y.data(), into_x.data(), n);
                cpu::axpy_f32(alpha, x.data(), into_y.data(), into_y.data(), n);
                test::require(same_bits(baseline, into_x) && same_bits(baseline, into_y),
                              "parallel axpy permits either input as output");
            }
        }
    }
    // Odd feature counts make parallel blocks begin in the middle of a row.
    for (const auto shape : {std::pair<std::int64_t, std::int64_t>{0, 0}, {0, 7}, {1, 1},
                             {cpu::default_grain - 1, 7}, {cpu::default_grain, 128},
                             {cpu::default_grain + 1, 3}, {cpu::default_grain + 1, 1},
                             {193 * 257, 257}}) {
        const auto n = shape.first, features = shape.second;
        const auto x = make_input(n, 15), bias = make_input(features, 16);
        std::vector<float> want(x.size()), baseline(x.size()), actual(x.size());
        reference::bias_add(x.data(), bias.data(), want.data(), n, features);
        seq::bias_add_f32(x.data(), bias.data(), baseline.data(), n, features);
        test::require(same_bits(want, baseline), "sequential bias_add vs float64 reference");
        auto in_place = x;
        seq::bias_add_f32(in_place.data(), bias.data(), in_place.data(), n, features);
        test::require(same_bits(want, in_place), "sequential bias_add in place");
        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            cpu::bias_add_f32(x.data(), bias.data(), actual.data(), n, features);
            test::require(same_bits(baseline, actual), "bias_add parallel vs sequential");
            in_place = x;
            cpu::bias_add_f32(in_place.data(), bias.data(), in_place.data(), n, features);
            test::require(same_bits(baseline, in_place), "parallel bias_add in place");
        }
    }
    cpu::set_max_threads(original_threads);
}

void test_axpy_special_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float large = std::numeric_limits<float>::max();
    const std::vector<float> pattern{0, -0.0f, 1.1f, -2, large, -large, inf, -inf, nan};
    const auto n = cpu::default_grain + 17;
    std::vector<float> x(static_cast<std::size_t>(n)), y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = pattern[i % pattern.size()];
        y[i] = pattern[(i + 3) % pattern.size()];
    }
    const int original_threads = cpu::max_threads();
    for (float alpha : {0.0f, -1.25f, 2.0f, inf, nan}) {
        std::vector<float> want(x.size()), baseline(x.size()), actual(x.size());
        reference::axpy(alpha, x.data(), y.data(), want.data(), n);
        seq::axpy_f32(alpha, x.data(), y.data(), baseline.data(), n);
        for (std::size_t i = 0; i < x.size(); ++i) {
            test::require(std::isnan(want[i]) ? std::isnan(baseline[i])
                                             : test::close_f32(baseline[i], want[i]),
                          "axpy float64 reference for special values and coefficients");
        }
        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            cpu::axpy_f32(alpha, x.data(), y.data(), actual.data(), n);
            test::require(same_bits(baseline, actual), "axpy exact special-value agreement");
        }
    }
    cpu::set_max_threads(original_threads);
}

void test_dot() {
    const int original_threads = cpu::max_threads();
    for (std::int64_t n : sizes) {
        const auto x = make_input(n, 5);
        const auto y = make_input(n, 6);
        const auto want = reference::dot(x.data(), y.data(), n);
        const float s = seq::dot_f32(x.data(), y.data(), n);
        test::require(std::abs(static_cast<double>(s) - want.value) <=
                          reference::dot_tolerance(want),
                      "sequential dot vs float64 reference, n=" + std::to_string(n));
        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            const float p = cpu::dot_f32(x.data(), y.data(), n);
            if (threads == 1) {
                // One thread adds in the same order, so the result must be identical.
                test::require(std::memcmp(&s, &p, sizeof(float)) == 0,
                              "parallel dot with 1 thread equals sequential, n=" +
                                  std::to_string(n));
            }
            test::require(std::abs(static_cast<double>(p) - want.value) <=
                              reference::dot_tolerance(want),
                          "parallel dot vs float64 reference, n=" + std::to_string(n) +
                              ", threads=" + std::to_string(threads));
        }
    }
    cpu::set_max_threads(original_threads);
}

std::uint32_t bits(float value) {
    std::uint32_t result;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void test_unary_special_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float large = std::numeric_limits<float>::max();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> pattern{0.0f, -0.0f, inf, -inf, tiny, -tiny, large, -large,
                                     1.1f, -1.1f, std::nextafter(1.0f, 2.0f), nan, -nan,
                                     -1000.0f, -104.0f, -100.0f, -90.0f, -20.0f, -2.0f,
                                     2.0f, 20.0f, 88.0f, 88.7f, 89.0f, 100.0f, 1000.0f};
    std::vector<float> input(static_cast<std::size_t>(cpu::default_grain + 17));
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = pattern[i % pattern.size()];
    }
    const int original_threads = cpu::max_threads();
    const auto n = static_cast<std::int64_t>(input.size());
    for (const UnaryOp& op : unary_ops) {
        const std::string name = op.name;
        std::vector<float> expected(input.size()), sequential(input.size());
        op.reference(input.data(), expected.data(), n);
        op.sequential(input.data(), sequential.data(), n);
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (std::isnan(expected[i])) {
                test::require(std::isnan(sequential[i]), name + " propagates NaN");
            } else {
                test::require(test::close_f32(sequential[i], expected[i], op.reference_tolerance),
                              name + " matches float64 reference for special and rounding cases");
            }
            if (name == "neg") {
                test::require(bits(sequential[i]) == (bits(input[i]) ^ 0x80000000u),
                              "neg flips only the sign bit");
            } else if (name == "abs") {
                test::require(bits(sequential[i]) == (bits(input[i]) & 0x7fffffffu),
                              "abs clears only the sign bit, including zeros and NaNs");
            } else if (name == "relu" && (std::isnan(input[i]) || input[i] == 0.0f)) {
                test::require(bits(sequential[i]) == bits(input[i]),
                              "relu preserves NaNs and signed zeros");
            }
        }
        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<float> parallel(input.size());
            op.parallel(input.data(), parallel.data(), n);
            test::require(same_bits(sequential, parallel), name + " exact special-value results");
            auto in_place = input;
            op.parallel(in_place.data(), in_place.data(), n);
            test::require(same_bits(sequential, in_place), name + " special values in place");
        }
    }
    cpu::set_max_threads(original_threads);
}

void test_errors() {
    float x = 1.0f;
    test::require_throws<std::invalid_argument>([&] { seq::add_f32(&x, &x, &x, -1); },
                                                "negative n");
    test::require_throws<std::invalid_argument>([&] { seq::mul_f32(nullptr, &x, &x, 1); },
                                                "null input");
    test::require_throws<std::invalid_argument>([&] { seq::sub_f32(&x, &x, nullptr, 1); },
                                                "null output");
    seq::axpy_f32(2, nullptr, nullptr, nullptr, 0);
    seq::bias_add_f32(nullptr, nullptr, nullptr, 0, 0);
    test::require_throws<std::invalid_argument>([&] { seq::axpy_f32(2, &x, &x, &x, -1); },
                                                "sequential axpy negative length");
    test::require_throws<std::invalid_argument>([&] { seq::axpy_f32(2, nullptr, &x, &x, 1); },
                                                "sequential axpy null input");
    test::require_throws<std::invalid_argument>([&] { seq::axpy_f32(2, &x, &x, nullptr, 1); },
                                                "sequential axpy null output");
    for (const auto dimensions : {std::pair<std::int64_t, std::int64_t>{-1, 1}, {0, -1},
                                  {1, 0}, {2, 3}}) {
        test::require_throws<std::invalid_argument>(
            [&] { seq::bias_add_f32(&x, &x, &x, dimensions.first, dimensions.second); },
            "sequential bias_add invalid dimensions");
    }
    test::require_throws<std::invalid_argument>(
        [&] { seq::bias_add_f32(nullptr, &x, &x, 1, 1); }, "sequential bias_add null input");
    test::require_throws<std::invalid_argument>(
        [&] { seq::bias_add_f32(&x, nullptr, &x, 1, 1); }, "sequential bias_add null bias");
    test::require_throws<std::invalid_argument>(
        [&] { seq::bias_add_f32(&x, &x, nullptr, 1, 1); }, "sequential bias_add null output");
    seq::add_f32(nullptr, nullptr, nullptr, 0); // n == 0 never touches memory
    test::require(seq::dot_f32(nullptr, nullptr, 0) == 0.0f, "empty dot is 0");
    for (const UnaryOp& op : unary_ops) {
        op.sequential(nullptr, nullptr, 0);
        test::require_throws<std::invalid_argument>([&] { op.sequential(&x, &x, -1); },
                                                    "negative unary length");
        test::require_throws<std::invalid_argument>([&] { op.sequential(nullptr, &x, 1); },
                                                    "null unary input");
        test::require_throws<std::invalid_argument>([&] { op.sequential(&x, nullptr, 1); },
                                                    "null unary output");
    }
}

} // namespace

int main() {
    return test::run("sequential_test", [] {
        test_elementwise();
        test_in_place();
        test_dot();
        test_axpy_and_bias();
        test_axpy_special_values();
        test_unary_special_values();
        test_errors();
    });
}
