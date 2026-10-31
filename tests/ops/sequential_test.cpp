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
#include <stdexcept>
#include <string>
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
    0, 1, 15, 16, 17, 1000, cpu::default_grain, cpu::default_grain + 1, 100003, 1 << 20};

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

void test_errors() {
    float x = 1.0f;
    test::require_throws<std::invalid_argument>([&] { seq::add_f32(&x, &x, &x, -1); },
                                                "negative n");
    test::require_throws<std::invalid_argument>([&] { seq::mul_f32(nullptr, &x, &x, 1); },
                                                "null input");
    test::require_throws<std::invalid_argument>([&] { seq::sub_f32(&x, &x, nullptr, 1); },
                                                "null output");
    seq::add_f32(nullptr, nullptr, nullptr, 0); // n == 0 never touches memory
    test::require(seq::dot_f32(nullptr, nullptr, 0) == 0.0f, "empty dot is 0");
}

} // namespace

int main() {
    return test::run("sequential_test", [] {
        test_elementwise();
        test_in_place();
        test_dot();
        test_errors();
    });
}
