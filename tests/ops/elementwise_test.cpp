// Tests CPU elementwise kernels against the reference loops, across sizes around the parallel
// cutoff and several thread counts. Results must match bit for bit.
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/parallel.hpp"
#include "reference/elementwise.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
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
    const std::int64_t sizes[] = {
        0, 1, 15, 16, 17, 1000, default_grain, default_grain + 1, 100003, (1 << 20) + 7};
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
}

void test_errors() {
    float x = 1.0f;
    test::require_throws<std::invalid_argument>([&] { add_f32(&x, &x, &x, -1); }, "negative n");
    test::require_throws<std::invalid_argument>([&] { add_f32(nullptr, &x, &x, 1); }, "null input");
    test::require_throws<std::invalid_argument>([&] { mul_f32(&x, &x, nullptr, 1); },
                                                "null output");
    add_f32(nullptr, nullptr, nullptr, 0); // n == 0 never touches memory
}

} // namespace

int main() {
    return test::run("elementwise_test", [] {
        test_matches_reference();
        test_in_place();
        test_errors();
    });
}
