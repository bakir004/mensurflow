// Tests ger (rank-1 update): the CPU kernel against a float64 reference at several thread counts and
// sizes around the parallel cutoff, then mensurflow::ger through the public API.
#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 4;
    }
    return v;
}

// alpha * x[i] * y[j] + a[i, j], rounded to float once: with inputs that are multiples of 0.25 and
// alpha a power of two, the float kernel computes exactly this.
std::vector<float> reference(float alpha, const std::vector<float>& x, const std::vector<float>& y,
                             const std::vector<float>* a) {
    std::vector<float> out;
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (std::size_t j = 0; j < y.size(); ++j) {
            // Adding 0.0 would turn a -0 product into +0, so add nothing when there is no a.
            const double product = static_cast<double>(alpha) * x[i] * y[j];
            out.push_back(
                static_cast<float>(a == nullptr ? product : product + (*a)[i * y.size() + j]));
        }
    }
    return out;
}

void test_kernel() {
    const int original_threads = cpu::max_threads();
    // Around parallel_threshold (32768 elements): 181 x 181 is just below, 182 x 182 just above.
    const std::int64_t shapes[][2] = {{1, 1},     {1, 9},     {9, 1},      {13, 7},
                                      {181, 181}, {182, 182}, {1000, 300}, {3, 40000}};
    for (const auto& shape : shapes) {
        const std::int64_t m = shape[0], n = shape[1];
        const auto x = make_input(m, 1);
        const auto y = make_input(n, 2);
        const auto a = make_input(m * n, 3);
        for (const bool with_a : {false, true}) {
            const auto want = reference(2.0f, x, y, with_a ? &a : nullptr);
            for (int threads : {1, 2, 3, original_threads}) {
                cpu::set_max_threads(threads);
                std::vector<float> out(want.size(), 123.0f);
                cpu::ger_f32(2.0f, x.data(), y.data(), with_a ? a.data() : nullptr, out.data(), m,
                             n);
                test::require(test::matches_f32(out, want),
                              "ger " + std::to_string(m) + "x" + std::to_string(n) +
                                  (with_a ? " with a" : "") +
                                  " threads=" + std::to_string(threads));
            }
        }
        // In place: out == a.
        std::vector<float> in_place = a;
        cpu::ger_f32(2.0f, x.data(), y.data(), in_place.data(), in_place.data(), m, n);
        test::require(test::matches_f32(in_place, reference(2.0f, x, y, &a)), "ger in place");
    }
    cpu::set_max_threads(original_threads);
}

void test_api() {
    const auto x = make_input(4, 4);
    const auto y = make_input(3, 5);
    const auto a = make_input(12, 6);
    for (Device device : list_devices()) {
        const Tensor tx = Tensor::from_vector(x, {4}, device);
        const Tensor ty = Tensor::from_vector(y, {3}, device);
        const Tensor ta = Tensor::from_vector(a, {4, 3}, device);
        const Tensor out = ger(0.5f, tx, ty, &ta);
        test::require(out.shape() == Shape{4, 3}, "ger shape");
        test::require(out.device() == device, "ger result on the wrong device");
        test::require(test::matches_f32(out.to_vector<float>(), reference(0.5f, x, y, &a)),
                      "ger values on " + to_string(device));
        test::require(
            test::matches_f32(ger(1.0f, tx, ty).to_vector<float>(), reference(1.0f, x, y, nullptr)),
            "ger without a on " + to_string(device));
    }
    // A strided x (every other element) is accepted.
    const Tensor long_x = Tensor::from_vector(make_input(8, 7), {8});
    const Tensor every_other = long_x.slice(0, 0, 8, 2);
    const Tensor ty = Tensor::from_vector(y, {3});
    test::require(ger(1.0f, every_other, ty).to_vector<float>() ==
                      ger(1.0f, every_other.contiguous(), ty).to_vector<float>(),
                  "ger of a strided view");
    test::require(ger(1.0f, Tensor::ones({0}), ty).shape() == Shape{0, 3}, "ger of an empty x");

    using E = std::invalid_argument;
    test::require_throws<E>([&] { ger(1.0f, Tensor::ones({2, 2}), ty); }, "2-D x");
    const Tensor wrong_a = Tensor::ones({3, 3});
    test::require_throws<E>([&] { ger(1.0f, Tensor::ones({4}), ty, &wrong_a); }, "a shape");
    test::require_throws<E>([&] { ger(1.0f, Tensor::ones({4}, DType::I32), ty); }, "dtypes");
}

} // namespace

int main() {
    return test::run("ger_test", [] {
        test_kernel();
        test_api();
    });
}
