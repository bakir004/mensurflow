// Tests mensurflow::dot on every device this build can use: results against a float64
// reference, views, empty input, and error handling. On a machine with a GPU build this checks
// the vendor path; on others it checks the CPU path, with the same assertions.
#include "mensurflow/mensurflow.hpp"
#include "reference/blas.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;

namespace {

// Deterministic values in roughly [-2, 2].
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 8) % 4001 - 2000) / 1000;
    }
    return v;
}

void check_close(float got, const reference::DotResult& want, const std::string& what) {
    const double error = std::abs(static_cast<double>(got) - want.value);
    test::require(error <= reference::dot_tolerance(want),
                  what + ": got " + std::to_string(got) + ", want " + std::to_string(want.value));
}

void test_matches_reference(Device device) {
    const std::string where = " on " + to_string(device);
    for (std::int64_t n : {1, 2, 15, 16, 17, 1000, 32768, 32769, 100003, (1 << 20) + 7}) {
        const auto a = make_input(n, 1);
        const auto b = make_input(n, 2);
        const Tensor result =
            dot(Tensor::from_vector(a, {n}, device), Tensor::from_vector(b, {n}, device));
        test::require(result.dim() == 0 && result.dtype() == DType::F32, "scalar f32 result");
        test::require(result.device() == device, "result stays on the input device");
        check_close(result.item<float>(), reference::dot(a.data(), b.data(), n),
                    "n=" + std::to_string(n) + where);
    }
}

void test_known_values(Device device) {
    const Tensor a = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device);
    const Tensor b = Tensor::from_vector(std::vector<float>{4, -5, 6}, {3}, device);
    test::require(dot(a, b).item<float>() == 12.0f, "1*4 + 2*-5 + 3*6 = 12");
    test::require(dot(a, a).item<float>() == 14.0f, "a . a");
}

void test_empty(Device device) {
    const Tensor none = Tensor::empty({0}, DType::F32, device);
    const Tensor result = dot(none, none);
    test::require(result.dim() == 0 && result.item<float>() == 0.0f, "empty dot is 0");
}

void test_views(Device device) {
    const std::int64_t n = 1001;
    const auto a = make_input(n, 3);
    const auto b = make_input(n, 4);
    const Tensor ta = Tensor::from_vector(a, {n}, device);
    const Tensor tb = Tensor::from_vector(b, {n}, device);

    // Every other element: strided input, made contiguous before the kernel runs.
    std::vector<float> ea, eb;
    for (std::int64_t i = 0; i < n; i += 2) {
        ea.push_back(a[static_cast<std::size_t>(i)]);
        eb.push_back(b[static_cast<std::size_t>(i)]);
    }
    check_close(dot(ta.slice(0, 0, n, 2), tb.slice(0, 0, n, 2)).item<float>(),
                reference::dot(ea.data(), eb.data(), static_cast<std::int64_t>(ea.size())),
                "strided slice");

    // A contiguous slice that starts partway into the allocation (non-zero offset).
    check_close(dot(ta.slice(0, 100, 900), tb.slice(0, 100, 900)).item<float>(),
                reference::dot(a.data() + 100, b.data() + 100, 800), "offset slice");

    // A broadcast scalar (stride 0): sum of 3 * b[i].
    const Tensor three = Tensor::from_vector(std::vector<float>{3}, {1}, device).expand({n});
    double sum_b = 0;
    for (float x : b) {
        sum_b += x;
    }
    reference::DotResult want;
    want.value = 3 * sum_b;
    for (float x : b) {
        want.magnitude += std::abs(3.0 * x);
    }
    check_close(dot(three, tb).item<float>(), want, "expanded scalar");
}

void test_errors(Device device) {
    const Tensor v3 = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device);
    const Tensor v4 = Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {4}, device);
    const Tensor m = Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {2, 2}, device);
    const Tensor ints = Tensor::from_vector(std::vector<std::int32_t>{1, 2, 3}, {3}, device);

    test::require_throws<std::invalid_argument>([&] { dot(v3, v4); }, "length mismatch");
    test::require_throws<std::invalid_argument>([&] { dot(m, m); }, "2-D input");
    test::require_throws<std::invalid_argument>([&] { dot(Tensor(), v3); }, "undefined input");
    test::require_throws<std::invalid_argument>([&] { dot(v3, ints); }, "dtype mismatch");
    // No int32 kernel exists on any backend yet; the failure must name the op, not crash.
    test::require_throws<std::runtime_error>([&] { dot(ints, ints); }, "unsupported dtype");
}

void test_mixed_devices() {
    const auto devices = list_devices();
    if (devices.size() < 2) {
        return; // Needs a GPU build.
    }
    const Tensor on_cpu = Tensor::from_vector(std::vector<float>{1, 2}, {2}, devices[0]);
    const Tensor elsewhere = on_cpu.to(devices[1]);
    test::require_throws<std::invalid_argument>([&] { dot(on_cpu, elsewhere); }, "mixed devices");
    test::require(dot(elsewhere, elsewhere).item<float>() == 5.0f, "moved tensor works");
}

} // namespace

int main() {
    return test::run("dot_test", [] {
        const auto devices = list_devices();
        test::require(!devices.empty(), "at least the CPU is available");
        for (Device device : devices) {
            std::cout << "  dot on " << to_string(device) << '\n';
            test_matches_reference(device);
            test_known_values(device);
            test_empty(device);
            test_views(device);
            test_errors(device);
        }
        test_mixed_devices();
    });
}
