// AXPY correctness tests for the public API, including its legacy axpy.hpp header.
// The same cases run on CPU, Auto, and each available GPU backend.
#include "prs/axpy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
// Turn a failed check into an exception handled by main(), without a test framework.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Exercise the same input/output contract regardless of the selected hardware.
void test(prs::Backend backend) {
    // Cover empty inputs, the 256-thread CUDA block boundary, and a large workload.
    for (std::size_t n : {0u, 1u, 255u, 256u, 257u, 1000003u}) {
        std::vector<float> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 7.0f;
            y[i] = static_cast<float>(i % 19) / 3.0f;
        }
        const auto initial = y;
        prs::axpy(-2.5f, x, y, backend);
        // Compare each result against the formula, allowing float rounding differences.
        for (std::size_t i = 0; i < n; ++i) {
            const float expected = -2.5f * x[i] + initial[i];
            require(std::isfinite(y[i]) &&
                    std::abs(y[i] - expected) <= 1e-5f * std::max(1.0f, std::abs(expected)),
                    "AXPY result mismatch");
        }
    }

    // Input and output may share one vector; alpha=0 must leave its values unchanged.
    std::vector<float> same{1, -2, 3};
    prs::axpy(2, same, same, backend);
    require(same == std::vector<float>({3, -6, 9}), "Aliasing failed");
    prs::axpy(0, same, same, backend);
    require(same == std::vector<float>({3, -6, 9}), "Zero alpha failed");

    // Invalid vector lengths must fail through the public API.
    const std::vector<float> x{1};
    std::vector<float> y{1, 2};
    bool rejected = false;
    try { prs::axpy(1, x, y, backend); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Mismatched sizes were accepted");
}
} // namespace

int main() {
    try {
        // Show the simplest public call before running the reusable test cases.
        std::vector<float> x{1, 2, 3}, y{10, 20, 30};
        prs::axpy(2, x, y); // CPU by default
        std::cout << "CPU example: " << y[0] << ' ' << y[1] << ' ' << y[2] << '\n';
        test(prs::Backend::CPU);
        std::cout << "CPU tests passed\n";
        test(prs::Backend::Auto);
        std::cout << "Auto AXPY tests passed\n";
        // Optional hardware is skipped explicitly so this test also runs on CPU-only hosts.
        if (prs::metal_available()) {
            test(prs::Backend::Metal);
            std::cout << "Metal tests passed\n";
        } else {
            std::cout << "Metal tests skipped: backend or device unavailable\n";
        }
        if (prs::cuda_available()) {
            test(prs::Backend::CUDA);
            std::cout << "CUDA tests passed\n";
        } else {
            std::cout << "CUDA tests skipped: backend or device unavailable\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
