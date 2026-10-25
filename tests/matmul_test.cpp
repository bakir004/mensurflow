// Matrix multiplication and backend-selection tests for the public API in ops.hpp.
// Covers numerical results, aliasing, empty shapes, rejected inputs, and GPU availability.
#include "prs/ops.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
// Report a failed condition to main(), which exits with a nonzero status for CTest.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// Confirm that an invalid operation throws the expected exception type.
template <typename Exception, typename F> void rejects(F operation, const char* message) {
    try { operation(); }
    catch (const Exception&) { return; }
    throw std::runtime_error(message);
}
// Build deterministic row-major inputs for A[M,K] * B[K,N] and verify every output.
void test_case(std::size_t m, std::size_t n, std::size_t k, prs::Backend backend) {
    std::vector<float> a(m*k), b(k*n), c(m*n, 123.0f);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 7;
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 9;
    const auto initial_a = a, initial_b = b;
    prs::matmul(a, b, c, m, n, k, backend);
    // Accumulate the reference in double precision and allow float rounding in the backend.
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            double expected = 0;
            for (std::size_t p = 0; p < k; ++p)
                expected += static_cast<double>(a[row*k+p]) * b[p*n+col];
            const float value = c[row*n+col];
            require(std::isfinite(value) &&
                    std::abs(value - expected) <= 1e-4 * std::max(1.0, std::abs(expected)),
                    "Matmul result mismatch");
        }
    }
    require(a == initial_a && b == initial_b, "Matmul changed an input");
}
void test(prs::Backend backend) {
    // Empty outputs and K=0 exercise the public API's special cases.
    test_case(0, 0, 0, backend);
    test_case(0, 3, 7, backend);
    test_case(5, 0, 3, backend);
    test_case(3, 5, 0, backend);
    // Include vectors, full 16x16 GPU tiles, and shapes with incomplete edge tiles.
    test_case(1, 1, 1, backend);
    test_case(1, 19, 7, backend);
    test_case(19, 1, 7, backend);
    test_case(16, 16, 16, backend);
    test_case(17, 23, 19, backend);
    test_case(31, 33, 65, backend);
    test_case(65, 47, 257, backend);

    // The API must preserve inputs needed later when C aliases A, B, or both.
    const std::vector<float> a{1, 2, 3, 4}, b{5, 6, 7, 8};
    const std::vector<float> expected{19, 22, 43, 50};
    auto alias_a = a;
    prs::matmul(alias_a, b, alias_a, 2, 2, 2, backend);
    require(alias_a == expected, "Matmul C=A alias failed");
    auto alias_b = b;
    prs::matmul(a, alias_b, alias_b, 2, 2, 2, backend);
    require(alias_b == expected, "Matmul C=B alias failed");
    auto alias_all = a;
    prs::matmul(alias_all, alias_all, alias_all, 2, 2, 2, backend);
    require(alias_all == std::vector<float>({7,10,15,22}), "Matmul all-alias failed");

    // Reject mismatched storage without changing the caller's output buffer.
    std::vector<float> c(4, 99);
    rejects<std::invalid_argument>([&] { prs::matmul({1}, b, c, 2, 2, 2, backend); }, "Accepted bad A size");
    rejects<std::invalid_argument>([&] { prs::matmul(a, {1}, c, 2, 2, 2, backend); }, "Accepted bad B size");
    std::vector<float> small(1);
    rejects<std::invalid_argument>([&] { prs::matmul(a, b, small, 2, 2, 2, backend); }, "Accepted bad C size");
    require(c == std::vector<float>(4, 99), "Rejected matmul modified output");
    rejects<std::invalid_argument>([&] { prs::matmul({}, {}, c, 0, 0, 0, backend); }, "Accepted nonempty C for empty output");
    // Check both element-count multiplication and conversion from elements to bytes.
    const auto max = std::numeric_limits<std::size_t>::max();
    std::vector<float> empty;
    rejects<std::overflow_error>([&] { prs::matmul({}, {}, empty, max, 2, 2, backend); }, "Accepted dimension overflow");
    rejects<std::overflow_error>([&] { prs::matmul({}, {}, empty, max/sizeof(float)+1, 1, 1, backend); }, "Accepted byte overflow");
}
// Auto prefers CUDA, then Metal, then CPU; explicit unavailable backends must fail.
void test_selection() {
    const auto selected = prs::resolve_backend();
    require(selected == (prs::cuda_available() ? prs::Backend::CUDA :
            prs::metal_available() ? prs::Backend::Metal : prs::Backend::CPU), "Auto preference failed");
    require(prs::backend_available(prs::Backend::Auto), "Auto unavailable");
    const auto invalid = static_cast<prs::Backend>(99);
    rejects<std::invalid_argument>([&] { prs::resolve_backend(invalid); }, "Accepted unknown backend");
    // Even an empty operation must reject an explicitly requested unavailable device.
    for (auto backend : {prs::Backend::CUDA, prs::Backend::Metal}) {
        if (!prs::backend_available(backend)) {
            std::vector<float> empty;
            rejects<std::runtime_error>([&] { prs::matmul({}, {}, empty, 0, 0, 0, backend); }, "Unavailable backend silently accepted");
            rejects<std::runtime_error>([&] { prs::axpy(1, {}, empty, backend); }, "Unavailable AXPY silently accepted");
        }
    }
}
} // namespace

int main() {
    try {
        test_selection();
        // Reuse the same contract tests and report optional GPU skips on CPU-only hosts.
        for (auto backend : {prs::Backend::CPU, prs::Backend::CUDA, prs::Backend::Metal, prs::Backend::Auto}) {
            if (prs::backend_available(backend)) {
                test(backend);
                std::cout << prs::backend_name(backend) << " matmul tests passed\n";
            } else {
                std::cout << prs::backend_name(backend) << " matmul tests skipped: backend/device unavailable\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
