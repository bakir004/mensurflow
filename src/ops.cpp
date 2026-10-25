// Shared implementation of the public operations: validate dimensions, choose a
// backend, and handle empty or aliased outputs before invoking backend code.
#include "prs/ops.hpp"
#include "backends/interface.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace prs {
// Helpers stay local to this translation unit rather than becoming public API.
namespace {
const detail::BackendInterface& implementation(Backend backend) {
    // Resolve availability first, then return the selected adapter by reference.
    switch (resolve_backend(backend)) {
    case Backend::CPU:
        return detail::cpu_backend();
#ifdef PRS_HAS_CUDA
    case Backend::CUDA:
        return detail::cuda_backend();
#endif
#ifdef PRS_HAS_METAL
    case Backend::Metal:
        return detail::metal_backend();
#endif
#ifdef PRS_HAS_OPENCL
    case Backend::OpenCL:
        return detail::opencl_backend();
#endif
    default:
        throw std::runtime_error("Backend implementation unavailable");
    }
}

std::size_t elements(std::size_t rows, std::size_t cols) {
    // Check multiplication before it happens, including the eventual byte count.
    const auto max = std::numeric_limits<std::size_t>::max();
    if (cols && rows > max / cols) {
        throw std::overflow_error("Operation dimensions overflow");
    }
    const auto count = rows * cols;
    if (count > max / sizeof(float)) {
        throw std::overflow_error("Operation byte size overflow");
    }
    return count;
}
} // namespace

void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y, Backend backend) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("AXPY: x and y must have equal sizes");
    }
    elements(x.size(), 1);
    const auto& impl = implementation(backend); // Explicit requests never silently fall back.
    if (!x.empty()) {
        impl.axpy(alpha, x.data(), y.data(), x.size());
    }
}

void matmul(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c,
            std::size_t m, std::size_t n, std::size_t k, Backend backend) {
    const auto a_count = elements(m, k);
    const auto b_count = elements(k, n);
    const auto c_count = elements(m, n);
    if (a.size() != a_count || b.size() != b_count || c.size() != c_count) {
        throw std::invalid_argument("Matmul: vector sizes must match M*K, K*N, M*N");
    }
    const auto& impl = implementation(backend);
    // Resolve even empty operations so explicit unavailable backends still fail.
    if (c_count == 0) {
        return;
    }
    if (k == 0) {
        std::fill(c.begin(), c.end(), 0.0f);
        return;
    }
    if (&c == &a || &c == &b) {
        // Preserve inputs until every output has been computed, then replace C.
        std::vector<float> result(c_count);
        impl.matmul(a.data(), b.data(), result.data(), m, n, k);
        c.swap(result);
    } else {
        impl.matmul(a.data(), b.data(), c.data(), m, n, k);
    }
}
} // namespace prs
