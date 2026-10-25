#include "prs/ops.hpp"
#include "backends/interface.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace prs {
namespace {
const detail::BackendInterface& implementation(Backend backend) {
    switch (resolve_backend(backend)) {
    case Backend::CPU: return detail::cpu_backend();
#ifdef PRS_HAS_CUDA
    case Backend::CUDA: return detail::cuda_backend();
#endif
#ifdef PRS_HAS_METAL
    case Backend::Metal: return detail::metal_backend();
#endif
    default: throw std::runtime_error("Backend implementation unavailable");
    }
}
std::size_t elements(std::size_t rows, std::size_t cols) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (cols && rows > max / cols) throw std::overflow_error("Operation dimensions overflow");
    const auto count = rows * cols;
    if (count > max / sizeof(float)) throw std::overflow_error("Operation byte size overflow");
    return count;
}
} // namespace
void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y, Backend backend) {
    if (x.size() != y.size()) throw std::invalid_argument("AXPY: x and y must have equal sizes");
    elements(x.size(), 1);
    const auto& impl = implementation(backend); // Explicit requests never silently fall back.
    if (!x.empty()) impl.axpy(alpha, x.data(), y.data(), x.size());
}
void matmul(const std::vector<float>& a, const std::vector<float>& b,
            std::vector<float>& c, std::size_t m, std::size_t n, std::size_t k,
            Backend backend) {
    const auto ac = elements(m, k), bc = elements(k, n), cc = elements(m, n);
    if (a.size() != ac || b.size() != bc || c.size() != cc) {
        throw std::invalid_argument("Matmul: vector sizes must match M*K, K*N, M*N");
    }
    const auto& impl = implementation(backend);
    if (!cc) return;
    if (!k) { std::fill(c.begin(), c.end(), 0.0f); return; }
    if (&c == &a || &c == &b) {
        std::vector<float> result(cc);
        impl.matmul(a.data(), b.data(), result.data(), m, n, k);
        c.swap(result);
    } else {
        impl.matmul(a.data(), b.data(), c.data(), m, n, k);
    }
}
} // namespace prs
