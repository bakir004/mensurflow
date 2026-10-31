// Naive double-precision references for BLAS ops. Slow and obviously correct; optimized kernels
// are compared against these within a tolerance, since float reductions may reorder additions.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

struct DotResult {
    double value = 0;     // sum(x[i] * y[i])
    double magnitude = 0; // sum(|x[i] * y[i]|), the scale that rounding error is relative to
};

inline DotResult dot(const float* x, const float* y, std::int64_t n) {
    DotResult r;
    for (std::int64_t i = 0; i < n; ++i) {
        const double p = static_cast<double>(x[i]) * static_cast<double>(y[i]);
        r.value += p;
        r.magnitude += std::abs(p);
    }
    return r;
}

// Float32 accumulation error grows with n, so allow a few thousand ulps of the magnitude. GPU
// reductions use trees and CPUs use partial sums; both stay well inside this.
inline double dot_tolerance(const DotResult& r) { return 2e-4 * r.magnitude + 1e-30; }

// Row-major C[m,n] = A[m,k] * B[k,n] in double. magnitude[i*n+j] is sum_p |A[i,p] * B[p,j]|, the
// scale that float rounding error in that entry is relative to (same idea as DotResult).
struct MatmulResult {
    std::vector<double> value;
    std::vector<double> magnitude;
};

inline MatmulResult matmul(const float* a, const float* b, std::int64_t m, std::int64_t n,
                           std::int64_t k) {
    MatmulResult r;
    r.value.assign(static_cast<std::size_t>(m * n), 0.0);
    r.magnitude.assign(static_cast<std::size_t>(m * n), 0.0);
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            double sum = 0.0, magnitude = 0.0;
            for (std::int64_t p = 0; p < k; ++p) {
                const double product = static_cast<double>(a[i * k + p]) * b[p * n + j];
                sum += product;
                magnitude += std::abs(product);
            }
            r.value[static_cast<std::size_t>(i * n + j)] = sum;
            r.magnitude[static_cast<std::size_t>(i * n + j)] = magnitude;
        }
    }
    return r;
}

// Float32 accumulation over k terms: allow a few thousand ulps of the entry's magnitude.
inline double matmul_tolerance(double magnitude) { return 2e-4 * magnitude + 1e-30; }

} // namespace reference
