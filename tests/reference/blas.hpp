// Naive double-precision references for BLAS ops. Slow and obviously correct; optimized kernels
// are compared against these within a tolerance, since float reductions may reorder additions.
#pragma once

#include <cmath>
#include <cstdint>

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

} // namespace reference
