// Naive double-precision references for full-tensor reductions.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace reference {

struct ReductionResult {
    double value = 0.0;
    double magnitude = 0.0;
};

inline ReductionResult sum(const float* x, std::int64_t n) {
    ReductionResult result;
    for (std::int64_t i = 0; i < n; ++i) {
        result.value += static_cast<double>(x[i]);
        result.magnitude += std::abs(static_cast<double>(x[i]));
    }
    return result;
}

inline ReductionResult mean(const float* x, std::int64_t n) {
    ReductionResult result = sum(x, n);
    result.value /= static_cast<double>(n);
    result.magnitude /= static_cast<double>(n);
    return result;
}

inline float max(const float* x, std::int64_t n) {
    float result = x[0];
    for (std::int64_t i = 1; i < n; ++i) {
        if (std::isnan(x[i])) {
            return x[i];
        }
        if (std::isnan(result)) {
            return result;
        }
        if (result == 0.0f && x[i] == 0.0f) {
            result = 0.0f;
            continue;
        }
        result = std::fmax(result, x[i]);
    }
    return result;
}

inline double tolerance(const ReductionResult& result) {
    return 2e-4 * result.magnitude + 1e-30;
}

} // namespace reference
