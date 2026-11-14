// Naive double-precision references for reductions.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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

inline double tolerance(const ReductionResult& result) { return 2e-4 * result.magnitude + 1e-30; }

// Variance along `axis` of a row-major tensor, by the two-pass formula in double: the mean first,
// then the sum of squared deviations divided by (count - correction); NaN when that is <= 0.
// Returns one value per output position, in row-major order of the reduced shape.
inline std::vector<double> var_axis(const std::vector<float>& x,
                                    const std::vector<std::int64_t>& shape, int axis,
                                    std::int64_t correction) {
    std::int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d) {
        outer *= shape[static_cast<std::size_t>(d)];
    }
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[static_cast<std::size_t>(axis)];
    std::vector<double> out;
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t i = 0; i < inner; ++i) {
            auto at = [&](std::int64_t r) {
                return static_cast<double>(
                    x[static_cast<std::size_t>((o * width + r) * inner + i)]);
            };
            double sum = 0.0;
            for (std::int64_t r = 0; r < width; ++r) {
                sum += at(r);
            }
            const double mean = sum / static_cast<double>(width);
            double squares = 0.0;
            for (std::int64_t r = 0; r < width; ++r) {
                squares += (at(r) - mean) * (at(r) - mean);
            }
            out.push_back(width - correction > 0
                              ? squares / static_cast<double>(width - correction)
                              : std::numeric_limits<double>::quiet_NaN());
        }
    }
    return out;
}

// d(sum(var_axis(x) * grad)) / dx = grad * 2 * (x - mean) / (count - correction), by the two-pass
// mean in double.
inline std::vector<double> var_axis_backward(const std::vector<float>& x,
                                             const std::vector<std::int64_t>& shape, int axis,
                                             std::int64_t correction,
                                             const std::vector<float>& grad) {
    std::int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d) {
        outer *= shape[static_cast<std::size_t>(d)];
    }
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[static_cast<std::size_t>(axis)];
    std::vector<double> gx(x.size());
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t i = 0; i < inner; ++i) {
            auto index = [&](std::int64_t r) {
                return static_cast<std::size_t>((o * width + r) * inner + i);
            };
            double sum = 0.0;
            for (std::int64_t r = 0; r < width; ++r) {
                sum += x[index(r)];
            }
            const double mean = sum / static_cast<double>(width);
            const double g = grad[static_cast<std::size_t>(o * inner + i)];
            for (std::int64_t r = 0; r < width; ++r) {
                gx[index(r)] = width - correction > 0
                                   ? g * 2.0 * (x[index(r)] - mean) /
                                         static_cast<double>(width - correction)
                                   : std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    return gx;
}

// d(sum(sqrt(var_axis(x)) * grad)) / dx = grad * (x - mean) / ((count - correction) * std); 0
// where std is 0, NaN where count - correction <= 0. Two-pass mean and variance in double.
inline std::vector<double> std_axis_backward(const std::vector<float>& x,
                                             const std::vector<std::int64_t>& shape, int axis,
                                             std::int64_t correction,
                                             const std::vector<float>& grad) {
    const std::vector<double> variance = var_axis(x, shape, axis, correction);
    // var's gradient is grad * 2 (x - mean) / (count - correction); std's is that / (2 std).
    const std::vector<float> ones(grad.size(), 1.0f);
    const std::vector<double> unit = var_axis_backward(x, shape, axis, correction, ones);
    std::int64_t inner = 1;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[static_cast<std::size_t>(axis)];
    std::vector<double> gx(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::int64_t flat = static_cast<std::int64_t>(i);
        const auto out = static_cast<std::size_t>((flat / (width * inner)) * inner + flat % inner);
        const double deviation = std::sqrt(variance[out]);
        gx[i] = deviation == 0.0 ? 0.0 : grad[out] * unit[i] / (2.0 * deviation);
    }
    return gx;
}

// Index of the smallest of x[0..n): the first NaN if there is one, otherwise the first index
// holding the minimum. n > 0.
inline std::int64_t argmin(const float* x, std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        if (std::isnan(x[i])) {
            return i;
        }
    }
    std::int64_t best = 0;
    for (std::int64_t i = 1; i < n; ++i) {
        if (x[i] < x[best]) {
            best = i;
        }
    }
    return best;
}

} // namespace reference
