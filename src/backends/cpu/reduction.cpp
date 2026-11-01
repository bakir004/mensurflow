#include "backends/cpu/reduction.hpp"

#include "backends/cpu/parallel.hpp"

#include <cmath>
#include <stdexcept>

namespace mensurflow::cpu {

float sum_f32(const float* x, std::int64_t n) {
    if (n < 0) {
        throw std::invalid_argument("sum: negative length");
    }
    if (n > 0 && x == nullptr) {
        throw std::invalid_argument("sum: null pointer");
    }
    const double total = parallel_sum(n, [x](std::int64_t begin, std::int64_t end) {
        double partial = 0.0;
        for (std::int64_t i = begin; i < end; ++i) {
            partial += static_cast<double>(x[i]);
        }
        return partial;
    });
    return static_cast<float>(total);
}

float mean_f32(const float* x, std::int64_t n) {
    if (n <= 0) {
        throw std::invalid_argument(n == 0 ? "mean: empty input" : "mean: negative length");
    }
    const double total = parallel_sum(n, [x](std::int64_t begin, std::int64_t end) {
        double partial = 0.0;
        for (std::int64_t i = begin; i < end; ++i) {
            partial += static_cast<double>(x[i]);
        }
        return partial;
    });
    return static_cast<float>(total / static_cast<double>(n));
}

float max_f32(const float* x, std::int64_t n) {
    if (n <= 0) {
        throw std::invalid_argument(n == 0 ? "max: empty input" : "max: negative length");
    }
    if (x == nullptr) {
        throw std::invalid_argument("max: null pointer");
    }
    return parallel_max(n, [x](std::int64_t begin, std::int64_t end) {
        float result = x[begin];
        for (std::int64_t i = begin + 1; i < end; ++i) {
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
    });
}

} // namespace mensurflow::cpu
