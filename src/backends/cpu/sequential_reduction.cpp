#include "backends/cpu/sequential_reduction.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mensurflow::cpu::sequential {

namespace {

void validate(const float* x, std::int64_t n, const char* operation, bool allow_empty) {
    if (n < 0) {
        throw std::invalid_argument(std::string(operation) + ": negative length");
    }
    if (!allow_empty && n == 0) {
        throw std::invalid_argument(std::string(operation) + ": empty input");
    }
    if (n > 0 && x == nullptr) {
        throw std::invalid_argument(std::string(operation) + ": null pointer");
    }
}

} // namespace

float sum_f32(const float* x, std::int64_t n) {
    validate(x, n, "sum", true);
    double total = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        total += static_cast<double>(x[i]);
    }
    return static_cast<float>(total);
}

float mean_f32(const float* x, std::int64_t n) {
    validate(x, n, "mean", false);
    double total = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        total += static_cast<double>(x[i]);
    }
    return static_cast<float>(total / static_cast<double>(n));
}

float max_f32(const float* x, std::int64_t n) {
    validate(x, n, "max", false);
    float result = x[0];
    for (std::int64_t i = 1; i < n; ++i) {
        if (std::isnan(x[i])) {
            return x[i];
        }
        if (std::isnan(result)) {
            return result;
        }
        result = std::fmax(result, x[i]);
    }
    return result;
}

} // namespace mensurflow::cpu::sequential
