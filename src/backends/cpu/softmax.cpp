#include "backends/cpu/softmax.hpp"

#include "backends/cpu/parallel.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace mensurflow::cpu {

namespace {

void softmax_rows(const float* x, float* out, std::int64_t begin, std::int64_t end,
                  std::int64_t width) {
    for (std::int64_t row = begin; row < end; ++row) {
        const float* input = x + row * width;
        float* output = out + row * width;
        float maximum = -std::numeric_limits<float>::infinity();
        bool has_nan = false;
        std::int64_t positive_infinities = 0;
        for (std::int64_t i = 0; i < width; ++i) {
            if (std::isnan(input[i])) {
                has_nan = true;
            } else if (input[i] == std::numeric_limits<float>::infinity()) {
                ++positive_infinities;
            } else if (input[i] > maximum) {
                maximum = input[i];
            }
        }
        if (has_nan) {
            for (std::int64_t i = 0; i < width; ++i) {
                output[i] = std::numeric_limits<float>::quiet_NaN();
            }
            continue;
        }
        if (positive_infinities > 0) {
            const float probability = 1.0f / static_cast<float>(positive_infinities);
            for (std::int64_t i = 0; i < width; ++i) {
                output[i] = input[i] == std::numeric_limits<float>::infinity()
                                ? probability
                                : 0.0f;
            }
            continue;
        }
        double denominator = 0.0;
        for (std::int64_t i = 0; i < width; ++i) {
            denominator += std::exp(static_cast<double>(input[i]) - maximum);
        }
        for (std::int64_t i = 0; i < width; ++i) {
            output[i] = static_cast<float>(
                std::exp(static_cast<double>(input[i]) - maximum) / denominator);
        }
    }
}

} // namespace

void softmax_f32(const float* x, float* out, std::int64_t rows, std::int64_t width) {
    if (rows < 0 || width < 0) {
        throw std::invalid_argument("softmax: negative dimension");
    }
    if (rows == 0 || width == 0) {
        return;
    }
    if (x == nullptr || out == nullptr) {
        throw std::invalid_argument("softmax: null pointer");
    }
    const double work = static_cast<double>(rows) * static_cast<double>(width);
    const std::int64_t grain = work < 262144.0 ? rows : 0;
    parallel_for(rows, [=](std::int64_t begin, std::int64_t end) {
        softmax_rows(x, out, begin, end, width);
    }, grain, 1);
}

} // namespace mensurflow::cpu
