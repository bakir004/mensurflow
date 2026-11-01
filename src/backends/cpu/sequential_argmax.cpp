#include "backends/cpu/sequential_argmax.hpp"

#include <cmath>
#include <stdexcept>

namespace mensurflow::cpu::sequential {

void argmax_f32(const float* x, std::int64_t* out, std::int64_t rows, std::int64_t width) {
    if (rows < 0 || width < 0) {
        throw std::invalid_argument("argmax: negative dimension");
    }
    if (rows == 0) {
        return;
    }
    if (width == 0) {
        throw std::invalid_argument("argmax: empty axis");
    }
    if (x == nullptr || out == nullptr) {
        throw std::invalid_argument("argmax: null pointer");
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* input = x + row * width;
        std::int64_t best = 0;
        if (std::isnan(input[0])) {
            out[row] = 0;
            continue;
        }
        for (std::int64_t i = 1; i < width; ++i) {
            if (std::isnan(input[i])) {
                best = i;
                break;
            }
            if (input[i] > input[best]) {
                best = i;
            }
        }
        out[row] = best;
    }
}

} // namespace mensurflow::cpu::sequential
