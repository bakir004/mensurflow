#include "backends/cpu/argmax.hpp"

#include "backends/cpu/parallel.hpp"

#include <cmath>
#include <stdexcept>

namespace mensurflow::cpu {

namespace {

void argmax_rows(const float* x, std::int64_t* out, std::int64_t begin, std::int64_t end,
                 std::int64_t width) {
    for (std::int64_t row = begin; row < end; ++row) {
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

} // namespace

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
    const double work = static_cast<double>(rows) * static_cast<double>(width);
    const std::int64_t grain = work < 262144.0 ? rows : 0;
    parallel_for(rows, [=](std::int64_t begin, std::int64_t end) {
        argmax_rows(x, out, begin, end, width);
    }, grain, 1);
}

} // namespace mensurflow::cpu
