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

namespace {

template <typename Reduce>
void reduce_axis(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                 std::int64_t axis, Reduce reduce) {
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::int64_t i = 0; i < axis; ++i) {
        outer *= shape[i];
    }
    for (std::int64_t i = axis + 1; i < rank; ++i) {
        inner *= shape[i];
    }
    const std::int64_t width = shape[axis];
    parallel_for(outer * inner, [=](std::int64_t begin, std::int64_t end) {
        for (std::int64_t index = begin; index < end; ++index) {
            const std::int64_t row = index / inner;
            const std::int64_t column = index % inner;
            out[index] = reduce(x + row * width * inner + column, width, inner);
        }
    });
}

} // namespace

void sum_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis) {
    reduce_axis(x, out, shape, rank, axis, [](const float* start, std::int64_t width,
                                              std::int64_t stride) {
        double result = 0.0;
        for (std::int64_t i = 0; i < width; ++i) {
            result += start[i * stride];
        }
        return static_cast<float>(result);
    });
}

void mean_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                   std::int64_t axis) {
    reduce_axis(x, out, shape, rank, axis, [](const float* start, std::int64_t width,
                                               std::int64_t stride) {
        double result = 0.0;
        for (std::int64_t i = 0; i < width; ++i) {
            result += start[i * stride];
        }
        return static_cast<float>(result / static_cast<double>(width));
    });
}

void max_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis) {
    reduce_axis(x, out, shape, rank, axis, [](const float* start, std::int64_t width,
                                              std::int64_t stride) {
        float result = start[0];
        for (std::int64_t i = 1; i < width; ++i) {
            const float value = start[i * stride];
            if (std::isnan(value)) {
                return value;
            }
            if (std::isnan(result)) {
                return result;
            }
            result = std::fmax(result, value);
        }
        return result;
    });
}

} // namespace mensurflow::cpu
