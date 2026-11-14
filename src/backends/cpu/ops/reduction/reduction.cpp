// CPU reduction kernels: one loop per operation. MENSURFLOW_PARALLEL_FOR (threads.hpp) splits the
// loop across threads when the build has OpenMP; sums and products combine the threads' partial
// results with OpenMP's `reduction` clause. max and min need NaN and signed-zero rules a reduction
// clause does not have, so they have two versions: a plain loop, and a parallel one that takes the
// maximum of fixed-size chunks in parallel, then of the chunks.
#include "backends/cpu/ops/reduction/reduction.hpp"

#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mensurflow::cpu {

float sum_f32(const float* x, std::int64_t n) {
    double sum = 0.0;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold) reduction(+ : sum))
    for (std::int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]);
    }
    return static_cast<float>(sum);
}

float mean_f32(const float* x, std::int64_t n) {
    double sum = 0.0;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold) reduction(+ : sum))
    for (std::int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]);
    }
    return static_cast<float>(sum / static_cast<double>(n));
}

float max_f32_sequential(const float* x, std::int64_t n) {
    float result = x[0];
    for (std::int64_t i = 1; i < n; ++i) {
        if (std::isnan(result)) {
            break;
        }
        if (std::isnan(x[i])) {
            result = x[i];
            break;
        }
        if (result == 0.0f && x[i] == 0.0f) {
            result = 0.0f; // +0 beats -0
            continue;
        }
        result = std::fmax(result, x[i]);
    }
    return result;
}

float max_f32_parallel(const float* x, std::int64_t n) {
    const std::int64_t chunks = (n + parallel_threshold - 1) / parallel_threshold;
    std::vector<float> partial(static_cast<std::size_t>(chunks));
    MENSURFLOW_PARALLEL_FOR(if (chunks > 1))
    for (std::int64_t c = 0; c < chunks; ++c) {
        const std::int64_t begin = c * parallel_threshold;
        const std::int64_t end = std::min(begin + parallel_threshold, n);
        float result = x[begin];
        for (std::int64_t i = begin + 1; i < end; ++i) {
            if (std::isnan(result)) {
                break;
            }
            if (std::isnan(x[i])) {
                result = x[i];
                break;
            }
            if (result == 0.0f && x[i] == 0.0f) {
                result = 0.0f;
                continue;
            }
            result = std::fmax(result, x[i]);
        }
        partial[static_cast<std::size_t>(c)] = result;
    }
    float result = partial[0];
    for (std::size_t c = 1; c < partial.size(); ++c) {
        if (std::isnan(result)) {
            break;
        }
        if (std::isnan(partial[c])) {
            result = partial[c];
            break;
        }
        if (result == 0.0f && partial[c] == 0.0f) {
            result = 0.0f;
            continue;
        }
        result = std::fmax(result, partial[c]);
    }
    return result;
}

float min_f32_sequential(const float* x, std::int64_t n) {
    float result = x[0];
    for (std::int64_t i = 1; i < n; ++i) {
        if (std::isnan(result)) {
            break;
        }
        if (std::isnan(x[i])) {
            result = x[i];
            break;
        }
        result = std::fmin(result, x[i]);
    }
    return result;
}

float min_f32_parallel(const float* x, std::int64_t n) {
    const std::int64_t chunks = (n + parallel_threshold - 1) / parallel_threshold;
    std::vector<float> partial(static_cast<std::size_t>(chunks));
    MENSURFLOW_PARALLEL_FOR(if (chunks > 1))
    for (std::int64_t c = 0; c < chunks; ++c) {
        const std::int64_t begin = c * parallel_threshold;
        const std::int64_t end = std::min(begin + parallel_threshold, n);
        float result = x[begin];
        for (std::int64_t i = begin + 1; i < end; ++i) {
            if (std::isnan(result)) {
                break;
            }
            if (std::isnan(x[i])) {
                result = x[i];
                break;
            }
            result = std::fmin(result, x[i]);
        }
        partial[static_cast<std::size_t>(c)] = result;
    }
    float result = partial[0];
    for (std::size_t c = 1; c < partial.size(); ++c) {
        if (std::isnan(result)) {
            break;
        }
        if (std::isnan(partial[c])) {
            result = partial[c];
            break;
        }
        result = std::fmin(result, partial[c]);
    }
    return result;
}

float prod_f32(const float* x, std::int64_t n) {
    double product = 1.0;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold) reduction(* : product))
    for (std::int64_t i = 0; i < n; ++i) {
        product *= static_cast<double>(x[i]);
    }
    return static_cast<float>(product);
}

// Axis reductions view x as [outer, width, inner], where width is the reduced axis. Output entry
// `index` = (o, i) reduces x[o, 0..width, i], whose elements are `inner` apart in memory.

void sum_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis) {
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::int64_t d = 0; d < axis; ++d) {
        outer *= shape[d];
    }
    for (std::int64_t d = axis + 1; d < rank; ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[axis];
    MENSURFLOW_PARALLEL_FOR(if (outer * width * inner > parallel_work_threshold))
    for (std::int64_t index = 0; index < outer * inner; ++index) {
        const float* start = x + (index / inner) * width * inner + index % inner;
        double sum = 0.0;
        for (std::int64_t r = 0; r < width; ++r) {
            sum += static_cast<double>(start[r * inner]);
        }
        out[index] = static_cast<float>(sum);
    }
}

void mean_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                   std::int64_t axis) {
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::int64_t d = 0; d < axis; ++d) {
        outer *= shape[d];
    }
    for (std::int64_t d = axis + 1; d < rank; ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[axis];
    MENSURFLOW_PARALLEL_FOR(if (outer * width * inner > parallel_work_threshold))
    for (std::int64_t index = 0; index < outer * inner; ++index) {
        const float* start = x + (index / inner) * width * inner + index % inner;
        double sum = 0.0;
        for (std::int64_t r = 0; r < width; ++r) {
            sum += static_cast<double>(start[r * inner]);
        }
        out[index] = static_cast<float>(sum / static_cast<double>(width));
    }
}

void max_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis) {
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::int64_t d = 0; d < axis; ++d) {
        outer *= shape[d];
    }
    for (std::int64_t d = axis + 1; d < rank; ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[axis];
    MENSURFLOW_PARALLEL_FOR(if (outer * width * inner > parallel_work_threshold))
    for (std::int64_t index = 0; index < outer * inner; ++index) {
        const float* start = x + (index / inner) * width * inner + index % inner;
        float result = start[0];
        for (std::int64_t r = 1; r < width; ++r) {
            const float value = start[r * inner];
            if (std::isnan(result)) {
                break;
            }
            if (std::isnan(value)) {
                result = value;
                break;
            }
            result = std::fmax(result, value);
        }
        out[index] = result;
    }
}

void min_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis) {
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::int64_t d = 0; d < axis; ++d) {
        outer *= shape[d];
    }
    for (std::int64_t d = axis + 1; d < rank; ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[axis];
    MENSURFLOW_PARALLEL_FOR(if (outer * width * inner > parallel_work_threshold))
    for (std::int64_t index = 0; index < outer * inner; ++index) {
        const float* start = x + (index / inner) * width * inner + index % inner;
        float result = start[0];
        for (std::int64_t r = 1; r < width; ++r) {
            const float value = start[r * inner];
            if (std::isnan(result)) {
                break;
            }
            if (std::isnan(value)) {
                result = value;
                break;
            }
            result = std::fmin(result, value);
        }
        out[index] = result;
    }
}

void prod_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                   std::int64_t axis) {
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::int64_t d = 0; d < axis; ++d) {
        outer *= shape[d];
    }
    for (std::int64_t d = axis + 1; d < rank; ++d) {
        inner *= shape[d];
    }
    const std::int64_t width = shape[axis];
    MENSURFLOW_PARALLEL_FOR(if (outer * width * inner > parallel_work_threshold))
    for (std::int64_t index = 0; index < outer * inner; ++index) {
        const float* start = x + (index / inner) * width * inner + index % inner;
        double product = 1.0;
        for (std::int64_t r = 0; r < width; ++r) {
            product *= static_cast<double>(start[r * inner]);
        }
        out[index] = static_cast<float>(product);
    }
}

void argmax_f32(const float* x, std::int64_t* out, std::int64_t rows, std::int64_t width) {
    MENSURFLOW_PARALLEL_FOR(if (rows * width > parallel_work_threshold))
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* values = x + row * width;
        std::int64_t best = 0;
        for (std::int64_t i = 1; i < width && !std::isnan(values[best]); ++i) {
            if (std::isnan(values[i]) || values[i] > values[best]) {
                best = i;
            }
        }
        out[row] = best;
    }
}

namespace {

// x viewed as [outer, width, inner] around the reduced axis.
struct AxisLayout {
    std::int64_t outer = 1, width = 1, inner = 1;
};

AxisLayout axis_layout(const std::int64_t* shape, std::int64_t rank, std::int64_t axis) {
    AxisLayout layout;
    for (std::int64_t d = 0; d < axis; ++d) {
        layout.outer *= shape[d];
    }
    for (std::int64_t d = axis + 1; d < rank; ++d) {
        layout.inner *= shape[d];
    }
    layout.width = shape[axis];
    return layout;
}

struct Moments {
    double mean = 0.0;
    double m2 = 0.0; // sum of squared deviations from the mean
};

// Welford's one-pass update over `count` elements `stride` apart: each new value moves the mean
// by its deviation / k and adds deviation * (value - new mean) to M2. Unlike sum(x^2) - n mean^2
// it never subtracts two large, nearly equal numbers, so a large mean costs no precision.
Moments welford(const float* start, std::int64_t count, std::int64_t stride) {
    Moments m;
    for (std::int64_t k = 0; k < count; ++k) {
        const double value = static_cast<double>(start[k * stride]);
        const double delta = value - m.mean;
        m.mean += delta / static_cast<double>(k + 1);
        m.m2 += delta * (value - m.mean);
    }
    return m;
}

// count - correction, or 0 when that is not positive (the result is then NaN).
double variance_divisor(std::int64_t count, std::int64_t correction) {
    return count - correction > 0 ? static_cast<double>(count - correction) : 0.0;
}

constexpr double quiet_nan = std::numeric_limits<double>::quiet_NaN();

} // namespace

void var_axis_f32_sequential(const float* x, float* out, const std::int64_t* shape,
                             std::int64_t rank, std::int64_t axis, std::int64_t correction) {
    const AxisLayout l = axis_layout(shape, rank, axis);
    const double divisor = variance_divisor(l.width, correction);
    for (std::int64_t index = 0; index < l.outer * l.inner; ++index) {
        const float* start = x + (index / l.inner) * l.width * l.inner + index % l.inner;
        const Moments m = welford(start, l.width, l.inner);
        out[index] = static_cast<float>(divisor > 0.0 ? m.m2 / divisor : quiet_nan);
    }
}

void var_backward_f32_sequential(const float* x, const float* grad, float* grad_x,
                                 const std::int64_t* shape, std::int64_t rank, std::int64_t axis,
                                 std::int64_t correction) {
    const AxisLayout l = axis_layout(shape, rank, axis);
    const double divisor = variance_divisor(l.width, correction);
    for (std::int64_t index = 0; index < l.outer * l.inner; ++index) {
        const std::int64_t offset = (index / l.inner) * l.width * l.inner + index % l.inner;
        const Moments m = welford(x + offset, l.width, l.inner);
        const double scale =
            divisor > 0.0 ? 2.0 * static_cast<double>(grad[index]) / divisor : quiet_nan;
        for (std::int64_t r = 0; r < l.width; ++r) {
            const std::int64_t i = offset + r * l.inner;
            grad_x[i] = static_cast<float>(scale * (static_cast<double>(x[i]) - m.mean));
        }
    }
}

void argmin_f32_sequential(const float* x, std::int64_t* out, std::int64_t rows,
                           std::int64_t width) {
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* values = x + row * width;
        std::int64_t best = 0;
        for (std::int64_t i = 1; i < width && !std::isnan(values[best]); ++i) {
            if (std::isnan(values[i]) || values[i] < values[best]) {
                best = i;
            }
        }
        out[row] = best;
    }
}

} // namespace mensurflow::cpu
