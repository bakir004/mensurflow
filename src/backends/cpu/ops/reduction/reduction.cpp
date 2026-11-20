// CPU reduction kernels: one loop per operation. MENSURFLOW_PARALLEL_FOR (threads.hpp) splits the
// loop across threads when the build has OpenMP; sums and products combine the threads' partial
// results with OpenMP's `reduction` clause. max and min need NaN and signed-zero rules a reduction
// clause does not have, so they have two versions: a plain loop, and a parallel one that takes the
// maximum of fixed-size chunks in parallel, then of the chunks.
#include "backends/cpu/ops/reduction/reduction.hpp"

#include "backends/cpu/parallel.hpp"
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

void std_axis_f32_sequential(const float* x, float* out, const std::int64_t* shape,
                             std::int64_t rank, std::int64_t axis, std::int64_t correction) {
    const AxisLayout l = axis_layout(shape, rank, axis);
    const double divisor = variance_divisor(l.width, correction);
    for (std::int64_t index = 0; index < l.outer * l.inner; ++index) {
        const float* start = x + (index / l.inner) * l.width * l.inner + index % l.inner;
        const Moments m = welford(start, l.width, l.inner);
        out[index] = static_cast<float>(divisor > 0.0 ? std::sqrt(m.m2 / divisor) : quiet_nan);
    }
}

void std_backward_f32_sequential(const float* x, const float* grad, float* grad_x,
                                 const std::int64_t* shape, std::int64_t rank, std::int64_t axis,
                                 std::int64_t correction) {
    const AxisLayout l = axis_layout(shape, rank, axis);
    const double divisor = variance_divisor(l.width, correction);
    for (std::int64_t index = 0; index < l.outer * l.inner; ++index) {
        const std::int64_t offset = (index / l.inner) * l.width * l.inner + index % l.inner;
        const Moments m = welford(x + offset, l.width, l.inner);
        double scale = quiet_nan;
        if (divisor > 0.0) {
            const double deviation = std::sqrt(m.m2 / divisor);
            // A constant row has std 0 and every x - mean is 0: its gradient is 0, not 0 / 0.
            scale = deviation == 0.0 ? 0.0
                                     : static_cast<double>(grad[index]) / (divisor * deviation);
        }
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

// ---- Parallel versions of argmin, var and std ----------------------------------------------

namespace {

// The index of the smallest of values[begin, end) by argmin's rules (first minimum, first NaN).
// NaN is tested as v != v (one compare; MSVC may call a function for std::isnan), and the best
// value is kept in a local so the loop does not reload it.
std::int64_t argmin_range(const float* values, std::int64_t begin, std::int64_t end) {
    std::int64_t best = begin;
    float best_value = values[begin];
    for (std::int64_t i = begin + 1; i < end && best_value == best_value; ++i) {
        const float value = values[i];
        if (value != value || value < best_value) {
            best = i;
            best_value = value;
        }
    }
    return best;
}

// The winner of two candidates, `first` from earlier in the row: a NaN first wins, then a NaN
// second, then the strictly smaller second; otherwise the earlier one (ties keep the first).
std::int64_t argmin_merge(const float* values, std::int64_t first, std::int64_t second) {
    if (first < 0 || std::isnan(values[first])) {
        return first < 0 ? second : first;
    }
    return std::isnan(values[second]) || values[second] < values[first] ? second : first;
}

// Moments of a block: element count, mean and M2.
struct Block {
    double count = 0.0, mean = 0.0, m2 = 0.0;
};

// Chan et al.'s combination of two blocks' moments: exact for the union up to rounding, and it
// never subtracts large sums.
Block chan_merge(const Block& a, const Block& b) {
    if (a.count == 0.0) {
        return b;
    }
    if (b.count == 0.0) {
        return a;
    }
    const double count = a.count + b.count;
    const double delta = b.mean - a.mean;
    return {count, a.mean + delta * (b.count / count),
            a.m2 + b.m2 + delta * delta * (a.count * b.count / count)};
}

constexpr std::int64_t lanes = 8;          // interleaved Welford streams on a contiguous axis
constexpr std::int64_t inner_chunk = 256;  // output positions updated together on a strided axis
constexpr std::int64_t block_elements = std::int64_t{1} << 15;

// Moments of x[begin, end) on a contiguous axis: `lanes` interleaved Welford streams (element
// begin + k * lanes + l goes to lane l), merged in lane order at the end. The lanes' updates are
// independent, so the loop over them vectorizes, and each step needs one division for all lanes.
Block contiguous_block(const float* x, std::int64_t begin, std::int64_t end) {
    double mean[lanes] = {}, m2[lanes] = {};
    const std::int64_t steps = (end - begin) / lanes;
    for (std::int64_t k = 0; k < steps; ++k) {
        const float* v = x + begin + k * lanes;
        const double inverse = 1.0 / static_cast<double>(k + 1);
        for (std::int64_t l = 0; l < lanes; ++l) {
            const double value = static_cast<double>(v[l]);
            const double delta = value - mean[l];
            mean[l] += delta * inverse;
            m2[l] += delta * (value - mean[l]);
        }
    }
    Block result;
    for (std::int64_t l = 0; l < lanes; ++l) {
        result = chan_merge(result, {static_cast<double>(steps), mean[l], m2[l]});
    }
    // The tail (fewer than `lanes` elements) as a block of its own.
    const Moments tail = welford(x + begin + steps * lanes, end - begin - steps * lanes, 1);
    return chan_merge(result,
                      {static_cast<double>(end - begin - steps * lanes), tail.mean, tail.m2});
}

// The plan of a blocked reduction over x viewed as [outer, width, inner].
struct MomentPlan {
    AxisLayout l;
    std::int64_t chunk;  // inner positions per task (1 on a contiguous axis)
    std::int64_t chunks; // tasks across inner
    std::int64_t rows;   // axis elements per block
    std::int64_t blocks; // blocks along the axis
};

MomentPlan moment_plan(const std::int64_t* shape, std::int64_t rank, std::int64_t axis) {
    MomentPlan p;
    p.l = axis_layout(shape, rank, axis);
    p.chunk = std::min(p.l.inner, inner_chunk);
    p.chunks = (p.l.inner + p.chunk - 1) / p.chunk;
    p.rows = std::max<std::int64_t>(lanes, block_elements / p.chunk);
    p.blocks = std::max<std::int64_t>(1, (p.l.width + p.rows - 1) / p.rows);
    return p;
}

// Moments of block b of the axis for inner positions [i0, i1) of outer index o, into out[i - i0].
void strided_block(const float* x, const MomentPlan& p, std::int64_t o, std::int64_t b,
                   std::int64_t i0, std::int64_t i1, Block* out) {
    const std::int64_t r0 = b * p.rows;
    const std::int64_t r1 = std::min(p.l.width, r0 + p.rows);
    if (p.l.inner == 1) {
        out[0] = contiguous_block(x + o * p.l.width, r0, r1);
        return;
    }
    const std::int64_t n = i1 - i0;
    std::vector<double> mean(static_cast<std::size_t>(n), 0.0), m2(mean);
    for (std::int64_t r = r0; r < r1; ++r) {
        const float* row = x + (o * p.l.width + r) * p.l.inner + i0;
        const double inverse = 1.0 / static_cast<double>(r - r0 + 1);
        for (std::int64_t i = 0; i < n; ++i) {
            const double value = static_cast<double>(row[i]);
            const double delta = value - mean[static_cast<std::size_t>(i)];
            mean[static_cast<std::size_t>(i)] += delta * inverse;
            m2[static_cast<std::size_t>(i)] += delta * (value - mean[static_cast<std::size_t>(i)]);
        }
    }
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = {static_cast<double>(r1 - r0), mean[static_cast<std::size_t>(i)],
                  m2[static_cast<std::size_t>(i)]};
    }
}

// The moments of every output position [outer * inner], blocks computed in parallel and merged
// in block order.
std::vector<Block> blocked_moments(const float* x, const MomentPlan& p) {
    const std::int64_t outputs = p.l.outer * p.l.inner;
    // partial[(o * blocks + b) * inner + i]
    std::vector<Block> partial(static_cast<std::size_t>(outputs * p.blocks));
    const std::int64_t tasks = p.l.outer * p.blocks * p.chunks;
    parallel_for(tasks, grain_for(moments_parallel_cutoff, std::min(p.rows, p.l.width) * p.chunk),
                 [&](std::int64_t t0, std::int64_t t1) {
                     for (std::int64_t t = t0; t < t1; ++t) {
                         const std::int64_t c = t % p.chunks;
                         const std::int64_t b = (t / p.chunks) % p.blocks;
                         const std::int64_t o = t / (p.chunks * p.blocks);
                         const std::int64_t i0 = c * p.chunk;
                         const std::int64_t i1 = std::min(p.l.inner, i0 + p.chunk);
                         strided_block(x, p, o, b, i0, i1,
                                       partial.data() + (o * p.blocks + b) * p.l.inner + i0);
                     }
                 });
    if (p.blocks == 1) {
        return partial;
    }
    std::vector<Block> result(static_cast<std::size_t>(outputs));
    parallel_for(outputs, grain_for(moments_parallel_cutoff, p.blocks),
                 [&](std::int64_t j0, std::int64_t j1) {
                     for (std::int64_t j = j0; j < j1; ++j) {
                         const std::int64_t o = j / p.l.inner;
                         const std::int64_t i = j % p.l.inner;
                         Block merged;
                         for (std::int64_t b = 0; b < p.blocks; ++b) {
                             merged = chan_merge(
                                 merged,
                                 partial[static_cast<std::size_t>((o * p.blocks + b) * p.l.inner + i)]);
                         }
                         result[static_cast<std::size_t>(j)] = merged;
                     }
                 });
    return result;
}

template <bool Sqrt>
void moment_axis_parallel(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                          std::int64_t axis, std::int64_t correction) {
    const MomentPlan p = moment_plan(shape, rank, axis);
    const double divisor = variance_divisor(p.l.width, correction);
    const std::vector<Block> moments = blocked_moments(x, p);
    parallel_for(static_cast<std::int64_t>(moments.size()), moments_parallel_cutoff,
                 [&](std::int64_t j0, std::int64_t j1) {
                     for (std::int64_t j = j0; j < j1; ++j) {
                         const double m2 = moments[static_cast<std::size_t>(j)].m2;
                         const double variance = divisor > 0.0 ? m2 / divisor : quiet_nan;
                         out[j] = static_cast<float>(Sqrt ? std::sqrt(variance) : variance);
                     }
                 });
}

// grad_x = scale * (x - mean), with scale per output position: 2 grad / divisor for var,
// grad / (divisor * std) (0 where std is 0) for std, NaN where divisor <= 0.
template <bool Std>
void moment_backward_parallel(const float* x, const float* grad, float* grad_x,
                              const std::int64_t* shape, std::int64_t rank, std::int64_t axis,
                              std::int64_t correction) {
    const MomentPlan p = moment_plan(shape, rank, axis);
    const double divisor = variance_divisor(p.l.width, correction);
    const std::vector<Block> moments = blocked_moments(x, p);
    std::vector<double> scale(moments.size()), mean(moments.size());
    parallel_for(static_cast<std::int64_t>(moments.size()), moments_parallel_cutoff,
                 [&](std::int64_t j0, std::int64_t j1) {
                     for (auto j = static_cast<std::size_t>(j0); j < static_cast<std::size_t>(j1);
                          ++j) {
                         mean[j] = moments[j].mean;
                         const double g = static_cast<double>(grad[j]);
                         if (divisor <= 0.0) {
                             scale[j] = quiet_nan;
                         } else if (Std) {
                             const double deviation = std::sqrt(moments[j].m2 / divisor);
                             scale[j] = deviation == 0.0 ? 0.0 : g / (divisor * deviation);
                         } else {
                             scale[j] = 2.0 * g / divisor;
                         }
                     }
                 });
    // One task per row of `inner` contiguous elements.
    const std::int64_t rows = p.l.outer * p.l.width;
    parallel_for(rows, grain_for(moments_parallel_cutoff, p.l.inner),
                 [&](std::int64_t r0, std::int64_t r1) {
                     for (std::int64_t r = r0; r < r1; ++r) {
                         const std::int64_t o = r / p.l.width;
                         const float* in = x + r * p.l.inner;
                         float* out = grad_x + r * p.l.inner;
                         const double* s = scale.data() + o * p.l.inner;
                         const double* m = mean.data() + o * p.l.inner;
                         for (std::int64_t i = 0; i < p.l.inner; ++i) {
                             out[i] = static_cast<float>(s[i] * (static_cast<double>(in[i]) - m[i]));
                         }
                     }
                 });
}

} // namespace

void argmin_f32_parallel(const float* x, std::int64_t* out, std::int64_t rows,
                         std::int64_t width) {
    // argmin is exact under any split, so the choice may depend on the thread count.
    if (width <= argmin_block || rows >= max_threads()) {
        parallel_for(rows, grain_for(moments_parallel_cutoff, width),
                     [&](std::int64_t r0, std::int64_t r1) {
                         for (std::int64_t r = r0; r < r1; ++r) {
                             out[r] = argmin_range(x + r * width, 0, width);
                         }
                     });
        return;
    }
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* values = x + r * width;
        out[r] = parallel_reduce(
            width, argmin_block, std::int64_t{-1},
            [&](std::int64_t begin, std::int64_t end) { return argmin_range(values, begin, end); },
            [&](std::int64_t a, std::int64_t b) { return argmin_merge(values, a, b); },
            grain_for(moments_parallel_cutoff, argmin_block));
    }
}

void var_axis_f32_parallel(const float* x, float* out, const std::int64_t* shape,
                           std::int64_t rank, std::int64_t axis, std::int64_t correction) {
    moment_axis_parallel<false>(x, out, shape, rank, axis, correction);
}

void std_axis_f32_parallel(const float* x, float* out, const std::int64_t* shape,
                           std::int64_t rank, std::int64_t axis, std::int64_t correction) {
    moment_axis_parallel<true>(x, out, shape, rank, axis, correction);
}

void var_backward_f32_parallel(const float* x, const float* grad, float* grad_x,
                               const std::int64_t* shape, std::int64_t rank, std::int64_t axis,
                               std::int64_t correction) {
    moment_backward_parallel<false>(x, grad, grad_x, shape, rank, axis, correction);
}

void std_backward_f32_parallel(const float* x, const float* grad, float* grad_x,
                               const std::int64_t* shape, std::int64_t rank, std::int64_t axis,
                               std::int64_t correction) {
    moment_backward_parallel<true>(x, grad, grad_x, shape, rank, axis, correction);
}

} // namespace mensurflow::cpu
