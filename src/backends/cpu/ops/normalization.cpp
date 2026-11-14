#include "backends/cpu/ops/normalization.hpp"

#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

// Promises the compiler that the pointers marked with it do not overlap, which lets it vectorize
// loops that read one array and write another.
#if defined(_MSC_VER)
#define MENSURFLOW_RESTRICT __restrict
#else
#define MENSURFLOW_RESTRICT __restrict__
#endif

namespace mensurflow::cpu {

namespace {

// ---------------------------------------------------------------------------- line sums
// Sums over one contiguous line use eight independent float lanes (so the compiler can vectorize
// them and the add chain is short) and combine the lanes in double. The lane order is fixed, so
// the result does not depend on the thread count.
constexpr int lanes = 8;

double combine(const float (&acc)[lanes]) {
    double total = 0.0;
    for (int k = 0; k < lanes; ++k) {
        total += static_cast<double>(acc[k]);
    }
    return total;
}

[[maybe_unused]] double line_sum_portable(const float* x, std::int64_t n) {
    float acc[lanes] = {};
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        for (int k = 0; k < lanes; ++k) {
            acc[k] += x[i + k];
        }
    }
    double total = combine(acc);
    for (; i < n; ++i) {
        total += static_cast<double>(x[i]);
    }
    return total;
}

// sum of (x - mean)^2
[[maybe_unused]] double line_sum_sq_dev_portable(const float* x, std::int64_t n, float mean) {
    float acc[lanes] = {};
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        for (int k = 0; k < lanes; ++k) {
            const float d = x[i + k] - mean;
            acc[k] += d * d;
        }
    }
    double total = combine(acc);
    for (; i < n; ++i) {
        const float d = x[i] - mean;
        total += static_cast<double>(d * d);
    }
    return total;
}

// sum of a * b
[[maybe_unused]] double line_dot_portable(const float* a, const float* b, std::int64_t n) {
    float acc[lanes] = {};
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        for (int k = 0; k < lanes; ++k) {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    double total = combine(acc);
    for (; i < n; ++i) {
        total += static_cast<double>(a[i] * b[i]);
    }
    return total;
}

// sum of a * (x - mean)
[[maybe_unused]] double line_dot_dev_portable(const float* a, const float* x, std::int64_t n,
                                              float mean) {
    float acc[lanes] = {};
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        for (int k = 0; k < lanes; ++k) {
            acc[k] += a[i + k] * (x[i + k] - mean);
        }
    }
    double total = combine(acc);
    for (; i < n; ++i) {
        total += static_cast<double>(a[i] * (x[i] - mean));
    }
    return total;
}

// sum of dy and sum of dy * (x - mean) in one pass.
[[maybe_unused]] void line_sum_and_dot_dev_portable(const float* dy, const float* x, std::int64_t n,
                                                    float mean, double& sum, double& dot) {
    float acc_sum[lanes] = {};
    float acc_dot[lanes] = {};
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        for (int k = 0; k < lanes; ++k) {
            acc_sum[k] += dy[i + k];
            acc_dot[k] += dy[i + k] * (x[i + k] - mean);
        }
    }
    sum = combine(acc_sum);
    dot = combine(acc_dot);
    for (; i < n; ++i) {
        sum += static_cast<double>(dy[i]);
        dot += static_cast<double>(dy[i] * (x[i] - mean));
    }
}

// ----------------------------------------------------------------------- SSE2 versions
// The portable loops above are what the compiler may or may not vectorize (MSVC does not keep the
// eight lanes in registers). Where SSE2 exists (every x86-64 CPU, so no special build flag) the same
// eight lanes are two __m128 accumulators: lane k holds elements i % 8 == k, exactly like the
// portable code, so both give the same bits.
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define MENSURFLOW_NORM_SSE2 1
#include <emmintrin.h>

double line_sum(const float* x, std::int64_t n) {
    __m128 lo = _mm_setzero_ps(), hi = _mm_setzero_ps();
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        lo = _mm_add_ps(lo, _mm_loadu_ps(x + i));
        hi = _mm_add_ps(hi, _mm_loadu_ps(x + i + 4));
    }
    float acc[lanes];
    _mm_storeu_ps(acc, lo);
    _mm_storeu_ps(acc + 4, hi);
    double total = combine(acc);
    for (; i < n; ++i) {
        total += static_cast<double>(x[i]);
    }
    return total;
}

double line_sum_sq_dev(const float* x, std::int64_t n, float mean) {
    const __m128 m = _mm_set1_ps(mean);
    __m128 lo = _mm_setzero_ps(), hi = _mm_setzero_ps();
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        const __m128 a = _mm_sub_ps(_mm_loadu_ps(x + i), m);
        const __m128 b = _mm_sub_ps(_mm_loadu_ps(x + i + 4), m);
        lo = _mm_add_ps(lo, _mm_mul_ps(a, a));
        hi = _mm_add_ps(hi, _mm_mul_ps(b, b));
    }
    float acc[lanes];
    _mm_storeu_ps(acc, lo);
    _mm_storeu_ps(acc + 4, hi);
    double total = combine(acc);
    for (; i < n; ++i) {
        const float d = x[i] - mean;
        total += static_cast<double>(d * d);
    }
    return total;
}

double line_dot(const float* a, const float* b, std::int64_t n) {
    __m128 lo = _mm_setzero_ps(), hi = _mm_setzero_ps();
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        lo = _mm_add_ps(lo, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        hi = _mm_add_ps(hi, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    float acc[lanes];
    _mm_storeu_ps(acc, lo);
    _mm_storeu_ps(acc + 4, hi);
    double total = combine(acc);
    for (; i < n; ++i) {
        total += static_cast<double>(a[i] * b[i]);
    }
    return total;
}

double line_dot_dev(const float* a, const float* x, std::int64_t n, float mean) {
    const __m128 m = _mm_set1_ps(mean);
    __m128 lo = _mm_setzero_ps(), hi = _mm_setzero_ps();
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        lo = _mm_add_ps(lo, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_sub_ps(_mm_loadu_ps(x + i), m)));
        hi = _mm_add_ps(
            hi, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_sub_ps(_mm_loadu_ps(x + i + 4), m)));
    }
    float acc[lanes];
    _mm_storeu_ps(acc, lo);
    _mm_storeu_ps(acc + 4, hi);
    double total = combine(acc);
    for (; i < n; ++i) {
        total += static_cast<double>(a[i] * (x[i] - mean));
    }
    return total;
}

void line_sum_and_dot_dev(const float* dy, const float* x, std::int64_t n, float mean, double& sum,
                          double& dot) {
    const __m128 m = _mm_set1_ps(mean);
    __m128 sum_lo = _mm_setzero_ps(), sum_hi = _mm_setzero_ps();
    __m128 dot_lo = _mm_setzero_ps(), dot_hi = _mm_setzero_ps();
    std::int64_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        const __m128 g0 = _mm_loadu_ps(dy + i), g1 = _mm_loadu_ps(dy + i + 4);
        sum_lo = _mm_add_ps(sum_lo, g0);
        sum_hi = _mm_add_ps(sum_hi, g1);
        dot_lo = _mm_add_ps(dot_lo, _mm_mul_ps(g0, _mm_sub_ps(_mm_loadu_ps(x + i), m)));
        dot_hi = _mm_add_ps(dot_hi, _mm_mul_ps(g1, _mm_sub_ps(_mm_loadu_ps(x + i + 4), m)));
    }
    float acc_sum[lanes], acc_dot[lanes];
    _mm_storeu_ps(acc_sum, sum_lo);
    _mm_storeu_ps(acc_sum + 4, sum_hi);
    _mm_storeu_ps(acc_dot, dot_lo);
    _mm_storeu_ps(acc_dot + 4, dot_hi);
    sum = combine(acc_sum);
    dot = combine(acc_dot);
    for (; i < n; ++i) {
        sum += static_cast<double>(dy[i]);
        dot += static_cast<double>(dy[i] * (x[i] - mean));
    }
}

#else

double line_sum(const float* x, std::int64_t n) { return line_sum_portable(x, n); }

double line_sum_sq_dev(const float* x, std::int64_t n, float mean) {
    return line_sum_sq_dev_portable(x, n, mean);
}

double line_dot(const float* a, const float* b, std::int64_t n) {
    return line_dot_portable(a, b, n);
}

double line_dot_dev(const float* a, const float* x, std::int64_t n, float mean) {
    return line_dot_dev_portable(a, x, n, mean);
}

void line_sum_and_dot_dev(const float* dy, const float* x, std::int64_t n, float mean, double& sum,
                          double& dot) {
    line_sum_and_dot_dev_portable(dy, x, n, mean, sum, dot);
}

#endif

// sum of x^2
double line_sum_sq(const float* x, std::int64_t n) { return line_sum_sq_dev(x, n, 0.0f); }

float inverse_std(double variance, float eps) {
    return static_cast<float>(1.0 / std::sqrt(variance + static_cast<double>(eps)));
}

// Sums that cross lines are accumulated per block of consecutive lines. The number of blocks
// depends only on the number of lines, never on the thread count, so the result is the same on
// every machine; the blocks are what the threads share.
constexpr std::int64_t max_blocks = 64;

std::int64_t block_count(std::int64_t lines) { return std::min(lines, max_blocks); }

std::int64_t block_begin(std::int64_t block, std::int64_t blocks, std::int64_t lines) {
    return block * lines / blocks;
}

bool worth_threads(std::int64_t elements) { return elements > parallel_threshold; }

} // namespace

// ------------------------------------------------------------------------------ layer norm

void layer_norm_f32(const float* x, const float* w, const float* b, float* y, std::int64_t rows,
                    std::int64_t width, float eps) {
    const double inv_width = 1.0 / static_cast<double>(width);
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(rows * width)))
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* in = x + r * width;
        float* out = y + r * width;
        const float mean = static_cast<float>(line_sum(in, width) * inv_width);
        const float inv = inverse_std(line_sum_sq_dev(in, width, mean) * inv_width, eps);
        for (std::int64_t j = 0; j < width; ++j) {
            out[j] = (in[j] - mean) * inv * w[j] + b[j];
        }
    }
}

void layer_norm_backward_f32(const float* x, const float* dy, const float* w, float* dx, float* dw,
                             float* db, std::int64_t rows, std::int64_t width, float eps) {
    const std::int64_t blocks = block_count(rows);
    // Per block: partial sums of dw and db, then two scratch lines (xhat and w * dy), laid out
    // [block][dw | db | xhat | g][width]. Only the sums need zeroing, and the buffer is allocated
    // once instead of once per block.
    const std::int64_t stride = 4 * width;
    const std::unique_ptr<float[]> partial(new float[static_cast<std::size_t>(blocks * stride)]);
    const double inv_width = 1.0 / static_cast<double>(width);
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(rows * width)) schedule(static))
    for (std::int64_t blk = 0; blk < blocks; ++blk) {
        float* dw_part = partial.get() + blk * stride;
        float* db_part = dw_part + width;
        float* xhat = db_part + width;
        float* g = xhat + width;
        std::fill(dw_part, dw_part + 2 * width, 0.0f);
        for (std::int64_t r = block_begin(blk, blocks, rows);
             r < block_begin(blk + 1, blocks, rows); ++r) {
            const float* in = x + r * width;
            const float* grad = dy + r * width;
            const float mean = static_cast<float>(line_sum(in, width) * inv_width);
            const float inv = inverse_std(line_sum_sq_dev(in, width, mean) * inv_width, eps);
            for (std::int64_t j = 0; j < width; ++j) {
                const float n = (in[j] - mean) * inv;
                xhat[j] = n;
                g[j] = grad[j] * w[j];
                dw_part[j] += grad[j] * n;
                db_part[j] += grad[j];
            }
            const double sum_g = line_sum(g, width), sum_gx = line_dot(g, xhat, width);
            const float mean_g = static_cast<float>(sum_g * inv_width);
            const float mean_gx = static_cast<float>(sum_gx * inv_width);
            float* out = dx + r * width;
            for (std::int64_t j = 0; j < width; ++j) {
                out[j] = inv * (g[j] - mean_g - xhat[j] * mean_gx);
            }
        }
    }
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(blocks * width)) schedule(static))
    for (std::int64_t j = 0; j < width; ++j) {
        float sum_w = 0.0f;
        float sum_b = 0.0f;
        for (std::int64_t blk = 0; blk < blocks; ++blk) {
            sum_w += partial[static_cast<std::size_t>(blk * stride + j)];
            sum_b += partial[static_cast<std::size_t>(blk * stride + width + j)];
        }
        dw[j] = sum_w;
        db[j] = sum_b;
    }
}

// -------------------------------------------------------------------------------- RMS norm

void rms_norm_f32(const float* x, const float* w, float* y, std::int64_t rows, std::int64_t width,
                  float eps) {
    const double inv_width = 1.0 / static_cast<double>(width);
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(rows * width)))
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* in = x + r * width;
        float* out = y + r * width;
        const float inv = inverse_std(line_sum_sq(in, width) * inv_width, eps);
        for (std::int64_t j = 0; j < width; ++j) {
            out[j] = in[j] * inv * w[j];
        }
    }
}

void rms_norm_backward_f32(const float* x, const float* dy, const float* w, float* dx, float* dw,
                           std::int64_t rows, std::int64_t width, float eps) {
    const std::int64_t blocks = block_count(rows);
    const std::int64_t stride = 3 * width; // [block][dw | xhat | g][width]
    const std::unique_ptr<float[]> partial(new float[static_cast<std::size_t>(blocks * stride)]);
    const double inv_width = 1.0 / static_cast<double>(width);
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(rows * width)) schedule(static))
    for (std::int64_t blk = 0; blk < blocks; ++blk) {
        float* dw_part = partial.get() + blk * stride;
        float* xhat = dw_part + width;
        float* g = xhat + width;
        std::fill(dw_part, dw_part + width, 0.0f);
        for (std::int64_t r = block_begin(blk, blocks, rows);
             r < block_begin(blk + 1, blocks, rows); ++r) {
            const float* in = x + r * width;
            const float* grad = dy + r * width;
            const float inv = inverse_std(line_sum_sq(in, width) * inv_width, eps);
            for (std::int64_t j = 0; j < width; ++j) {
                const float n = in[j] * inv;
                xhat[j] = n;
                g[j] = grad[j] * w[j];
                dw_part[j] += grad[j] * n;
            }
            const float mean_gx = static_cast<float>(line_dot(g, xhat, width) * inv_width);
            float* out = dx + r * width;
            for (std::int64_t j = 0; j < width; ++j) {
                out[j] = inv * (g[j] - xhat[j] * mean_gx);
            }
        }
    }
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(blocks * width)) schedule(static))
    for (std::int64_t j = 0; j < width; ++j) {
        float sum = 0.0f;
        for (std::int64_t blk = 0; blk < blocks; ++blk) {
            sum += partial[static_cast<std::size_t>(blk * stride + j)];
        }
        dw[j] = sum;
    }
}

// ------------------------------------------------------------------------------ batch norm
namespace {

// Per-channel sums over the n * s elements of each channel of an [n, c, s] tensor.
//
// For s > 1 the unit of work is a plane (one channel of one sample: s contiguous floats). Each
// plane's sums go to a [sample][channel] table, which is then added in sample order.
// For s == 1 ([n, c] input; one channel's elements are c apart) the unit is a block of rows, and
// each block adds whole rows (contiguous over the channels) into its own per-channel sums; the
// blocks are then added in order.
// Either way the additions happen in an order fixed by the shape alone.
struct Dims {
    std::int64_t n, c, s;
};

// Rows added in float before their sums are added to a double accumulator, for the [n, c] path.
constexpr std::int64_t row_group = 32;

// sums[ch] = sum over the channel of x.
void channel_sum(const float* x, const Dims& d, double* sums) {
    const std::int64_t dn = d.n, dc = d.c, ds = d.s;
    if (ds > 1) {
        std::vector<double> part(static_cast<std::size_t>(dn * dc));
        MENSURFLOW_PARALLEL_FOR(if (worth_threads(dn * dc * ds)) schedule(static))
        for (std::int64_t p = 0; p < dn * dc; ++p) {
            part[static_cast<std::size_t>(p)] = line_sum(x + p * ds, ds);
        }
        for (std::int64_t ch = 0; ch < dc; ++ch) {
            double total = 0.0;
            for (std::int64_t i = 0; i < dn; ++i) {
                total += part[static_cast<std::size_t>(i * dc + ch)];
            }
            sums[ch] = total;
        }
        return;
    }
    const std::int64_t blocks = block_count(dn);
    std::vector<double> part(static_cast<std::size_t>(blocks * dc), 0.0);
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(dn * dc)) schedule(static))
    for (std::int64_t blk = 0; blk < blocks; ++blk) {
        double* acc = part.data() + blk * dc;
        // Rows are added in float (vectorized across the channels) in groups of row_group rows,
        // and each group's sums are then added to the double accumulators.
        std::vector<float> group(static_cast<std::size_t>(dc));
        const std::int64_t end = block_begin(blk + 1, blocks, dn);
        for (std::int64_t r = block_begin(blk, blocks, dn); r < end;) {
            const std::int64_t stop = std::min(end, r + row_group);
            std::fill(group.begin(), group.end(), 0.0f);
            for (; r < stop; ++r) {
                const float* MENSURFLOW_RESTRICT row = x + r * dc;
                float* MENSURFLOW_RESTRICT g = group.data();
                for (std::int64_t ch = 0; ch < dc; ++ch) {
                    g[ch] += row[ch];
                }
            }
            for (std::int64_t ch = 0; ch < dc; ++ch) {
                acc[ch] += static_cast<double>(group[static_cast<std::size_t>(ch)]);
            }
        }
    }
    // Blocks are added in order; the block loop is outside so the reads are contiguous.
    std::fill(sums, sums + dc, 0.0);
    for (std::int64_t blk = 0; blk < blocks; ++blk) {
        const double* MENSURFLOW_RESTRICT block = part.data() + blk * dc;
        double* MENSURFLOW_RESTRICT out = sums;
        for (std::int64_t ch = 0; ch < dc; ++ch) {
            out[ch] += block[ch];
        }
    }
}

// sums[ch] = sum over the channel of (x - mean[ch])^2 when `dy` is null, else of
// dy * (x - mean[ch]); and, when `dy` is given and `sum_dy` is not null, sum_dy[ch] = sum of dy.
void channel_deviation_sum(const float* x, const float* dy, const float* mean, const Dims& d,
                           double* sums, double* sum_dy) {
    const std::int64_t dn = d.n, dc = d.c, ds = d.s;
    const bool want_dy = dy != nullptr && sum_dy != nullptr;
    if (ds > 1) {
        std::vector<double> part(static_cast<std::size_t>(2 * dn * dc));
        MENSURFLOW_PARALLEL_FOR(if (worth_threads(dn * dc * ds)) schedule(static))
        for (std::int64_t p = 0; p < dn * dc; ++p) {
            const float m = mean[p % dc];
            const float* xs = x + p * ds;
            if (want_dy) {
                line_sum_and_dot_dev(dy + p * ds, xs, ds, m,
                                     part[static_cast<std::size_t>(2 * p + 1)],
                                     part[static_cast<std::size_t>(2 * p)]);
            } else {
                part[static_cast<std::size_t>(2 * p)] = dy != nullptr
                                                            ? line_dot_dev(dy + p * ds, xs, ds, m)
                                                            : line_sum_sq_dev(xs, ds, m);
                part[static_cast<std::size_t>(2 * p + 1)] = 0.0;
            }
        }
        for (std::int64_t ch = 0; ch < dc; ++ch) {
            double total = 0.0;
            double total_dy = 0.0;
            for (std::int64_t i = 0; i < dn; ++i) {
                total += part[static_cast<std::size_t>(2 * (i * dc + ch))];
                total_dy += part[static_cast<std::size_t>(2 * (i * dc + ch) + 1)];
            }
            sums[ch] = total;
            if (want_dy) {
                sum_dy[ch] = total_dy;
            }
        }
        return;
    }
    const std::int64_t blocks = block_count(dn);
    std::vector<double> part(static_cast<std::size_t>(2 * blocks * dc), 0.0);
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(dn * dc)) schedule(static))
    for (std::int64_t blk = 0; blk < blocks; ++blk) {
        double* acc = part.data() + 2 * blk * dc;
        double* acc_dy = acc + dc;
        // Float sums over groups of row_group rows, then added to the double accumulators.
        std::vector<float> group(static_cast<std::size_t>(2 * dc));
        float* MENSURFLOW_RESTRICT group_sum = group.data();
        float* MENSURFLOW_RESTRICT group_dy = group_sum + dc;
        const float* MENSURFLOW_RESTRICT mean_p = mean;
        const std::int64_t end = block_begin(blk + 1, blocks, dn);
        for (std::int64_t r = block_begin(blk, blocks, dn); r < end;) {
            const std::int64_t stop = std::min(end, r + row_group);
            std::fill(group.begin(), group.end(), 0.0f);
            for (; r < stop; ++r) {
                const float* MENSURFLOW_RESTRICT xr = x + r * dc;
                if (dy != nullptr) {
                    const float* MENSURFLOW_RESTRICT gr = dy + r * dc;
                    for (std::int64_t ch = 0; ch < dc; ++ch) {
                        group_sum[ch] += gr[ch] * (xr[ch] - mean_p[ch]);
                        group_dy[ch] += gr[ch];
                    }
                } else {
                    for (std::int64_t ch = 0; ch < dc; ++ch) {
                        const float dev = xr[ch] - mean_p[ch];
                        group_sum[ch] += dev * dev;
                    }
                }
            }
            for (std::int64_t ch = 0; ch < dc; ++ch) {
                acc[ch] += static_cast<double>(group_sum[ch]);
                acc_dy[ch] += static_cast<double>(group_dy[ch]);
            }
        }
    }
    std::fill(sums, sums + dc, 0.0);
    if (want_dy) {
        std::fill(sum_dy, sum_dy + dc, 0.0);
    }
    for (std::int64_t blk = 0; blk < blocks; ++blk) {
        const double* MENSURFLOW_RESTRICT block = part.data() + 2 * blk * dc;
        double* MENSURFLOW_RESTRICT out = sums;
        for (std::int64_t ch = 0; ch < dc; ++ch) {
            out[ch] += block[ch];
        }
        if (want_dy) {
            double* MENSURFLOW_RESTRICT out_dy = sum_dy;
            for (std::int64_t ch = 0; ch < dc; ++ch) {
                out_dy[ch] += block[dc + ch];
            }
        }
    }
}

// The batch mean and biased variance of each channel.
void batch_statistics(const float* x, const Dims& d, std::vector<double>& mean,
                      std::vector<double>& variance) {
    const double count = static_cast<double>(d.n * d.s);
    channel_sum(x, d, mean.data());
    std::vector<float> mean_f(static_cast<std::size_t>(d.c));
    for (std::size_t i = 0; i < mean_f.size(); ++i) {
        mean[i] /= count;
        mean_f[i] = static_cast<float>(mean[i]);
    }
    channel_deviation_sum(x, nullptr, mean_f.data(), d, variance.data(), nullptr);
    for (double& v : variance) {
        v /= count;
    }
}

// out = in * scale[ch] + shift[ch] over [n, c, s], where the shift already folds in the mean.
void channel_affine(const float* in, float* out, const float* scale, const float* shift,
                    const Dims& d) {
    const std::int64_t dn = d.n, dc = d.c, ds = d.s;
    if (ds > 1) {
        MENSURFLOW_PARALLEL_FOR(if (worth_threads(dn * dc * ds)) schedule(static))
        for (std::int64_t p = 0; p < dn * dc; ++p) {
            const float k = scale[p % dc];
            const float t = shift[p % dc];
            const float* src = in + p * ds;
            float* dst = out + p * ds;
            for (std::int64_t i = 0; i < ds; ++i) {
                dst[i] = src[i] * k + t;
            }
        }
        return;
    }
    MENSURFLOW_PARALLEL_FOR(if (worth_threads(dn * dc)) schedule(static))
    for (std::int64_t r = 0; r < dn; ++r) {
        for (std::int64_t ch = 0; ch < dc; ++ch) {
            out[r * dc + ch] = in[r * dc + ch] * scale[ch] + shift[ch];
        }
    }
}

} // namespace

void batch_norm_f32(const float* x, const float* w, const float* b, float* running_mean,
                    float* running_var, float* y, std::int64_t n, std::int64_t c, std::int64_t s,
                    bool training, float momentum, float eps) {
    const Dims d{n, c, s};
    const auto channels = static_cast<std::size_t>(c);
    std::vector<float> scale(channels);
    std::vector<float> shift(channels);
    // y = (x - mean) * (w * inv) + b = x * scale + shift, with shift = b - mean * scale.
    if (training) {
        std::vector<double> mean(channels);
        std::vector<double> variance(channels);
        batch_statistics(x, d, mean, variance);
        const double count = static_cast<double>(n * s);
        const double unbias = count / (count - 1.0); // callers guarantee count >= 2
        for (std::size_t i = 0; i < channels; ++i) {
            const float m = static_cast<float>(mean[i]);
            scale[i] = w[i] * inverse_std(variance[i], eps);
            shift[i] = b[i] - m * scale[i];
            running_mean[i] = (1.0f - momentum) * running_mean[i] + momentum * m;
            running_var[i] = (1.0f - momentum) * running_var[i] +
                             momentum * static_cast<float>(variance[i] * unbias);
        }
    } else {
        for (std::size_t i = 0; i < channels; ++i) {
            scale[i] = w[i] * inverse_std(static_cast<double>(running_var[i]), eps);
            shift[i] = b[i] - running_mean[i] * scale[i];
        }
    }
    channel_affine(x, y, scale.data(), shift.data(), d);
}

void batch_norm_backward_f32(const float* x, const float* dy, const float* w,
                             const float* running_mean, const float* running_var, float* dx,
                             float* dw, float* db, std::int64_t n, std::int64_t c, std::int64_t s,
                             bool training, float eps) {
    const Dims d{n, c, s};
    const auto channels = static_cast<std::size_t>(c);
    std::vector<float> mean(channels);
    std::vector<float> inv(channels);
    if (training) {
        std::vector<double> mean_d(channels);
        std::vector<double> variance(channels);
        batch_statistics(x, d, mean_d, variance);
        for (std::size_t i = 0; i < channels; ++i) {
            mean[i] = static_cast<float>(mean_d[i]);
            inv[i] = inverse_std(variance[i], eps);
        }
    } else {
        for (std::size_t i = 0; i < channels; ++i) {
            mean[i] = running_mean[i];
            inv[i] = inverse_std(static_cast<double>(running_var[i]), eps);
        }
    }
    // db = sum dy; dw = sum dy * (x - mean) * inv.
    std::vector<double> sum_dy(channels);
    std::vector<double> sum_dy_dev(channels);
    channel_deviation_sum(x, dy, mean.data(), d, sum_dy_dev.data(), sum_dy.data());
    const double count = static_cast<double>(n * s);
    // dx = w * inv * (dy - db / count - (x - mean) * inv * dw / count) in training, and
    // dx = w * inv * dy in evaluation (the statistics are constants). Both are
    // dx = a * dy + (x - mean) * b + e with per-channel a, b, e.
    std::vector<float> a(channels);
    std::vector<float> b(channels, 0.0f);
    std::vector<float> e(channels, 0.0f);
    for (std::size_t i = 0; i < channels; ++i) {
        const double inv_i = static_cast<double>(inv[i]);
        dw[i] = static_cast<float>(sum_dy_dev[i] * inv_i);
        db[i] = static_cast<float>(sum_dy[i]);
        a[i] = w[i] * inv[i];
        if (training) {
            e[i] = static_cast<float>(-static_cast<double>(a[i]) * sum_dy[i] / count);
            b[i] = static_cast<float>(-static_cast<double>(a[i]) * inv_i * inv_i * sum_dy_dev[i] /
                                      count);
        }
    }
    const float* const a_p = a.data();
    const float* const b_p = b.data();
    const float* const e_p = e.data();
    const float* const mean_p = mean.data();
    if (s > 1) {
        MENSURFLOW_PARALLEL_FOR(if (worth_threads(n * c * s)) schedule(static))
        for (std::int64_t p = 0; p < n * c; ++p) {
            const std::int64_t ch = p % c;
            const float m = mean_p[ch];
            const float ka = a_p[ch], kb = b_p[ch], ke = e_p[ch];
            const float* MENSURFLOW_RESTRICT xs = x + p * s;
            const float* MENSURFLOW_RESTRICT gs = dy + p * s;
            float* MENSURFLOW_RESTRICT out = dx + p * s;
            for (std::int64_t i = 0; i < s; ++i) {
                out[i] = ka * gs[i] + (xs[i] - m) * kb + ke;
            }
        }
    } else {
        MENSURFLOW_PARALLEL_FOR(if (worth_threads(n * c)) schedule(static))
        for (std::int64_t r = 0; r < n; ++r) {
            const float* MENSURFLOW_RESTRICT xr = x + r * c;
            const float* MENSURFLOW_RESTRICT gr = dy + r * c;
            float* MENSURFLOW_RESTRICT out = dx + r * c;
            for (std::int64_t ch = 0; ch < c; ++ch) {
                out[ch] = a_p[ch] * gr[ch] + (xr[ch] - mean_p[ch]) * b_p[ch] + e_p[ch];
            }
        }
    }
}

} // namespace mensurflow::cpu
