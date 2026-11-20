// 2-D pooling. Every window is clipped to the input, so padding cells are never read. The
// sequential versions take one [h, w] plane of one channel at a time; the parallel versions are
// at the end.
#include "backends/cpu/ops/conv/pooling.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define MENSURFLOW_POOL_AVX2 1
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#define MENSURFLOW_POOL_TARGET
#else
#define MENSURFLOW_POOL_TARGET __attribute__((target("avx2")))
#endif

namespace mensurflow::cpu {

namespace {

// The input rows [begin, end) of output row o, clipped to [0, size).
struct Span {
    std::int64_t begin, end;
};

Span window(std::int64_t o, std::int64_t stride, std::int64_t pad, std::int64_t kernel,
            std::int64_t size) {
    const std::int64_t start = o * stride - pad;
    return {std::max<std::int64_t>(start, 0), std::min(start + kernel, size)};
}

} // namespace

void max_pool2d_f32_sequential(const float* x, float* y, std::int64_t* indices,
                               const Pool2dShape& s) {
    const std::int64_t planes = s.n * s.c;
    for (std::int64_t plane = 0; plane < planes; ++plane) {
        const float* in = x + plane * s.h * s.w;
        float* out = y + plane * s.h_out * s.w_out;
        std::int64_t* out_index = indices + plane * s.h_out * s.w_out;
        for (std::int64_t oh = 0; oh < s.h_out; ++oh) {
            const Span rows = window(oh, s.stride_h, s.pad_h, s.kh, s.h);
            for (std::int64_t ow = 0; ow < s.w_out; ++ow) {
                const Span cols = window(ow, s.stride_w, s.pad_w, s.kw, s.w);
                // Row-major scan; a later cell wins only if it is strictly larger or the first
                // NaN, and nothing beats a NaN.
                std::int64_t best = rows.begin * s.w + cols.begin;
                for (std::int64_t ih = rows.begin; ih < rows.end && !std::isnan(in[best]); ++ih) {
                    for (std::int64_t iw = cols.begin; iw < cols.end; ++iw) {
                        const float value = in[ih * s.w + iw];
                        if (std::isnan(value)) {
                            best = ih * s.w + iw;
                            break;
                        }
                        if (value > in[best]) {
                            best = ih * s.w + iw;
                        }
                    }
                }
                out[oh * s.w_out + ow] = in[best];
                out_index[oh * s.w_out + ow] = best;
            }
        }
    }
}

void max_pool2d_backward_f32_sequential(const float* grad_y, const std::int64_t* indices,
                                        float* grad_x, const Pool2dShape& s) {
    const std::int64_t planes = s.n * s.c;
    const std::int64_t outputs = s.h_out * s.w_out;
    std::fill(grad_x, grad_x + planes * s.h * s.w, 0.0f);
    for (std::int64_t plane = 0; plane < planes; ++plane) {
        float* out = grad_x + plane * s.h * s.w;
        for (std::int64_t i = plane * outputs; i < (plane + 1) * outputs; ++i) {
            out[indices[i]] += grad_y[i];
        }
    }
}

void avg_pool2d_f32_sequential(const float* x, float* y, const Pool2dShape& s) {
    const std::int64_t planes = s.n * s.c;
    const double divisor = static_cast<double>(s.kh * s.kw);
    for (std::int64_t plane = 0; plane < planes; ++plane) {
        const float* in = x + plane * s.h * s.w;
        float* out = y + plane * s.h_out * s.w_out;
        for (std::int64_t oh = 0; oh < s.h_out; ++oh) {
            const Span rows = window(oh, s.stride_h, s.pad_h, s.kh, s.h);
            for (std::int64_t ow = 0; ow < s.w_out; ++ow) {
                const Span cols = window(ow, s.stride_w, s.pad_w, s.kw, s.w);
                double sum = 0.0;
                for (std::int64_t ih = rows.begin; ih < rows.end; ++ih) {
                    for (std::int64_t iw = cols.begin; iw < cols.end; ++iw) {
                        sum += static_cast<double>(in[ih * s.w + iw]);
                    }
                }
                out[oh * s.w_out + ow] = static_cast<float>(sum / divisor);
            }
        }
    }
}

void avg_pool2d_backward_f32_sequential(const float* grad_y, float* grad_x, const Pool2dShape& s) {
    const std::int64_t planes = s.n * s.c;
    const float scale = 1.0f / static_cast<float>(s.kh * s.kw);
    std::fill(grad_x, grad_x + planes * s.h * s.w, 0.0f);
    for (std::int64_t plane = 0; plane < planes; ++plane) {
        const float* in = grad_y + plane * s.h_out * s.w_out;
        float* out = grad_x + plane * s.h * s.w;
        for (std::int64_t oh = 0; oh < s.h_out; ++oh) {
            const Span rows = window(oh, s.stride_h, s.pad_h, s.kh, s.h);
            for (std::int64_t ow = 0; ow < s.w_out; ++ow) {
                const Span cols = window(ow, s.stride_w, s.pad_w, s.kw, s.w);
                const float share = in[oh * s.w_out + ow] * scale;
                for (std::int64_t ih = rows.begin; ih < rows.end; ++ih) {
                    for (std::int64_t iw = cols.begin; iw < cols.end; ++iw) {
                        out[ih * s.w + iw] += share;
                    }
                }
            }
        }
    }
}

// ---- Parallel versions ---------------------------------------------------------------------
//
// Written for the compiler as much as for the reader: every value a loop needs is copied into a
// local first. MSVC does no type-based alias analysis, so a pointer or a size read through a
// reference (or a lambda's captures) would be reloaded after every float store, and that alone
// made the first version of these loops 2x slower than the sequential kernels on one thread. The
// window of each output column (and, for the gather, the outputs covering each input column) are
// computed once per call instead of with two divisions per element. NaN is tested as v != v, which
// MSVC compiles to one compare (std::isnan may be a call).

namespace {

// The outputs [begin, end) whose window covers input position i: o * stride - pad <= i <=
// o * stride - pad + kernel - 1, clipped to [0, n_out).
Span covering(std::int64_t i, std::int64_t stride, std::int64_t pad, std::int64_t kernel,
              std::int64_t n_out) {
    const std::int64_t low = i + pad - kernel + 1;
    const std::int64_t begin = low <= 0 ? 0 : (low + stride - 1) / stride;
    const std::int64_t end = std::min(n_out, (i + pad) / stride + 1);
    return {begin, std::max(begin, end)};
}

// The clipped input columns of every output column.
std::vector<Span> column_windows(const Pool2dShape& s) {
    std::vector<Span> spans(static_cast<std::size_t>(s.w_out));
    for (std::int64_t ow = 0; ow < s.w_out; ++ow) {
        spans[static_cast<std::size_t>(ow)] = window(ow, s.stride_w, s.pad_w, s.kw, s.w);
    }
    return spans;
}

// The output columns that cover every input column.
std::vector<Span> covering_columns(const Pool2dShape& s) {
    std::vector<Span> spans(static_cast<std::size_t>(s.w));
    for (std::int64_t iw = 0; iw < s.w; ++iw) {
        spans[static_cast<std::size_t>(iw)] = covering(iw, s.stride_w, s.pad_w, s.kw, s.w_out);
    }
    return spans;
}

// Output columns [ow_begin, ow_end) of output row oh of one plane, with the scan order and rules
// of the sequential kernel.
void max_pool_row(const float* in, float* out, std::int64_t* out_index, const Span rows,
                  const Span* columns, std::int64_t w, std::int64_t ow_begin,
                  std::int64_t ow_end) {
    for (std::int64_t ow = ow_begin; ow < ow_end; ++ow) {
        const Span cols = columns[ow];
        std::int64_t best = rows.begin * w + cols.begin;
        float best_value = in[best];
        for (std::int64_t ih = rows.begin; ih < rows.end && best_value == best_value; ++ih) {
            const float* line = in + ih * w;
            for (std::int64_t iw = cols.begin; iw < cols.end; ++iw) {
                const float value = line[iw];
                if (value != value) { // NaN: wins, and nothing beats it
                    best = ih * w + iw;
                    best_value = value;
                    break;
                }
                if (value > best_value) {
                    best = ih * w + iw;
                    best_value = value;
                }
            }
        }
        out[ow] = best_value;
        out_index[ow] = best;
    }
}

// ---- AVX2 max pooling: 8 output columns of one row at a time --------------------------------
//
// Each lane scans its window in the sequential order and keeps (best value, best index). A cell
// replaces the best if it is strictly greater (an ordered compare: false when either is NaN, so
// ties keep the earlier cell and nothing replaces a NaN) or if it is a NaN while the best is not
// (the first NaN wins). This is the scalar rule lane by lane, so values (bits) and indices are
// identical to the sequential kernel. The windows of the 8 columns must lie inside the input
// columns (no padding) and every load must stay inside the row; MaxPoolVector picks those columns.

// Which output columns of a row run 8 at a time: blocks starting at first, first + 8, ... and one
// last block starting at last (overlapping the previous one), all inside [first, last + 8).
struct MaxPoolVector {
    bool enabled = false;
    std::int64_t first = 0, last = 0;
};

MaxPoolVector max_pool_vector_plan(const Pool2dShape& s) {
    MaxPoolVector plan;
#ifdef MENSURFLOW_POOL_AVX2
    if (!packed_gemm_supported() || s.w_out < 8 ||
        s.h * s.w > std::numeric_limits<std::int32_t>::max()) {
        return plan;
    }
    // The furthest column a block reads, counted from its first window's first column: the last
    // window cell of lane 7 (load_columns reads nothing beyond it).
    const std::int64_t reach = (s.kw - 1) + 7 * s.stride_w;
    plan.first = (s.pad_w + s.stride_w - 1) / s.stride_w; // first window that starts at column >= 0
    const std::int64_t room = s.w - 1 - reach + s.pad_w;  // ow0 * stride <= room
    if (room < 0) {
        return plan;
    }
    plan.last = std::min(room / s.stride_w, s.w_out - 8);
    plan.enabled = plan.last >= plan.first;
#else
    (void)s;
#endif
    return plan;
}

#ifdef MENSURFLOW_POOL_AVX2

// 8 floats at p[0], p[stride], ..., p[7 * stride].
template <int Stride>
MENSURFLOW_POOL_TARGET inline __m256 load_columns(const float* p, __m256i gather_offsets) {
    if constexpr (Stride == 1) {
        (void)gather_offsets;
        return _mm256_loadu_ps(p);
    } else if constexpr (Stride == 2) {
        (void)gather_offsets;
        // a = p[0..7], b = p[7..14] (so nothing past p[14] is read). a0 a2 b1 b3 | a4 a6 b5 b7 =
        // p0 p2 p8 p10 | p4 p6 p12 p14, then the 64-bit pairs reordered to p0 p2 ... p14.
        const __m256 even = _mm256_shuffle_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(p + 7),
                                              _MM_SHUFFLE(3, 1, 2, 0));
        return _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(even), _MM_SHUFFLE(3, 1, 2, 0)));
    } else {
        return _mm256_i32gather_ps(p, gather_offsets, 4);
    }
}

// Output columns [ow0, ow0 + 8) of one output row; their windows start at input column c0 =
// ow0 * stride - pad >= 0 (lane l at c0 + l * stride) and span kw columns and the rows `rows`.
template <int Stride>
MENSURFLOW_POOL_TARGET void max_pool_block8(const float* in, float* out, std::int64_t* out_index,
                                            const Span rows, std::int64_t w, std::int64_t c0,
                                            std::int64_t stride, std::int64_t kw) {
    const __m256i offsets = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(static_cast<int>(stride)));
    const std::int64_t start = rows.begin * w + c0;
    __m256 best = load_columns<Stride>(in + start, offsets);
    __m256i best_index = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(start)), offsets);
    __m256 best_is_nan = _mm256_cmp_ps(best, best, _CMP_UNORD_Q);
    std::int64_t j0 = 1; // the first cell is already the best
    for (std::int64_t ih = rows.begin; ih < rows.end; ++ih) {
        const std::int64_t line = ih * w + c0;
        for (std::int64_t j = j0; j < kw; ++j) {
            const __m256 value = load_columns<Stride>(in + line + j, offsets);
            const __m256 is_nan = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
            const __m256 take = _mm256_or_ps(_mm256_cmp_ps(value, best, _CMP_GT_OQ),
                                             _mm256_andnot_ps(best_is_nan, is_nan));
            const __m256i index =
                _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(line + j)), offsets);
            best = _mm256_blendv_ps(best, value, take);
            best_index = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(best_index), _mm256_castsi256_ps(index), take));
            best_is_nan = _mm256_or_ps(best_is_nan, is_nan);
        }
        j0 = 0;
    }
    _mm256_storeu_ps(out, best);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_index),
                        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(best_index)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_index + 4),
                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(best_index, 1)));
}

template <int Stride>
MENSURFLOW_POOL_TARGET void max_pool_row_avx2(const float* in, float* out,
                                              std::int64_t* out_index, const Span rows,
                                              const Span* columns, const MaxPoolVector& plan,
                                              std::int64_t w, std::int64_t w_out,
                                              std::int64_t stride, std::int64_t pad,
                                              std::int64_t kw) {
    max_pool_row(in, out, out_index, rows, columns, w, 0, plan.first);
    std::int64_t ow0 = plan.first;
    for (; ow0 <= plan.last; ow0 += 8) {
        max_pool_block8<Stride>(in, out + ow0, out_index + ow0, rows, w, ow0 * stride - pad,
                                stride, kw);
    }
    if (ow0 - 8 < plan.last) { // the last block did not reach plan.last + 8: one overlapping block
        max_pool_block8<Stride>(in, out + plan.last, out_index + plan.last, rows, w,
                                plan.last * stride - pad, stride, kw);
    }
    max_pool_row(in, out, out_index, rows, columns, w, plan.last + 8, w_out);
}

#endif // MENSURFLOW_POOL_AVX2

void avg_pool_row(const float* in, float* out, const Span rows, const Span* columns,
                  std::int64_t w, std::int64_t w_out, double divisor) {
    for (std::int64_t ow = 0; ow < w_out; ++ow) {
        const Span cols = columns[ow];
        double sum = 0.0;
        for (std::int64_t ih = rows.begin; ih < rows.end; ++ih) {
            const float* line = in + ih * w;
            for (std::int64_t iw = cols.begin; iw < cols.end; ++iw) {
                sum += static_cast<double>(line[iw]);
            }
        }
        out[ow] = static_cast<float>(sum / divisor);
    }
}

} // namespace

void max_pool2d_f32_parallel(const float* x, float* y, std::int64_t* indices,
                             const Pool2dShape& s) {
    const std::vector<Span> columns = column_windows(s);
    const MaxPoolVector plan = max_pool_vector_plan(s);
    parallel_for(s.n * s.c * s.h_out, grain_for(pool2d_parallel_cutoff, s.w_out * s.kh * s.kw),
                 [&](std::int64_t r0, std::int64_t r1) {
                     const Pool2dShape shape = s;
                     const MaxPoolVector vec = plan;
                     const Span* cols = columns.data();
                     for (std::int64_t r = r0; r < r1; ++r) {
                         const std::int64_t plane = r / shape.h_out;
                         const std::int64_t oh = r % shape.h_out;
                         const std::int64_t out = r * shape.w_out;
                         const float* in = x + plane * shape.h * shape.w;
                         const Span rows =
                             window(oh, shape.stride_h, shape.pad_h, shape.kh, shape.h);
#ifdef MENSURFLOW_POOL_AVX2
                         if (vec.enabled) {
                             if (shape.stride_w == 1) {
                                 max_pool_row_avx2<1>(in, y + out, indices + out, rows, cols, vec,
                                                      shape.w, shape.w_out, 1, shape.pad_w,
                                                      shape.kw);
                             } else if (shape.stride_w == 2) {
                                 max_pool_row_avx2<2>(in, y + out, indices + out, rows, cols, vec,
                                                      shape.w, shape.w_out, 2, shape.pad_w,
                                                      shape.kw);
                             } else {
                                 max_pool_row_avx2<0>(in, y + out, indices + out, rows, cols, vec,
                                                      shape.w, shape.w_out, shape.stride_w,
                                                      shape.pad_w, shape.kw);
                             }
                             continue;
                         }
#else
                         (void)vec;
#endif
                         max_pool_row(in, y + out, indices + out, rows, cols, shape.w, 0,
                                      shape.w_out);
                     }
                 });
}

void max_pool2d_backward_f32_parallel(const float* grad_y, const std::int64_t* indices,
                                      float* grad_x, const Pool2dShape& s) {
    const std::int64_t outputs = s.h_out * s.w_out;
    const std::int64_t cells = s.h * s.w;
    parallel_for(s.n * s.c, grain_for(pool2d_parallel_cutoff, cells + outputs),
                 [&](std::int64_t p0, std::int64_t p1) {
                     const std::int64_t n_out = outputs, n_in = cells;
                     for (std::int64_t plane = p0; plane < p1; ++plane) {
                         float* out = grad_x + plane * n_in;
                         const float* gy = grad_y + plane * n_out;
                         const std::int64_t* index = indices + plane * n_out;
                         std::fill(out, out + n_in, 0.0f);
                         for (std::int64_t i = 0; i < n_out; ++i) {
                             out[index[i]] += gy[i];
                         }
                     }
                 });
}

void avg_pool2d_f32_parallel(const float* x, float* y, const Pool2dShape& s) {
    const std::vector<Span> columns = column_windows(s);
    parallel_for(s.n * s.c * s.h_out, grain_for(pool2d_parallel_cutoff, s.w_out * s.kh * s.kw),
                 [&](std::int64_t r0, std::int64_t r1) {
                     const Pool2dShape shape = s;
                     const Span* cols = columns.data();
                     const double divisor = static_cast<double>(shape.kh * shape.kw);
                     for (std::int64_t r = r0; r < r1; ++r) {
                         const std::int64_t plane = r / shape.h_out;
                         const std::int64_t oh = r % shape.h_out;
                         avg_pool_row(x + plane * shape.h * shape.w, y + r * shape.w_out,
                                      window(oh, shape.stride_h, shape.pad_h, shape.kh, shape.h),
                                      cols, shape.w, shape.w_out, divisor);
                     }
                 });
}

void avg_pool2d_backward_f32_parallel(const float* grad_y, float* grad_x, const Pool2dShape& s) {
    const std::vector<Span> covering_cols = covering_columns(s);
    // Each input cell gathers about (kh / stride_h) * (kw / stride_w) windows.
    const std::int64_t cost =
        s.w * std::max<std::int64_t>(1, (s.kh / s.stride_h) * (s.kw / s.stride_w));
    parallel_for(s.n * s.c * s.h, grain_for(pool2d_parallel_cutoff, cost),
                 [&](std::int64_t r0, std::int64_t r1) {
                     const Pool2dShape shape = s;
                     const Span* cover = covering_cols.data();
                     const float scale = 1.0f / static_cast<float>(shape.kh * shape.kw);
                     const std::int64_t outputs = shape.h_out * shape.w_out;
                     for (std::int64_t r = r0; r < r1; ++r) {
                         const std::int64_t plane = r / shape.h;
                         const std::int64_t ih = r % shape.h;
                         const float* in = grad_y + plane * outputs;
                         float* out = grad_x + r * shape.w;
                         const Span out_rows =
                             covering(ih, shape.stride_h, shape.pad_h, shape.kh, shape.h_out);
                         for (std::int64_t iw = 0; iw < shape.w; ++iw) {
                             const Span out_cols = cover[iw];
                             // From +0, share by share in increasing (oh, ow): the order of the
                             // sequential scatter, so the bits agree.
                             float sum = 0.0f;
                             for (std::int64_t oh = out_rows.begin; oh < out_rows.end; ++oh) {
                                 const float* line = in + oh * shape.w_out;
                                 for (std::int64_t ow = out_cols.begin; ow < out_cols.end; ++ow) {
                                     sum += line[ow] * scale;
                                 }
                             }
                             out[iw] = sum;
                         }
                     }
                 });
}

} // namespace mensurflow::cpu
