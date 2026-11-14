// 2-D pooling. Every window is clipped to the input, so padding cells are never read. The
// sequential versions take one [h, w] plane of one channel at a time; the parallel versions are
// at the end.
#include "backends/cpu/ops/conv/pooling.hpp"

#include "backends/cpu/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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

// The clipped input columns of every output column.
std::vector<Span> column_windows(const Pool2dShape& s) {
    std::vector<Span> spans(static_cast<std::size_t>(s.w_out));
    for (std::int64_t ow = 0; ow < s.w_out; ++ow) {
        spans[static_cast<std::size_t>(ow)] = window(ow, s.stride_w, s.pad_w, s.kw, s.w);
    }
    return spans;
}

// Output row oh of one plane, with the scan order and rules of the sequential kernel.
void max_pool_row(const float* in, float* out, std::int64_t* out_index, const Span rows,
                  const Span* columns, std::int64_t w, std::int64_t w_out) {
    for (std::int64_t ow = 0; ow < w_out; ++ow) {
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

} // namespace

void max_pool2d_f32_parallel(const float* x, float* y, std::int64_t* indices,
                             const Pool2dShape& s) {
    const std::vector<Span> columns = column_windows(s);
    parallel_for(s.n * s.c * s.h_out, grain_for(pool2d_parallel_cutoff, s.w_out * s.kh * s.kw),
                 [&](std::int64_t r0, std::int64_t r1) {
                     const Pool2dShape shape = s;
                     const Span* cols = columns.data();
                     for (std::int64_t r = r0; r < r1; ++r) {
                         const std::int64_t plane = r / shape.h_out;
                         const std::int64_t oh = r % shape.h_out;
                         const std::int64_t out = r * shape.w_out;
                         max_pool_row(x + plane * shape.h * shape.w, y + out, indices + out,
                                      window(oh, shape.stride_h, shape.pad_h, shape.kh, shape.h),
                                      cols, shape.w, shape.w_out);
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

} // namespace mensurflow::cpu
