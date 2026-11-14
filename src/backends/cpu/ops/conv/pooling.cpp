// Sequential 2-D pooling: one [h, w] plane of one channel at a time, every window clipped to the
// input, so padding cells are never read.
#include "backends/cpu/ops/conv/pooling.hpp"

#include <algorithm>
#include <cmath>

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

} // namespace mensurflow::cpu
