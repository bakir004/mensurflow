// Sequential 2-D convolution: im2col + GEMM forward, col2im for the data gradient. One image of
// the batch at a time, so the temporary column matrix is c_in * kh * kw by h_out * w_out floats.
#include "backends/cpu/ops/conv/conv2d.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mensurflow::cpu {

namespace {

std::size_t to_size(std::int64_t value) { return static_cast<std::size_t>(value); }

} // namespace

void im2col_f32(const float* image, float* cols, const Conv2dShape& s) {
    const std::int64_t spatial = s.h_out * s.w_out;
    for (std::int64_t c = 0; c < s.c_in; ++c) {
        for (std::int64_t i = 0; i < s.kh; ++i) {
            for (std::int64_t j = 0; j < s.kw; ++j) {
                float* row = cols + ((c * s.kh + i) * s.kw + j) * spatial;
                for (std::int64_t oh = 0; oh < s.h_out; ++oh) {
                    float* out = row + oh * s.w_out;
                    const std::int64_t ih = oh * s.stride_h - s.pad_h + i;
                    if (ih < 0 || ih >= s.h) {
                        std::fill(out, out + s.w_out, 0.0f);
                        continue;
                    }
                    const float* in = image + (c * s.h + ih) * s.w;
                    for (std::int64_t ow = 0; ow < s.w_out; ++ow) {
                        const std::int64_t iw = ow * s.stride_w - s.pad_w + j;
                        out[ow] = iw >= 0 && iw < s.w ? in[iw] : 0.0f;
                    }
                }
            }
        }
    }
}

void col2im_f32(const float* cols, float* image, const Conv2dShape& s) {
    const std::int64_t spatial = s.h_out * s.w_out;
    for (std::int64_t c = 0; c < s.c_in; ++c) {
        for (std::int64_t i = 0; i < s.kh; ++i) {
            for (std::int64_t j = 0; j < s.kw; ++j) {
                const float* row = cols + ((c * s.kh + i) * s.kw + j) * spatial;
                for (std::int64_t oh = 0; oh < s.h_out; ++oh) {
                    const std::int64_t ih = oh * s.stride_h - s.pad_h + i;
                    if (ih < 0 || ih >= s.h) {
                        continue;
                    }
                    const float* in = row + oh * s.w_out;
                    float* out = image + (c * s.h + ih) * s.w;
                    for (std::int64_t ow = 0; ow < s.w_out; ++ow) {
                        const std::int64_t iw = ow * s.stride_w - s.pad_w + j;
                        if (iw >= 0 && iw < s.w) {
                            out[iw] += in[ow];
                        }
                    }
                }
            }
        }
    }
}

void conv2d_f32_sequential(const float* x, const float* weight, const float* bias, float* y,
                           const Conv2dShape& s) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    if (s.n == 0 || s.c_out == 0 || spatial == 0) {
        return;
    }
    std::vector<float> cols(to_size(k * spatial));
    for (std::int64_t b = 0; b < s.n; ++b) {
        im2col_f32(x + b * s.c_in * s.h * s.w, cols.data(), s);
        float* yb = y + b * s.c_out * spatial;
        // y_b[c_out, spatial] = weight[c_out, k] * cols[k, spatial]; k == 0 writes zeros.
        matmul_f32_packed_strided(weight, k, 1, cols.data(), spatial, 1, yb, spatial, s.c_out,
                                  spatial, k);
        if (bias != nullptr) {
            for (std::int64_t co = 0; co < s.c_out; ++co) {
                float* plane = yb + co * spatial;
                for (std::int64_t p = 0; p < spatial; ++p) {
                    plane[p] += bias[co];
                }
            }
        }
    }
}

void conv2d_backward_data_f32_sequential(const float* grad_y, const float* weight, float* grad_x,
                                         const Conv2dShape& s) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    const std::int64_t image = s.c_in * s.h * s.w;
    std::fill(grad_x, grad_x + s.n * image, 0.0f);
    if (s.n == 0 || k == 0 || spatial == 0) {
        return;
    }
    std::vector<float> cols(to_size(k * spatial));
    for (std::int64_t b = 0; b < s.n; ++b) {
        // cols[k, spatial] = weight^T[k, c_out] * grad_y_b[c_out, spatial]. Element (r, co) of
        // weight^T is weight[co * k + r]: row stride 1, column stride k. c_out == 0 writes zeros.
        matmul_f32_packed_strided(weight, 1, k, grad_y + b * s.c_out * spatial, spatial, 1,
                                  cols.data(), spatial, k, spatial, s.c_out);
        col2im_f32(cols.data(), grad_x + b * image, s);
    }
}

void conv2d_backward_weight_f32_sequential(const float* grad_y, const float* x, float* grad_weight,
                                           const Conv2dShape& s) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    std::fill(grad_weight, grad_weight + s.c_out * k, 0.0f);
    if (s.n == 0 || s.c_out == 0 || k == 0 || spatial == 0) {
        return;
    }
    std::vector<float> cols(to_size(k * spatial));
    std::vector<float> partial(to_size(s.c_out * k));
    for (std::int64_t b = 0; b < s.n; ++b) {
        im2col_f32(x + b * s.c_in * s.h * s.w, cols.data(), s);
        // partial[c_out, k] = grad_y_b[c_out, spatial] * cols^T[spatial, k]. Element (p, r) of
        // cols^T is cols[r * spatial + p]: row stride 1, column stride spatial.
        matmul_f32_packed_strided(grad_y + b * s.c_out * spatial, spatial, 1, cols.data(), 1,
                                  spatial, partial.data(), k, s.c_out, k, spatial);
        for (std::size_t i = 0; i < partial.size(); ++i) {
            grad_weight[i] += partial[i];
        }
    }
}

void conv2d_backward_bias_f32_sequential(const float* grad_y, float* grad_bias,
                                         const Conv2dShape& s) {
    const std::int64_t spatial = s.h_out * s.w_out;
    for (std::int64_t co = 0; co < s.c_out; ++co) {
        double sum = 0.0;
        for (std::int64_t b = 0; b < s.n; ++b) {
            const float* plane = grad_y + (b * s.c_out + co) * spatial;
            for (std::int64_t p = 0; p < spatial; ++p) {
                sum += static_cast<double>(plane[p]);
            }
        }
        grad_bias[co] = static_cast<float>(sum);
    }
}

} // namespace mensurflow::cpu
