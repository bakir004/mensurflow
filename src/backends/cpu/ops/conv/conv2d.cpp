// 2-D convolution: im2col + GEMM forward, col2im for the data gradient. The sequential versions
// take one image of the batch at a time, so their temporary column matrix is c_in * kh * kw by
// h_out * w_out floats; the parallel versions are further down.
#include "backends/cpu/ops/conv/conv2d.hpp"

#include "backends/cpu/affinity.hpp"
#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
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

// ---- Parallel versions ---------------------------------------------------------------------

namespace {

// Elementwise and copy work below this many floats stays on one thread.
constexpr std::int64_t copy_cutoff = std::int64_t{1} << 15;

// Scratch memory that lives as long as the thread and only grows: `slot` picks one of a few
// independent buffers. A fresh std::vector per call costs a single-threaded zero fill of the whole
// column matrix plus a page fault for every 4 KB of it the first time it is written (about 1 ms of
// a 1.7 ms ResNet layer at batch 1). The contents are uninitialized.
float* workspace(int slot, std::int64_t floats) {
    struct Buffer {
        std::unique_ptr<float[]> data;
        std::int64_t size = 0;
    };
    thread_local Buffer buffers[3];
    Buffer& buffer = buffers[slot];
    if (buffer.size < floats) {
        buffer.data.reset(new float[static_cast<std::size_t>(floats)]);
        buffer.size = floats;
    }
    return buffer.data.get();
}

// The outputs o in [0, n_out) whose input o * stride - pad + offset lies in [0, n_in), as a range
// [begin, end) (empty when begin >= end).
struct Range {
    std::int64_t begin, end;
};

Range valid_outputs(std::int64_t n_out, std::int64_t stride, std::int64_t pad,
                    std::int64_t offset, std::int64_t n_in) {
    const std::int64_t low = pad - offset; // o * stride >= low
    const std::int64_t begin = low <= 0 ? 0 : (low + stride - 1) / stride;
    const std::int64_t high = n_in - 1 + pad - offset; // o * stride <= high
    const std::int64_t end = high < 0 ? 0 : std::min(n_out, high / stride + 1);
    return {begin, std::max(begin, end)};
}

// Rows [r0, r1) of one image's column matrix (row r = (c * kh + i) * kw + j), written with row
// stride ld. The valid columns of each output row are a plain (or strided) copy, the rest zeros.
void im2col_rows(const float* image, float* cols, std::int64_t ld, const Conv2dShape& s,
                 std::int64_t r0, std::int64_t r1) {
    for (std::int64_t r = r0; r < r1; ++r) {
        const std::int64_t c = r / (s.kh * s.kw);
        const std::int64_t i = (r / s.kw) % s.kh;
        const std::int64_t j = r % s.kw;
        const Range cols_ok = valid_outputs(s.w_out, s.stride_w, s.pad_w, j, s.w);
        float* row = cols + r * ld;
        for (std::int64_t oh = 0; oh < s.h_out; ++oh) {
            float* out = row + oh * s.w_out;
            const std::int64_t ih = oh * s.stride_h - s.pad_h + i;
            if (ih < 0 || ih >= s.h || cols_ok.begin >= cols_ok.end) {
                std::fill(out, out + s.w_out, 0.0f);
                continue;
            }
            const float* in = image + (c * s.h + ih) * s.w - s.pad_w + j;
            std::fill(out, out + cols_ok.begin, 0.0f);
            if (s.stride_w == 1) {
                std::copy(in + cols_ok.begin, in + cols_ok.end, out + cols_ok.begin);
            } else {
                for (std::int64_t ow = cols_ok.begin; ow < cols_ok.end; ++ow) {
                    out[ow] = in[ow * s.stride_w];
                }
            }
            std::fill(out + cols_ok.end, out + s.w_out, 0.0f);
        }
    }
}

// col2im for input channel c of one image: adds rows c * kh * kw .. (c + 1) * kh * kw - 1 of the
// column matrix (row stride ld) into plane[h, w]. Adds in the same order as col2im_f32.
void col2im_channel(const float* cols, std::int64_t ld, float* plane, std::int64_t c,
                    const Conv2dShape& s) {
    for (std::int64_t i = 0; i < s.kh; ++i) {
        for (std::int64_t j = 0; j < s.kw; ++j) {
            const float* row = cols + ((c * s.kh + i) * s.kw + j) * ld;
            const Range cols_ok = valid_outputs(s.w_out, s.stride_w, s.pad_w, j, s.w);
            for (std::int64_t oh = 0; oh < s.h_out; ++oh) {
                const std::int64_t ih = oh * s.stride_h - s.pad_h + i;
                if (ih < 0 || ih >= s.h) {
                    continue;
                }
                const float* in = row + oh * s.w_out;
                float* out = plane + ih * s.w - s.pad_w + j;
                for (std::int64_t ow = cols_ok.begin; ow < cols_ok.end; ++ow) {
                    out[ow * s.stride_w] += in[ow];
                }
            }
        }
    }
}

// Images per chunk of the folded strategy: as many as fit in the column budget, at least one.
std::int64_t images_per_chunk(const Conv2dShape& s) {
    const std::int64_t per_image =
        std::max<std::int64_t>(1, s.c_in * s.kh * s.kw * s.h_out * s.w_out);
    return std::clamp<std::int64_t>(conv2d_column_budget / per_image, 1,
                                    std::max<std::int64_t>(s.n, 1));
}

std::int64_t multiply_adds(const Conv2dShape& s) {
    return s.n * s.c_out * s.c_in * s.kh * s.kw * s.h_out * s.w_out;
}

// im2col of images [b0, b0 + count) into cols[K, count * spatial], one task per (image, row).
void im2col_chunk(const float* x, float* cols, const Conv2dShape& s, std::int64_t b0,
                  std::int64_t count) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    const std::int64_t ld = count * spatial;
    parallel_for(count * k, grain_for(copy_cutoff, spatial), [&](std::int64_t t0, std::int64_t t1) {
        for (std::int64_t t = t0; t < t1; ++t) {
            const std::int64_t b = t / k;
            const std::int64_t r = t % k;
            im2col_rows(x + (b0 + b) * s.c_in * s.h * s.w, cols + b * spatial, ld, s, r, r + 1);
        }
    });
}

// grad_y of images [b0, b0 + count) as one matrix [c_out, count * spatial]: grad_y itself when
// count is 1, otherwise gathered into `buffer`.
const float* fold_grad_y(const float* grad_y, float* buffer, const Conv2dShape& s,
                         std::int64_t b0, std::int64_t count) {
    const std::int64_t spatial = s.h_out * s.w_out;
    const float* first = grad_y + b0 * s.c_out * spatial;
    if (count == 1) {
        return first;
    }
    const std::int64_t ld = count * spatial;
    parallel_for(count * s.c_out, grain_for(copy_cutoff, spatial),
                 [&](std::int64_t t0, std::int64_t t1) {
                     for (std::int64_t t = t0; t < t1; ++t) {
                         const std::int64_t b = t / s.c_out;
                         const std::int64_t co = t % s.c_out;
                         const float* from = first + (b * s.c_out + co) * spatial;
                         std::copy(from, from + spatial, buffer + co * ld + b * spatial);
                     }
                 });
    return buffer;
}

// A 1x1 kernel with stride 1 and no padding: an image's column matrix is the image itself
// ([c_in, h * w]), so no im2col is needed and the products read x (and write grad_x) in place.
bool is_pointwise(const Conv2dShape& s) {
    return s.kh == 1 && s.kw == 1 && s.stride_h == 1 && s.stride_w == 1 && s.pad_h == 0 &&
           s.pad_w == 0;
}

// Threads for whole images per thread: at most one per physical core. Two single-threaded products
// on the two hardware threads of one core run at about half speed each, so 24 threads gain nothing
// over 12 on a 12-core CPU and spread a batch of 32 less evenly (2 images on some threads, 1 on
// others); measured at batch 32 the 1x1 layers ran 1.0-1.7x faster on 12 threads than on 24.
int image_threads() {
    const int cores = physical_core_count();
    return cores > 0 ? std::min(max_threads(), cores) : max_threads();
}

// y_b += bias for images [b0, b0 + count), one task per (image, channel).
void add_bias(float* y, const float* bias, const Conv2dShape& s, std::int64_t b0,
              std::int64_t count) {
    const std::int64_t spatial = s.h_out * s.w_out;
    parallel_for(count * s.c_out, grain_for(copy_cutoff, spatial),
                 [&](std::int64_t t0, std::int64_t t1) {
                     for (std::int64_t t = t0; t < t1; ++t) {
                         float* plane = y + (b0 * s.c_out + t) * spatial;
                         const float add = bias[t % s.c_out];
                         for (std::int64_t p = 0; p < spatial; ++p) {
                             plane[p] += add;
                         }
                     }
                 });
}

void conv2d_folded(const float* x, const float* weight, const float* bias, float* y,
                   const Conv2dShape& s) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    if (is_pointwise(s)) {
        // One product per image, each on all threads, reading the image in place.
        for (std::int64_t b = 0; b < s.n; ++b) {
            matmul_f32_packed_parallel_strided(weight, k, 1, x + b * k * spatial, spatial, 1,
                                               y + b * s.c_out * spatial, spatial, s.c_out,
                                               spatial, k);
        }
        if (bias != nullptr) {
            add_bias(y, bias, s, 0, s.n);
        }
        return;
    }
    const std::int64_t chunk = images_per_chunk(s);
    float* cols = workspace(0, std::max<std::int64_t>(k, 1) * chunk * spatial);
    float* folded = chunk > 1 ? workspace(1, s.c_out * chunk * spatial) : nullptr;
    for (std::int64_t b0 = 0; b0 < s.n; b0 += chunk) {
        const std::int64_t count = std::min(chunk, s.n - b0);
        const std::int64_t ld = count * spatial;
        im2col_chunk(x, cols, s, b0, count);
        float* yb = y + b0 * s.c_out * spatial;
        if (count == 1) {
            matmul_f32_packed_parallel_strided(weight, k, 1, cols, ld, 1, yb, ld, s.c_out, ld, k);
            if (bias != nullptr) {
                add_bias(y, bias, s, b0, 1);
            }
            continue;
        }
        // [c_out, count * spatial] = weight[c_out, k] * cols[k, count * spatial], then into NCHW
        // order plus the bias, one task per (image, channel).
        matmul_f32_packed_parallel_strided(weight, k, 1, cols, ld, 1, folded, ld, s.c_out, ld, k);
        parallel_for(count * s.c_out, grain_for(copy_cutoff, spatial),
                     [&](std::int64_t t0, std::int64_t t1) {
                         for (std::int64_t t = t0; t < t1; ++t) {
                             const std::int64_t b = t / s.c_out;
                             const std::int64_t co = t % s.c_out;
                             const float* from = folded + co * ld + b * spatial;
                             float* to = yb + (b * s.c_out + co) * spatial;
                             const float add = bias != nullptr ? bias[co] : 0.0f;
                             for (std::int64_t p = 0; p < spatial; ++p) {
                                 to[p] = from[p] + add;
                             }
                         }
                     });
    }
}

// Whole images per task; `grain` images at least per thread (s.n keeps everything on the calling
// thread, which is how small convolutions run: the same code, without the threads).
void conv2d_image_per_thread(const float* x, const float* weight, const float* bias, float* y,
                             const Conv2dShape& s, std::int64_t grain = 1) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    const bool pointwise = is_pointwise(s);
    parallel_for_upto(image_threads(), s.n, grain, [&](std::int64_t b0, std::int64_t b1) {
        float* cols = pointwise ? nullptr : workspace(0, std::max<std::int64_t>(k, 1) * spatial);
        for (std::int64_t b = b0; b < b1; ++b) {
            const float* image = x + b * s.c_in * s.h * s.w;
            if (!pointwise) {
                im2col_rows(image, cols, spatial, s, 0, k);
            }
            float* yb = y + b * s.c_out * spatial;
            matmul_f32_packed_strided(weight, k, 1, pointwise ? image : cols, spatial, 1, yb,
                                      spatial, s.c_out, spatial, k);
            if (bias != nullptr) {
                for (std::int64_t co = 0; co < s.c_out; ++co) {
                    float* plane = yb + co * spatial;
                    for (std::int64_t p = 0; p < spatial; ++p) {
                        plane[p] += bias[co];
                    }
                }
            }
        }
    });
}

// The data gradient one image per task: weight^T * grad_y_b on one thread, then col2im (straight
// into grad_x for a pointwise kernel).
void backward_data_per_image(const float* grad_y, const float* weight, float* grad_x,
                             const Conv2dShape& s, std::int64_t grain = 1) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    const std::int64_t image = s.c_in * s.h * s.w;
    const bool pointwise = is_pointwise(s);
    parallel_for_upto(image_threads(), s.n, grain, [&](std::int64_t b0, std::int64_t b1) {
        float* cols = pointwise ? nullptr : workspace(0, k * spatial);
        for (std::int64_t b = b0; b < b1; ++b) {
            float* out = grad_x + b * image;
            matmul_f32_packed_strided(weight, 1, k, grad_y + b * s.c_out * spatial, spatial, 1,
                                      pointwise ? out : cols, spatial, k, spatial, s.c_out);
            if (!pointwise) {
                std::fill(out, out + image, 0.0f);
                for (std::int64_t c = 0; c < s.c_in; ++c) {
                    col2im_channel(cols, spatial, out + c * s.h * s.w, c, s);
                }
            }
        }
    });
}

void backward_data_folded(const float* grad_y, const float* weight, float* grad_x,
                          const Conv2dShape& s) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    const std::int64_t plane = s.h * s.w;
    if (is_pointwise(s)) {
        for (std::int64_t b = 0; b < s.n; ++b) {
            matmul_f32_packed_parallel_strided(weight, 1, k, grad_y + b * s.c_out * spatial,
                                               spatial, 1, grad_x + b * k * spatial, spatial, k,
                                               spatial, s.c_out);
        }
        return;
    }
    const std::int64_t chunk = images_per_chunk(s);
    float* cols = workspace(0, k * chunk * spatial);
    float* folded = chunk > 1 ? workspace(1, s.c_out * chunk * spatial) : nullptr;
    for (std::int64_t b0 = 0; b0 < s.n; b0 += chunk) {
        const std::int64_t count = std::min(chunk, s.n - b0);
        const std::int64_t ld = count * spatial;
        const float* gy = fold_grad_y(grad_y, folded, s, b0, count);
        // cols[k, ld] = weight^T[k, c_out] * gy[c_out, ld].
        matmul_f32_packed_parallel_strided(weight, 1, k, gy, ld, 1, cols, ld, k, ld, s.c_out);
        parallel_for(count * s.c_in, grain_for(copy_cutoff, s.kh * s.kw * spatial),
                     [&](std::int64_t t0, std::int64_t t1) {
                         for (std::int64_t t = t0; t < t1; ++t) {
                             const std::int64_t b = t / s.c_in;
                             const std::int64_t c = t % s.c_in;
                             float* out = grad_x + ((b0 + b) * s.c_in + c) * plane;
                             std::fill(out, out + plane, 0.0f);
                             col2im_channel(cols + b * spatial, ld, out, c, s);
                         }
                     });
    }
}

// The weight gradient's size (c_out * K floats) below which it is summed in groups of images
// (each group's product on one thread) rather than as one product per chunk on all threads: a
// small gradient matrix leaves too little work to split one product across many threads.
constexpr std::int64_t weight_grouping_limit = std::int64_t{1} << 18;
constexpr std::int64_t weight_groups_max = 32;

// The weight gradient in fixed groups of images: group g holds images [n g / G, n (g + 1) / G),
// summed image by image on one thread into its own partial matrix; the partials are then added in
// group order. G depends only on the shape, so the result is the same at every thread count.
// single_group sums every image in one group on the calling thread (small convolutions).
void backward_weight_grouped(const float* grad_y, const float* x, float* grad_weight,
                             const Conv2dShape& s, bool single_group = false) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    const std::int64_t size = s.c_out * k;
    const bool pointwise = is_pointwise(s);
    const std::int64_t groups =
        single_group ? 1
                     : std::clamp<std::int64_t>(
                           std::min(conv2d_column_budget / size, weight_groups_max), 1, s.n);
    std::unique_ptr<float[]> partials(new float[static_cast<std::size_t>(groups * size)]);
    parallel_for(groups, 1, [&](std::int64_t g0, std::int64_t g1) {
        float* cols = pointwise ? nullptr : workspace(0, k * spatial);
        float* product = workspace(2, size);
        for (std::int64_t g = g0; g < g1; ++g) {
            float* partial = partials.get() + g * size;
            const std::int64_t first = s.n * g / groups;
            const std::int64_t last = s.n * (g + 1) / groups;
            for (std::int64_t b = first; b < last; ++b) {
                const float* image = x + b * s.c_in * s.h * s.w;
                if (!pointwise) {
                    im2col_rows(image, cols, spatial, s, 0, k);
                }
                // cols^T (p, r) = cols[r * spatial + p]: row stride 1, column stride spatial.
                float* target = b == first ? partial : product;
                matmul_f32_packed_strided(grad_y + b * s.c_out * spatial, spatial, 1,
                                          pointwise ? image : cols, 1, spatial, target, k,
                                          s.c_out, k, spatial);
                if (b != first) {
                    for (std::int64_t i = 0; i < size; ++i) {
                        partial[i] += product[i];
                    }
                }
            }
        }
    });
    parallel_for(size, copy_cutoff, [&](std::int64_t i0, std::int64_t i1) {
        for (std::int64_t i = i0; i < i1; ++i) {
            float sum = partials[to_size(i)];
            for (std::int64_t g = 1; g < groups; ++g) {
                sum += partials[to_size(g * size + i)];
            }
            grad_weight[i] = sum;
        }
    });
}

// The weight gradient per chunk of the batch: one product of depth chunk * spatial on all threads,
// added to grad_weight chunk after chunk.
void backward_weight_folded(const float* grad_y, const float* x, float* grad_weight,
                            const Conv2dShape& s) {
    const std::int64_t k = s.c_in * s.kh * s.kw;
    const std::int64_t spatial = s.h_out * s.w_out;
    const bool pointwise = is_pointwise(s);
    const std::int64_t chunk = pointwise ? 1 : images_per_chunk(s);
    float* cols = pointwise ? nullptr : workspace(0, k * chunk * spatial);
    float* folded = chunk > 1 ? workspace(1, s.c_out * chunk * spatial) : nullptr;
    float* partial = s.n > chunk ? workspace(2, s.c_out * k) : nullptr;
    for (std::int64_t b0 = 0; b0 < s.n; b0 += chunk) {
        const std::int64_t count = std::min(chunk, s.n - b0);
        const std::int64_t ld = count * spatial;
        const float* b_matrix = pointwise ? x + b0 * k * spatial : cols;
        if (!pointwise) {
            im2col_chunk(x, cols, s, b0, count);
        }
        const float* gy = fold_grad_y(grad_y, folded, s, b0, count);
        // [c_out, k] = gy[c_out, ld] * cols^T[ld, k]: the chunk's images summed in the depth.
        float* target = b0 == 0 ? grad_weight : partial;
        matmul_f32_packed_parallel_strided(gy, ld, 1, b_matrix, 1, ld, target, k, s.c_out, k, ld);
        if (b0 > 0) {
            parallel_for(s.c_out * k, copy_cutoff, [&](std::int64_t i0, std::int64_t i1) {
                for (std::int64_t i = i0; i < i1; ++i) {
                    grad_weight[i] += partial[i];
                }
            });
        }
    }
}

// Multiply-adds per image below which whole images per thread beat one large product (when there
// is at least one image per thread).
constexpr std::int64_t per_image_limit = std::int64_t{1} << 22;

// The rule for Auto, the same for the forward pass and the data gradient. Both strategies give the
// same bits, so it may depend on the thread count. Measured at batch 32 on 17 ResNet-18/50 layers
// (conv2d_benchmark --compare --more), forward and data gradient:
// - Folded really folds a 3x3 or 7x7 layer: one large product over a chunk of the batch, after an
//   im2col that per-image needs as well. It won or tied on every such layer at 12 and 24 threads.
// - A pointwise (1x1, stride 1) layer has no im2col, and Folded runs one product per image on all
//   threads, n products in a row, each paying the team's start and its uneven split. Whole images
//   per thread won 11 of 14 such cases at 12 threads (1.1-1.5x), lost 3 by up to 10%, and beat
//   Folded at 24 threads as well once limited to one thread per core (image_threads).
// - Whole images per thread need the batch to spread evenly: the busiest thread has
//   ceil(n / threads) images, so allow at most 1.25 times the average (32 images on 12 threads:
//   3 against 2.67, 1.125; 16 images on 12 threads would be 2 against 1.33, 1.5).
// - Small images (under per_image_limit multiply-adds) give one product too little work to split,
//   whatever the kernel.
Conv2dStrategy choose_strategy(const Conv2dShape& s) {
    const std::int64_t threads = image_threads();
    if (s.n < threads) {
        return Conv2dStrategy::Folded;
    }
    if (multiply_adds(s) / s.n < per_image_limit) {
        return Conv2dStrategy::ImagePerThread;
    }
    const std::int64_t busiest = (s.n + threads - 1) / threads;
    const bool even = 4 * busiest * threads <= 5 * s.n;
    return is_pointwise(s) && even ? Conv2dStrategy::ImagePerThread : Conv2dStrategy::Folded;
}

} // namespace

Conv2dStrategy conv2d_forward_strategy(const Conv2dShape& s) { return choose_strategy(s); }

Conv2dStrategy conv2d_backward_data_strategy(const Conv2dShape& s) { return choose_strategy(s); }

void conv2d_f32_parallel(const float* x, const float* weight, const float* bias, float* y,
                         const Conv2dShape& s, Conv2dStrategy strategy) {
    if (s.n == 0 || s.c_out == 0 || s.h_out * s.w_out == 0) {
        return;
    }
    if (multiply_adds(s) < conv2d_parallel_cutoff) {
        // Too small for threads, but the per-image code with its reused buffer and copy-based
        // im2col on this thread (the sequential baseline was 3-4x slower on small layers).
        conv2d_image_per_thread(x, weight, bias, y, s, s.n);
        return;
    }
    // Both strategies give the same bits; see choose_strategy for the rule.
    if (strategy == Conv2dStrategy::Auto) {
        strategy = conv2d_forward_strategy(s);
    }
    if (strategy == Conv2dStrategy::ImagePerThread) {
        conv2d_image_per_thread(x, weight, bias, y, s);
    } else {
        conv2d_folded(x, weight, bias, y, s);
    }
}

void conv2d_backward_data_f32_parallel(const float* grad_y, const float* weight, float* grad_x,
                                       const Conv2dShape& s) {
    conv2d_backward_data_f32_parallel_strategy(grad_y, weight, grad_x, s, Conv2dStrategy::Auto);
}

void conv2d_backward_data_f32_parallel_strategy(const float* grad_y, const float* weight,
                                                float* grad_x, const Conv2dShape& s,
                                                Conv2dStrategy strategy) {
    if (s.n == 0 || s.c_in * s.kh * s.kw == 0) {
        conv2d_backward_data_f32_sequential(grad_y, weight, grad_x, s);
        return;
    }
    if (multiply_adds(s) < conv2d_parallel_cutoff) {
        backward_data_per_image(grad_y, weight, grad_x, s, s.n); // on this thread
        return;
    }
    // The same bits either way (the products have the same depth, c_out); the same rule as the
    // forward pass.
    if (strategy == Conv2dStrategy::Auto) {
        strategy = conv2d_backward_data_strategy(s);
    }
    if (strategy == Conv2dStrategy::ImagePerThread) {
        backward_data_per_image(grad_y, weight, grad_x, s);
    } else {
        backward_data_folded(grad_y, weight, grad_x, s);
    }
}

void conv2d_backward_weight_f32_parallel(const float* grad_y, const float* x, float* grad_weight,
                                         const Conv2dShape& s) {
    if (s.n == 0 || s.c_out * s.c_in * s.kh * s.kw == 0 || s.h_out * s.w_out == 0) {
        conv2d_backward_weight_f32_sequential(grad_y, x, grad_weight, s);
        return;
    }
    if (multiply_adds(s) < conv2d_parallel_cutoff) {
        // One group on this thread: the images added in order, as the sequential version does.
        backward_weight_grouped(grad_y, x, grad_weight, s, true);
        return;
    }
    // The two round differently, so the choice depends on the shape only.
    if (s.n > 1 && s.c_out * s.c_in * s.kh * s.kw < weight_grouping_limit) {
        backward_weight_grouped(grad_y, x, grad_weight, s);
    } else {
        backward_weight_folded(grad_y, x, grad_weight, s);
    }
}

void conv2d_backward_bias_f32_parallel(const float* grad_y, float* grad_bias,
                                       const Conv2dShape& s) {
    const std::int64_t spatial = s.h_out * s.w_out;
    parallel_for(s.c_out, grain_for(copy_cutoff, s.n * spatial),
                 [&](std::int64_t c0, std::int64_t c1) {
                     for (std::int64_t co = c0; co < c1; ++co) {
                         double sum = 0.0;
                         for (std::int64_t b = 0; b < s.n; ++b) {
                             const float* plane = grad_y + (b * s.c_out + co) * spatial;
                             for (std::int64_t p = 0; p < spatial; ++p) {
                                 sum += static_cast<double>(plane[p]);
                             }
                         }
                         grad_bias[co] = static_cast<float>(sum);
                     }
                 });
}

} // namespace mensurflow::cpu
