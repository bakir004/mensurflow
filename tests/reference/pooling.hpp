// Direct references for 2-D pooling (NCHW): every window is walked over its full kH x kW extent,
// padding included, and padded cells are skipped by an explicit test.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

struct Pool2dSizes {
    std::int64_t n, c, h, w;
    std::int64_t kh, kw;
    std::int64_t stride_h, stride_w, pad_h, pad_w;

    std::int64_t h_out() const { return (h + 2 * pad_h - kh) / stride_h + 1; }

    std::int64_t w_out() const { return (w + 2 * pad_w - kw) / stride_w + 1; }

    std::int64_t x_size() const { return n * c * h * w; }

    std::int64_t y_size() const { return n * c * h_out() * w_out(); }
};

struct MaxPoolResult {
    std::vector<float> output;
    std::vector<std::int64_t> indices;
};

// Calls body(plane, output position, ih, iw) for every input cell of every window, in row-major
// order within the window; padding cells are skipped.
template <typename Body> void for_each_window_cell(const Pool2dSizes& s, Body&& body) {
    for (std::int64_t plane = 0; plane < s.n * s.c; ++plane) {
        for (std::int64_t oh = 0; oh < s.h_out(); ++oh) {
            for (std::int64_t ow = 0; ow < s.w_out(); ++ow) {
                const std::int64_t out = (plane * s.h_out() + oh) * s.w_out() + ow;
                for (std::int64_t i = 0; i < s.kh; ++i) {
                    for (std::int64_t j = 0; j < s.kw; ++j) {
                        const std::int64_t ih = oh * s.stride_h - s.pad_h + i;
                        const std::int64_t iw = ow * s.stride_w - s.pad_w + j;
                        if (ih < 0 || ih >= s.h || iw < 0 || iw >= s.w) {
                            continue; // padding: never a candidate
                        }
                        body(plane, out, ih, iw);
                    }
                }
            }
        }
    }
}

// A candidate replaces the current choice when there is none yet, or when the current choice is
// a number and the candidate is NaN or strictly larger. So: first maximum, first NaN.
inline MaxPoolResult max_pool2d(const std::vector<float>& x, const Pool2dSizes& s) {
    const auto outputs = static_cast<std::size_t>(s.y_size());
    MaxPoolResult result{std::vector<float>(outputs), std::vector<std::int64_t>(outputs, -1)};
    for_each_window_cell(s, [&](std::int64_t plane, std::int64_t out, std::int64_t ih,
                                std::int64_t iw) {
        const auto o = static_cast<std::size_t>(out);
        const float candidate = x[static_cast<std::size_t>((plane * s.h + ih) * s.w + iw)];
        const bool none = result.indices[o] < 0;
        const float current = result.output[o];
        if (none || (!std::isnan(current) && (std::isnan(candidate) || candidate > current))) {
            result.output[o] = candidate;
            result.indices[o] = ih * s.w + iw;
        }
    });
    return result;
}

// Scatter-add of grad_y into the chosen cells, in double.
inline std::vector<float> max_pool2d_backward(const std::vector<float>& grad_y,
                                              const std::vector<std::int64_t>& indices,
                                              const Pool2dSizes& s) {
    std::vector<double> gx(static_cast<std::size_t>(s.x_size()), 0.0);
    const std::int64_t outputs = s.h_out() * s.w_out();
    for (std::int64_t out = 0; out < s.y_size(); ++out) {
        const std::int64_t plane = out / outputs;
        gx[static_cast<std::size_t>(plane * s.h * s.w + indices[static_cast<std::size_t>(out)])] +=
            grad_y[static_cast<std::size_t>(out)];
    }
    std::vector<float> result(gx.size());
    for (std::size_t i = 0; i < gx.size(); ++i) {
        result[i] = static_cast<float>(gx[i]);
    }
    return result;
}

} // namespace reference
