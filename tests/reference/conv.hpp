// Direct-loop references for 2-D convolution (NCHW), accumulated in double: no im2col, no GEMM.
// Every output (or gradient) element is the plain sum over its window, written straight from the
// definition in include/mensurflow/ops/conv.hpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

struct Conv2dSizes {
    std::int64_t n, c_in, h, w;
    std::int64_t c_out, kh, kw;
    std::int64_t stride_h, stride_w, pad_h, pad_w;

    std::int64_t h_out() const { return (h + 2 * pad_h - kh) / stride_h + 1; }

    std::int64_t w_out() const { return (w + 2 * pad_w - kw) / stride_w + 1; }

    std::int64_t x_size() const { return n * c_in * h * w; }

    std::int64_t weight_size() const { return c_out * c_in * kh * kw; }

    std::int64_t y_size() const { return n * c_out * h_out() * w_out(); }
};

namespace detail {

inline std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

inline std::vector<float> to_float(const std::vector<double>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<float>(values[i]);
    }
    return out;
}

// Calls body(b, co, oh, ow, ci, ih, iw, i, j) for every product term of the convolution whose input
// cell (ih, iw) is inside the image; padding terms are zero and skipped.
template <typename Body> void for_each_term(const Conv2dSizes& s, Body&& body) {
    for (std::int64_t b = 0; b < s.n; ++b) {
        for (std::int64_t co = 0; co < s.c_out; ++co) {
            for (std::int64_t oh = 0; oh < s.h_out(); ++oh) {
                for (std::int64_t ow = 0; ow < s.w_out(); ++ow) {
                    for (std::int64_t ci = 0; ci < s.c_in; ++ci) {
                        for (std::int64_t i = 0; i < s.kh; ++i) {
                            for (std::int64_t j = 0; j < s.kw; ++j) {
                                const std::int64_t ih = oh * s.stride_h - s.pad_h + i;
                                const std::int64_t iw = ow * s.stride_w - s.pad_w + j;
                                if (ih >= 0 && ih < s.h && iw >= 0 && iw < s.w) {
                                    body(b, co, oh, ow, ci, ih, iw, i, j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace detail

// y[n, c_out, h_out, w_out]; bias may be empty (no bias).
inline std::vector<float> conv2d(const std::vector<float>& x, const std::vector<float>& weight,
                                 const std::vector<float>& bias, const Conv2dSizes& s) {
    using detail::at;
    const std::int64_t ho = s.h_out(), wo = s.w_out();
    std::vector<double> y(at(s.y_size()), 0.0);
    if (!bias.empty()) {
        for (std::int64_t b = 0; b < s.n; ++b) {
            for (std::int64_t co = 0; co < s.c_out; ++co) {
                for (std::int64_t p = 0; p < ho * wo; ++p) {
                    y[at((b * s.c_out + co) * ho * wo + p)] = bias[at(co)];
                }
            }
        }
    }
    detail::for_each_term(s, [&](auto b, auto co, auto oh, auto ow, auto ci, auto ih, auto iw,
                                 auto i, auto j) {
        y[at(((b * s.c_out + co) * ho + oh) * wo + ow)] +=
            static_cast<double>(weight[at(((co * s.c_in + ci) * s.kh + i) * s.kw + j)]) *
            x[at(((b * s.c_in + ci) * s.h + ih) * s.w + iw)];
    });
    return detail::to_float(y);
}

// d(sum(y * grad_y)) / dx.
inline std::vector<float> conv2d_backward_data(const std::vector<float>& grad_y,
                                               const std::vector<float>& weight,
                                               const Conv2dSizes& s) {
    using detail::at;
    const std::int64_t ho = s.h_out(), wo = s.w_out();
    std::vector<double> gx(at(s.x_size()), 0.0);
    detail::for_each_term(s, [&](auto b, auto co, auto oh, auto ow, auto ci, auto ih, auto iw,
                                 auto i, auto j) {
        gx[at(((b * s.c_in + ci) * s.h + ih) * s.w + iw)] +=
            static_cast<double>(weight[at(((co * s.c_in + ci) * s.kh + i) * s.kw + j)]) *
            grad_y[at(((b * s.c_out + co) * ho + oh) * wo + ow)];
    });
    return detail::to_float(gx);
}

// d(sum(y * grad_y)) / dweight, summed over the batch.
inline std::vector<float> conv2d_backward_weight(const std::vector<float>& grad_y,
                                                 const std::vector<float>& x,
                                                 const Conv2dSizes& s) {
    using detail::at;
    const std::int64_t ho = s.h_out(), wo = s.w_out();
    std::vector<double> gw(at(s.weight_size()), 0.0);
    detail::for_each_term(s, [&](auto b, auto co, auto oh, auto ow, auto ci, auto ih, auto iw,
                                 auto i, auto j) {
        gw[at(((co * s.c_in + ci) * s.kh + i) * s.kw + j)] +=
            static_cast<double>(x[at(((b * s.c_in + ci) * s.h + ih) * s.w + iw)]) *
            grad_y[at(((b * s.c_out + co) * ho + oh) * wo + ow)];
    });
    return detail::to_float(gw);
}

// d(sum(y * grad_y)) / dbias.
inline std::vector<float> conv2d_backward_bias(const std::vector<float>& grad_y,
                                               const Conv2dSizes& s) {
    using detail::at;
    const std::int64_t spatial = s.h_out() * s.w_out();
    std::vector<double> gb(at(s.c_out), 0.0);
    for (std::int64_t b = 0; b < s.n; ++b) {
        for (std::int64_t co = 0; co < s.c_out; ++co) {
            for (std::int64_t p = 0; p < spatial; ++p) {
                gb[at(co)] += grad_y[at((b * s.c_out + co) * spatial + p)];
            }
        }
    }
    return detail::to_float(gb);
}

} // namespace reference
