// conv2d forward and backward against the direct-loop reference in tests/reference/conv.hpp, a
// finite-difference check of every gradient, autograd, views, empty shapes and errors.
#include "backends/cpu/ops/conv/conv2d.hpp"
#include "backends/cpu/threads.hpp"
#include "gradient_check.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/conv.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
using reference::Conv2dSizes;

namespace {

struct Case {
    const char* name;
    Conv2dSizes s;
    bool bias;
};

// {n, c_in, h, w, c_out, kh, kw, stride_h, stride_w, pad_h, pad_w}
const Case cases[] = {
    {"odd non-square, stride and padding", {2, 3, 7, 5, 4, 3, 2, 2, 1, 1, 0}, true},
    {"1x1 kernel", {2, 5, 4, 6, 3, 1, 1, 1, 1, 0, 0}, true},
    {"1x1 kernel, stride 2", {1, 3, 7, 6, 2, 1, 1, 2, 2, 0, 0}, false},
    {"kernel as large as the input", {2, 2, 5, 4, 3, 5, 4, 1, 1, 0, 0}, true},
    {"kernel larger than the input, padded", {1, 2, 3, 2, 2, 5, 4, 1, 1, 1, 1}, true},
    {"batch 1, one channel", {1, 1, 6, 6, 1, 3, 3, 2, 2, 1, 1}, true},
    {"stride larger than the kernel", {2, 2, 9, 8, 3, 2, 2, 3, 3, 0, 0}, true},
    {"padding larger than half the kernel", {1, 2, 4, 5, 2, 3, 3, 1, 2, 2, 2}, false},
    {"stride does not divide the span", {1, 3, 10, 7, 2, 3, 2, 3, 2, 1, 0}, true},
    {"many channels", {3, 8, 13, 11, 16, 3, 3, 1, 1, 1, 1}, true},
};

Shape x_shape(const Conv2dSizes& s) { return {s.n, s.c_in, s.h, s.w}; }

Shape weight_shape(const Conv2dSizes& s) { return {s.c_out, s.c_in, s.kh, s.kw}; }

Shape y_shape(const Conv2dSizes& s) { return {s.n, s.c_out, s.h_out(), s.w_out()}; }

Size2d stride_of(const Conv2dSizes& s) { return {s.stride_h, s.stride_w}; }

Size2d padding_of(const Conv2dSizes& s) { return {s.pad_h, s.pad_w}; }

// Float32 products of values in [-1, 1]: allow a rounding error growing with the number of terms.
double tolerance(std::int64_t terms) { return 1e-5 * static_cast<double>(terms + 1); }

// Forward and the three backward ops against the reference, then the same gradients through
// autograd.
void test_against_reference(const Case& c) {
    const Conv2dSizes& s = c.s;
    const std::string label = std::string(c.name) + ": ";
    const auto x = test::random_values(s.x_size(), 1);
    const auto w = test::random_values(s.weight_size(), 2);
    const auto b = c.bias ? test::random_values(s.c_out, 3) : std::vector<float>{};
    const auto gy = test::random_values(s.y_size(), 4);

    const Tensor tx = Tensor::from_vector(x, x_shape(s));
    const Tensor tw = Tensor::from_vector(w, weight_shape(s));
    const Tensor tb = c.bias ? Tensor::from_vector(b, {s.c_out}) : Tensor();
    const Tensor tgy = Tensor::from_vector(gy, y_shape(s));

    const Tensor y = conv2d(tx, tw, c.bias ? &tb : nullptr, stride_of(s), padding_of(s), {1, 1},
                            1, Execution::CPU);
    test::require(y.shape() == y_shape(s), label + "output shape " + to_string(y.shape()));
    test::require_close(y.to_vector<float>(), reference::conv2d(x, w, b, s),
                        tolerance(s.c_in * s.kh * s.kw), 0.0, label + "forward");

    const auto want_gx = reference::conv2d_backward_data(gy, w, s);
    const auto want_gw = reference::conv2d_backward_weight(gy, x, s);
    const auto want_gb = reference::conv2d_backward_bias(gy, s);
    const double gx_tol = tolerance(s.c_out * s.kh * s.kw);
    const double gw_tol = tolerance(s.n * s.h_out() * s.w_out());
    test::require_close(conv2d_backward_data(tgy, tw, x_shape(s), stride_of(s), padding_of(s))
                            .to_vector<float>(),
                        want_gx, gx_tol, 0.0, label + "backward data");
    test::require_close(conv2d_backward_weight(tgy, tx, weight_shape(s), stride_of(s),
                                               padding_of(s))
                            .to_vector<float>(),
                        want_gw, gw_tol, 0.0, label + "backward weight");
    test::require_close(conv2d_backward_bias(tgy).to_vector<float>(), want_gb, gw_tol, 0.0,
                        label + "backward bias");

    // Autograd records on the forward op and calls the same backward ops.
    Tensor ax = Tensor::from_vector(x, x_shape(s));
    Tensor aw = Tensor::from_vector(w, weight_shape(s));
    Tensor ab = c.bias ? Tensor::from_vector(b, {s.c_out}) : Tensor();
    ax.set_requires_grad();
    aw.set_requires_grad();
    if (c.bias) {
        ab.set_requires_grad();
    }
    test::weighted_loss(conv2d(ax, aw, c.bias ? &ab : nullptr, stride_of(s), padding_of(s)), gy)
        .backward();
    test::require_close(ax.grad().to_vector<float>(), want_gx, gx_tol, 0.0,
                        label + "autograd x");
    test::require_close(aw.grad().to_vector<float>(), want_gw, gw_tol, 0.0,
                        label + "autograd weight");
    if (c.bias) {
        test::require_close(ab.grad().to_vector<float>(), want_gb, gw_tol, 0.0,
                            label + "autograd bias");
    }
}

// Central differences of sum(conv2d(x, w, b) * gy) against the backward ops, for x, w and b.
void test_finite_differences(const Conv2dSizes& s, const std::string& label) {
    const auto x = test::random_values(s.x_size(), 11);
    const auto w = test::random_values(s.weight_size(), 12);
    const auto b = test::random_values(s.c_out, 13);
    const auto gy = test::random_values(s.y_size(), 14);
    auto forward = [&](const std::vector<float>& xv, const std::vector<float>& wv,
                       const std::vector<float>& bv) {
        const Tensor tb = Tensor::from_vector(bv, {s.c_out});
        return conv2d(Tensor::from_vector(xv, x_shape(s)), Tensor::from_vector(wv, weight_shape(s)),
                      &tb, stride_of(s), padding_of(s))
            .to_vector<float>();
    };
    // The loss is linear in each argument, so central differences are exact up to rounding.
    const float eps = 0.1f;
    const double tol = 2e-3;
    const Tensor tgy = Tensor::from_vector(gy, y_shape(s));
    test::require_close(
        conv2d_backward_data(tgy, Tensor::from_vector(w, weight_shape(s)), x_shape(s),
                             stride_of(s), padding_of(s))
            .to_vector<float>(),
        test::numeric_gradient([&](const std::vector<float>& v) { return forward(v, w, b); }, x,
                               gy, eps),
        tol, 0.0, label + ": finite differences, data");
    test::require_close(
        conv2d_backward_weight(tgy, Tensor::from_vector(x, x_shape(s)), weight_shape(s),
                               stride_of(s), padding_of(s))
            .to_vector<float>(),
        test::numeric_gradient([&](const std::vector<float>& v) { return forward(x, v, b); }, w,
                               gy, eps),
        tol, 0.0, label + ": finite differences, weight");
    test::require_close(
        conv2d_backward_bias(tgy).to_vector<float>(),
        test::numeric_gradient([&](const std::vector<float>& v) { return forward(x, w, v); }, b,
                               gy, eps),
        tol, 0.0, label + ": finite differences, bias");
}

// im2col and col2im are adjoint: <im2col(x), c> == <x, col2im(c)> for any x and c.
void test_im2col_adjoint() {
    cpu::Conv2dShape s;
    s.n = 1;
    s.c_in = 2;
    s.h = 5;
    s.w = 4;
    s.c_out = 1;
    s.kh = 3;
    s.kw = 2;
    s.stride_h = 2;
    s.stride_w = 1;
    s.pad_h = 1;
    s.pad_w = 1;
    s.h_out = (s.h + 2 * s.pad_h - s.kh) / s.stride_h + 1;
    s.w_out = (s.w + 2 * s.pad_w - s.kw) / s.stride_w + 1;
    const std::int64_t rows = s.c_in * s.kh * s.kw, columns = s.h_out * s.w_out;
    const auto x = test::random_values(s.c_in * s.h * s.w, 21);
    const auto c = test::random_values(rows * columns, 22);
    std::vector<float> cols(static_cast<std::size_t>(rows * columns));
    cpu::im2col_f32(x.data(), cols.data(), s);
    std::vector<float> image(x.size(), 0.0f);
    cpu::col2im_f32(c.data(), image.data(), s);
    const double left = test::weighted_sum(cols, c);
    const double right = test::weighted_sum(image, x);
    test::require(std::abs(left - right) < 1e-4, "im2col and col2im are adjoint");

    // A padded cell of the first window reads as zero: row 0 (c 0, i 0, j 0), column 0 is
    // x[0, -1, -1].
    test::require(cols[0] == 0.0f, "im2col reads padding as zero");
}

// Views are packed before the kernel; the result matches the same values made contiguous.
void test_views() {
    const Conv2dSizes s{2, 3, 5, 6, 4, 3, 2, 1, 2, 1, 1};
    // x stored as NHWC and permuted to NCHW; weight every second column of a wider tensor; bias
    // every second element.
    const Tensor x_nhwc =
        Tensor::from_vector(test::random_values(s.x_size(), 31), {s.n, s.h, s.w, s.c_in});
    const Tensor x = x_nhwc.permute({0, 3, 1, 2});
    const Tensor wide =
        Tensor::from_vector(test::random_values(s.weight_size() * 2, 32),
                            {s.c_out, s.c_in, s.kh, 2 * s.kw});
    const Tensor w = wide.slice(3, 0, 2 * s.kw, 2);
    const Tensor b = Tensor::from_vector(test::random_values(2 * s.c_out, 33), {2 * s.c_out})
                         .slice(0, 0, 2 * s.c_out, 2);
    test::require(!x.is_contiguous() && !w.is_contiguous() && !b.is_contiguous(),
                  "views are not contiguous");
    const auto xv = x.to_vector<float>();
    const auto wv = w.to_vector<float>();
    const auto bv = b.to_vector<float>();
    test::require_close(conv2d(x, w, &b, stride_of(s), padding_of(s)).to_vector<float>(),
                        reference::conv2d(xv, wv, bv, s), tolerance(s.c_in * s.kh * s.kw), 0.0,
                        "conv2d of views");

    // A transposed gradient view goes through the backward ops too.
    const auto gy = test::random_values(s.y_size(), 34);
    const Tensor gy_t =
        Tensor::from_vector(gy, {s.n, s.c_out, s.w_out(), s.h_out()}).transpose(2, 3);
    const auto gyv = gy_t.to_vector<float>();
    test::require_close(conv2d_backward_data(gy_t, w, x_shape(s), stride_of(s), padding_of(s))
                            .to_vector<float>(),
                        reference::conv2d_backward_data(gyv, wv, s),
                        tolerance(s.c_out * s.kh * s.kw), 0.0, "backward data of views");
    test::require_close(conv2d_backward_weight(gy_t, x, weight_shape(s), stride_of(s),
                                               padding_of(s))
                            .to_vector<float>(),
                        reference::conv2d_backward_weight(gyv, xv, s),
                        tolerance(s.n * s.h_out() * s.w_out()), 0.0, "backward weight of views");

    // Autograd through a view: the gradient arrives in the view's own shape.
    Tensor leaf = Tensor::from_vector(test::random_values(s.x_size(), 31), {s.n, s.h, s.w, s.c_in});
    leaf.set_requires_grad();
    test::weighted_loss(conv2d(leaf.permute({0, 3, 1, 2}), w, &b, stride_of(s), padding_of(s)),
                        gyv)
        .backward();
    test::require(leaf.grad().shape() == leaf.shape(), "gradient of the NHWC leaf");
    const Tensor want_nchw =
        Tensor::from_vector(reference::conv2d_backward_data(gyv, wv, s), x_shape(s));
    test::require_close(leaf.grad().to_vector<float>(),
                        want_nchw.permute({0, 2, 3, 1}).to_vector<float>(),
                        tolerance(s.c_out * s.kh * s.kw), 0.0, "autograd through a permuted view");
}

void test_empty() {
    // Empty batch: empty output; the weight and bias gradients are zeros.
    const Tensor w = Tensor::from_vector(test::random_values(2 * 3 * 3 * 3, 41), {2, 3, 3, 3});
    const Tensor b = Tensor::from_vector(std::vector<float>{0.5f, -1.5f}, {2});
    const Tensor x0 = Tensor::empty({0, 3, 5, 5});
    const Tensor y0 = conv2d(x0, w, &b);
    test::require(y0.shape() == (Shape{0, 2, 3, 3}), "empty batch output shape");
    const Tensor gy0 = Tensor::empty({0, 2, 3, 3});
    test::require(conv2d_backward_data(gy0, w, x0.shape()).shape() == x0.shape(),
                  "empty batch data gradient");
    test::require_close(conv2d_backward_weight(gy0, x0, w.shape()).to_vector<float>(),
                        std::vector<float>(54, 0.0f), 0.0, 0.0, "empty batch weight gradient");
    test::require_close(conv2d_backward_bias(gy0).to_vector<float>(),
                        std::vector<float>{0.0f, 0.0f}, 0.0, 0.0, "empty batch bias gradient");

    // C_out == 0: empty output.
    const Tensor x = Tensor::from_vector(test::random_values(3 * 5 * 5, 42), {1, 3, 5, 5});
    test::require(conv2d(x, Tensor::empty({0, 3, 3, 3})).shape() == (Shape{1, 0, 3, 3}),
                  "no output channels");

    // C_in == 0: the output is the bias, or zeros.
    const Tensor x_no_channels = Tensor::empty({2, 0, 4, 4});
    const Tensor w_no_channels = Tensor::empty({2, 0, 3, 3});
    std::vector<float> want;
    for (int n = 0; n < 2; ++n) {
        for (float value : {0.5f, -1.5f}) {
            want.insert(want.end(), 4, value);
        }
    }
    test::require_close(conv2d(x_no_channels, w_no_channels, &b).to_vector<float>(), want, 0.0,
                        0.0, "no input channels gives the bias");
    test::require_close(conv2d(x_no_channels, w_no_channels).to_vector<float>(),
                        std::vector<float>(16, 0.0f), 0.0, 0.0, "no input channels, no bias");
}

void test_errors() {
    const Tensor x = Tensor::zeros({1, 2, 5, 5});
    const Tensor w = Tensor::zeros({3, 2, 3, 3});
    const Tensor b = Tensor::zeros({3});
    using E = std::invalid_argument;
    test::require_throws<E>([&] { conv2d(Tensor(), w); }, "undefined x");
    test::require_throws<E>([&] { conv2d(x, Tensor()); }, "undefined weight");
    test::require_throws<E>([&] { conv2d(Tensor::zeros({2, 5, 5}), w); }, "3-D input");
    test::require_throws<E>([&] { conv2d(x, Tensor::zeros({3, 2, 3})); }, "3-D weight");
    test::require_throws<E>([&] { conv2d(x, Tensor::zeros({3, 4, 3, 3})); }, "channel mismatch");
    const Tensor wrong_bias = Tensor::zeros({4});
    test::require_throws<E>([&] { conv2d(x, w, &wrong_bias); }, "bias length");
    const Tensor bias_2d = Tensor::zeros({3, 1});
    test::require_throws<E>([&] { conv2d(x, w, &bias_2d); }, "2-D bias");
    test::require_throws<E>([&] { conv2d(x, w, &b, {0, 1}); }, "zero stride");
    test::require_throws<E>([&] { conv2d(x, w, &b, {1, 1}, {0, -1}); }, "negative padding");
    test::require_throws<E>([&] { conv2d(x, w, &b, {1, 1}, {0, 0}, {2, 1}); }, "dilation 2");
    test::require_throws<E>([&] { conv2d(x, w, &b, {1, 1}, {0, 0}, {1, 1}, 2); }, "groups 2");
    test::require_throws<E>([&] { conv2d(x, Tensor::zeros({3, 2, 6, 3})); },
                            "kernel taller than the input");
    const Tensor wide = Tensor::zeros({3, 2, 3, 8});
    test::require_throws<E>([&] { conv2d(x, wide, nullptr, {1, 1}, {0, 1}); },
                            "kernel wider than the padded input");
    test::require_throws<E>([&] { conv2d(x, Tensor::zeros({3, 2, 0, 3})); }, "kernel height 0");
    const Tensor x_int = Tensor::from_vector(std::vector<std::int64_t>(50, 0), {1, 2, 5, 5});
    test::require_throws<E>([&] { conv2d(x_int, w); }, "dtype mismatch");

    // The backward ops check that grad_output has the forward output's shape.
    const Tensor gy = Tensor::zeros({1, 3, 3, 3});
    const Tensor wrong_gy = Tensor::zeros({1, 3, 3, 4});
    test::require_throws<E>([&] { conv2d_backward_data(wrong_gy, w, x.shape()); },
                            "data gradient shape");
    test::require_throws<E>([&] { conv2d_backward_data(gy, w, Shape{1, 2, 5}); },
                            "data input_shape rank");
    test::require_throws<E>([&] { conv2d_backward_data(gy, w, x.shape(), {2, 2}); },
                            "data gradient for another stride");
    test::require_throws<E>([&] { conv2d_backward_weight(gy, x, Shape{3, 2, 2, 3}); },
                            "weight gradient shape");
    test::require_throws<E>([&] { conv2d_backward_weight(gy, x, Shape{3, 1, 3, 3}); },
                            "weight_shape channels");
    test::require_throws<E>([&] { conv2d_backward_bias(Tensor::zeros({3, 3})); },
                            "bias gradient rank");
}

// Only some parameters require gradients; the others get none.
void test_partial_autograd() {
    const Conv2dSizes s{2, 2, 4, 5, 3, 2, 3, 1, 1, 0, 1};
    const auto gy = test::random_values(s.y_size(), 51);
    const Tensor x = Tensor::from_vector(test::random_values(s.x_size(), 52), x_shape(s));
    Tensor w = Tensor::from_vector(test::random_values(s.weight_size(), 53), weight_shape(s));
    w.set_requires_grad();
    test::weighted_loss(conv2d(x, w, nullptr, stride_of(s), padding_of(s)), gy).backward();
    test::require(!x.has_grad() && w.has_grad(), "only the weight gets a gradient");
    test::require_close(w.grad().to_vector<float>(),
                        reference::conv2d_backward_weight(gy, x.to_vector<float>(), s),
                        tolerance(s.n * s.h_out() * s.w_out()), 0.0, "weight-only autograd");
}

cpu::Conv2dShape kernel_shape(const Conv2dSizes& c) {
    cpu::Conv2dShape s;
    s.n = c.n;
    s.c_in = c.c_in;
    s.h = c.h;
    s.w = c.w;
    s.c_out = c.c_out;
    s.kh = c.kh;
    s.kw = c.kw;
    s.stride_h = c.stride_h;
    s.stride_w = c.stride_w;
    s.pad_h = c.pad_h;
    s.pad_w = c.pad_w;
    s.h_out = c.h_out();
    s.w_out = c.w_out();
    return s;
}

std::int64_t multiply_adds(const Conv2dSizes& c) {
    return c.n * c.c_out * c.c_in * c.kh * c.kw * c.h_out() * c.w_out();
}

// The four outputs of one convolution: y, grad_x, grad_weight, grad_bias.
struct Results {
    std::vector<float> y, gx, gw, gb;
};

template <typename Forward, typename Data, typename Weight, typename Bias>
Results run_kernels(const Conv2dSizes& c, Forward forward, Data data, Weight weight, Bias bias) {
    const cpu::Conv2dShape s = kernel_shape(c);
    const auto x = test::random_values(c.x_size(), 61);
    const auto w = test::random_values(c.weight_size(), 62);
    const auto b = test::random_values(c.c_out, 63);
    const auto gy = test::random_values(c.y_size(), 64);
    Results r{std::vector<float>(static_cast<std::size_t>(c.y_size())),
              std::vector<float>(static_cast<std::size_t>(c.x_size())),
              std::vector<float>(static_cast<std::size_t>(c.weight_size())),
              std::vector<float>(static_cast<std::size_t>(c.c_out))};
    forward(x.data(), w.data(), b.data(), r.y.data(), s);
    data(gy.data(), w.data(), r.gx.data(), s);
    weight(gy.data(), x.data(), r.gw.data(), s);
    bias(gy.data(), r.gb.data(), s);
    return r;
}

void require_same_bits(const Results& a, const Results& b, const std::string& label) {
    test::require(test::matches_f32(a.y, b.y) && test::matches_f32(a.gx, b.gx) &&
                      test::matches_f32(a.gw, b.gw) && test::matches_f32(a.gb, b.gb),
                  label + ": results differ in their bits");
}

// The parallel kernels against the sequential ones, at several thread counts and around the
// cutoff; every thread count must give the same bits as one thread.
void test_parallel() {
    const std::vector<Conv2dSizes> sizes = {
        // Below, at and above the cutoff of 2^20 multiply-adds: 16 * 16 * 64 * {63, 64, 65}.
        {1, 16, 64, 63, 16, 1, 1, 1, 1, 0, 0},
        {1, 16, 64, 64, 16, 1, 1, 1, 1, 0, 0},
        {1, 16, 64, 65, 16, 1, 1, 1, 1, 0, 0},
        {2, 8, 32, 32, 8, 2, 2, 1, 1, 0, 0},
        // ResNet-like layers, odd sizes and strides.
        {4, 16, 23, 19, 24, 3, 3, 1, 1, 1, 1},
        {3, 3, 41, 37, 16, 7, 7, 2, 2, 3, 3},
        {8, 32, 9, 9, 40, 3, 3, 2, 1, 1, 0},
        {1, 64, 14, 14, 64, 1, 1, 1, 1, 0, 0}, // pointwise, one image
        {8, 32, 10, 12, 48, 1, 1, 1, 1, 0, 0}, // pointwise, several images
        // A weight gradient of 256 * 1152 floats: one product per chunk rather than groups.
        {2, 128, 6, 6, 256, 3, 3, 1, 1, 1, 1},
        {30, 4, 12, 11, 6, 3, 3, 1, 1, 1, 1}, // more images than weight groups
        // Several chunks of the batch: 108 * 28900 floats of columns per image, two images per chunk.
        {5, 12, 170, 170, 3, 3, 3, 1, 1, 1, 1},
    };
    const int original = cpu::max_threads();
    for (const Conv2dSizes& c : sizes) {
        const std::string label = "parallel conv2d " + to_string(x_shape(c)) + " * " +
                                  to_string(weight_shape(c)) + ", " +
                                  std::to_string(multiply_adds(c)) + " multiply-adds";
        const Results sequential = run_kernels(
            c, cpu::conv2d_f32_sequential, cpu::conv2d_backward_data_f32_sequential,
            cpu::conv2d_backward_weight_f32_sequential, cpu::conv2d_backward_bias_f32_sequential);
        Results one_thread;
        for (int threads : {1, 2, 3, 7, original}) {
            cpu::set_max_threads(threads);
            for (cpu::Conv2dStrategy strategy :
                 {cpu::Conv2dStrategy::Auto, cpu::Conv2dStrategy::Folded,
                  cpu::Conv2dStrategy::ImagePerThread, cpu::Conv2dStrategy::Tiles}) {
                const Results parallel = run_kernels(
                    c,
                    [strategy](const float* x, const float* w, const float* b, float* y,
                               const cpu::Conv2dShape& s) {
                        cpu::conv2d_f32_parallel(x, w, b, y, s, strategy);
                    },
                    cpu::conv2d_backward_data_f32_parallel,
                    cpu::conv2d_backward_weight_f32_parallel,
                    cpu::conv2d_backward_bias_f32_parallel);
                const std::string at = label + ", " + std::to_string(threads) + " threads, strategy " +
                                       std::to_string(static_cast<int>(strategy));
                test::require_close(parallel.y, sequential.y,
                                    tolerance(c.c_in * c.kh * c.kw), 0.0, at + ": forward");
                test::require_close(parallel.gx, sequential.gx,
                                    tolerance(c.c_out * c.kh * c.kw), 0.0, at + ": data");
                test::require_close(parallel.gw, sequential.gw,
                                    tolerance(c.n * c.h_out() * c.w_out()), 0.0, at + ": weight");
                test::require_close(parallel.gb, sequential.gb, 1e-6, 1e-6, at + ": bias");
                if (threads == 1 && strategy == cpu::Conv2dStrategy::Auto) {
                    one_thread = parallel;
                } else {
                    // Every thread count and both strategies give the bits of one thread.
                    require_same_bits(parallel, one_thread, at);
                }
            }
        }
    }
    cpu::set_max_threads(original);
}

// The Auto rule (see choose_strategy and conv2d_forward_strategy in conv2d.cpp): a pointwise
// layer with a batch that spreads evenly goes to whole images per thread, a 3x3 layer of the same
// size to the folded product; single images and short depths below. Independent of the core
// count: a batch of 4 images per thread spreads evenly over any number of threads up to
// max_threads().
void test_strategy_rule() {
    auto shape = [](std::int64_t n, std::int64_t c, std::int64_t hw, std::int64_t k) {
        cpu::Conv2dShape s;
        s.n = n;
        s.c_in = s.c_out = c;
        s.h = s.w = s.h_out = s.w_out = hw;
        s.kh = s.kw = k;
        s.pad_h = s.pad_w = k / 2;
        return s;
    };
    const std::int64_t batch = 4 * cpu::max_threads();
    const cpu::Conv2dShape pointwise = shape(batch, 256, 16, 1); // 16.8 M multiply-adds per image
    const cpu::Conv2dShape square = shape(batch, 128, 16, 3);   // 37.7 M
    test::require(cpu::conv2d_forward_strategy(pointwise) == cpu::Conv2dStrategy::ImagePerThread &&
                      cpu::conv2d_backward_data_strategy(pointwise) ==
                          cpu::Conv2dStrategy::ImagePerThread,
                  "a pointwise layer with an even batch should run whole images per thread");
    test::require(cpu::conv2d_forward_strategy(square) == cpu::Conv2dStrategy::Folded &&
                      cpu::conv2d_backward_data_strategy(square) == cpu::Conv2dStrategy::Folded,
                  "a 3x3 layer should run the folded product");
    // One image: tiles of output pixels when every thread gets at least 64 of them (4096 pixels
    // cover up to 64 threads), the folded product when there are too few (49).
    test::require(cpu::conv2d_forward_strategy(shape(1, 64, 64, 3)) == cpu::Conv2dStrategy::Tiles,
                  "a single wide image should run in tiles");
    test::require(cpu::conv2d_forward_strategy(shape(1, 256, 7, 3)) == cpu::Conv2dStrategy::Folded &&
                      cpu::conv2d_backward_data_strategy(shape(1, 64, 64, 3)) ==
                          cpu::Conv2dStrategy::Folded,
                  "a small deep image, and every data gradient of one image, should run folded");
    // A short depth (3 channels, 3x3: 27) runs in tiles at any batch.
    cpu::Conv2dShape stem = shape(batch, 3, 64, 3);
    stem.c_out = 64;
    test::require(cpu::conv2d_forward_strategy(stem) == cpu::Conv2dStrategy::Tiles,
                  "a short-depth layer should run in tiles");
}

} // namespace

int main() {
    return test::run("conv2d_test", [] {
        test_strategy_rule();
        test_parallel();
        test_im2col_adjoint();
        for (const Case& c : cases) {
            test_against_reference(c);
        }
        test_finite_differences({2, 2, 5, 4, 3, 3, 2, 2, 1, 1, 1}, "stride 2x1, padding 1");
        test_finite_differences({1, 1, 3, 3, 2, 3, 3, 1, 1, 0, 0}, "kernel as large as the input");
        test_finite_differences({2, 3, 4, 3, 2, 1, 1, 2, 1, 0, 0}, "1x1 kernel, stride 2x1");
        test_views();
        test_empty();
        test_errors();
        test_partial_autograd();
    });
}
