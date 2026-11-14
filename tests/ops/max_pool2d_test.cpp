// max_pool2d forward and backward against the direct reference in tests/reference/pooling.hpp:
// values, indices, tie and NaN rules, padding, a finite-difference check of the gradient,
// autograd, views, empty shapes and errors.
#include "backends/cpu/ops/conv/pooling.hpp"
#include "backends/cpu/threads.hpp"
#include "gradient_check.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/pooling.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
using reference::Pool2dSizes;

namespace {

struct Case {
    const char* name;
    Pool2dSizes s;
};

// {n, c, h, w, kh, kw, stride_h, stride_w, pad_h, pad_w}
const Case cases[] = {
    {"odd non-square, stride and padding", {2, 3, 7, 5, 3, 2, 2, 1, 1, 1}},
    {"stride = kernel (the default)", {2, 2, 8, 6, 2, 2, 2, 2, 0, 0}},
    {"overlapping windows", {1, 2, 6, 7, 3, 3, 1, 1, 1, 1}},
    {"stride larger than the kernel", {2, 2, 9, 10, 2, 3, 3, 4, 0, 0}},
    {"1x1 kernel", {2, 3, 4, 5, 1, 1, 1, 1, 0, 0}},
    {"1x1 kernel, stride 2", {1, 2, 5, 6, 1, 1, 2, 2, 0, 0}},
    {"kernel as large as the input", {2, 3, 5, 4, 5, 4, 1, 1, 0, 0}},
    {"kernel as large as the padded input", {1, 2, 3, 2, 5, 4, 1, 1, 1, 1}},
    {"batch 1, one channel", {1, 1, 6, 6, 3, 3, 2, 2, 1, 1}},
    {"stride does not divide the span", {1, 2, 10, 7, 3, 2, 3, 2, 1, 0}},
};

Shape x_shape(const Pool2dSizes& s) { return {s.n, s.c, s.h, s.w}; }

Shape y_shape(const Pool2dSizes& s) { return {s.n, s.c, s.h_out(), s.w_out()}; }

Size2d kernel_of(const Pool2dSizes& s) { return {s.kh, s.kw}; }

Size2d stride_of(const Pool2dSizes& s) { return {s.stride_h, s.stride_w}; }

Size2d padding_of(const Pool2dSizes& s) { return {s.pad_h, s.pad_w}; }

void require_indices(const std::vector<std::int64_t>& got, const std::vector<std::int64_t>& want,
                     const std::string& label) {
    test::require(got.size() == want.size(), label + ": index count");
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(got[i] == want[i], label + ": index " + std::to_string(i) + " is " +
                                             std::to_string(got[i]) + ", expected " +
                                             std::to_string(want[i]));
    }
}

// Values 0.1 apart in a shuffled order, so every window has one clear maximum that a small
// perturbation cannot change.
std::vector<float> distinct_values(std::int64_t n, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.1f * static_cast<float>(i) - 0.05f * static_cast<float>(n);
    }
    const auto keys = test::random_values(n, seed);
    std::vector<std::size_t> order(values.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return keys[a] < keys[b] || (keys[a] == keys[b] && a < b);
    });
    std::vector<float> shuffled(values.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        shuffled[i] = values[order[i]];
    }
    return shuffled;
}

void test_against_reference(const Case& c) {
    const Pool2dSizes& s = c.s;
    const std::string label = std::string(c.name) + ": ";
    // Values rounded to a coarse grid, so windows also contain ties.
    auto x = test::random_values(s.x_size(), 1);
    for (float& value : x) {
        value = static_cast<float>(static_cast<int>(value * 4.0f)) / 4.0f;
    }
    const auto gy = test::random_values(s.y_size(), 2);
    const auto want = reference::max_pool2d(x, s);

    const Tensor tx = Tensor::from_vector(x, x_shape(s));
    const MaxPool2dResult result =
        max_pool2d(tx, kernel_of(s), stride_of(s), padding_of(s), Execution::CPU);
    test::require(result.output.shape() == y_shape(s) && result.indices.shape() == y_shape(s),
                  label + "output shapes");
    test::require(result.indices.dtype() == DType::I64, label + "indices are I64");
    test::require_close(result.output.to_vector<float>(), want.output, 0.0, 0.0,
                        label + "forward values");
    require_indices(result.indices.to_vector<std::int64_t>(), want.indices,
                    label + "forward indices");

    const auto want_gx = reference::max_pool2d_backward(gy, want.indices, s);
    const Tensor tgy = Tensor::from_vector(gy, y_shape(s));
    test::require_close(max_pool2d_backward(tgy, result.indices, x_shape(s)).to_vector<float>(),
                        want_gx, 1e-6, 1e-6, label + "backward");

    Tensor leaf = Tensor::from_vector(x, x_shape(s));
    leaf.set_requires_grad();
    test::weighted_loss(max_pool2d(leaf, kernel_of(s), stride_of(s), padding_of(s)).output, gy)
        .backward();
    test::require_close(leaf.grad().to_vector<float>(), want_gx, 1e-6, 1e-6,
                        label + "autograd");
}

void test_finite_differences(const Pool2dSizes& s, const std::string& label) {
    const auto x = distinct_values(s.x_size(), 7);
    const auto gy = test::random_values(s.y_size(), 8);
    auto forward = [&](const std::vector<float>& v) {
        return max_pool2d(Tensor::from_vector(v, x_shape(s)), kernel_of(s), stride_of(s),
                          padding_of(s))
            .output.to_vector<float>();
    };
    const MaxPool2dResult result =
        max_pool2d(Tensor::from_vector(x, x_shape(s)), kernel_of(s), stride_of(s), padding_of(s));
    test::require_close(max_pool2d_backward(Tensor::from_vector(gy, y_shape(s)), result.indices,
                                            x_shape(s))
                            .to_vector<float>(),
                        test::numeric_gradient(forward, x, gy, 0.01f), 1e-3, 0.0,
                        label + ": finite differences");
}

void test_ties_padding_and_nan() {
    // All values equal: every window picks its first cell in row-major order.
    const Tensor flat = Tensor::full({1, 1, 4, 4}, 2.0);
    require_indices(max_pool2d(flat, {2, 2}).indices.to_vector<std::int64_t>(), {0, 2, 8, 10},
                    "equal values pick the first cell");
    // Two maxima in one window, at (0, 1) and (1, 0): (0, 1) comes first in row-major order.
    const Tensor two =
        Tensor::from_vector(std::vector<float>{1, 5, 5, 0}, {1, 1, 2, 2});
    require_indices(max_pool2d(two, {2, 2}).indices.to_vector<std::int64_t>(), {1},
                    "tie goes to the first in row-major order");

    // All values negative and padded: the padding (zero) never wins.
    const Tensor negative = Tensor::full({1, 1, 3, 3}, -4.0);
    const MaxPool2dResult padded = max_pool2d(negative, {3, 3}, {1, 1}, {1, 1});
    test::require_close(padded.output.to_vector<float>(), std::vector<float>(9, -4.0f), 0.0, 0.0,
                        "padding never wins");
    // The first window covers rows -1..1 and columns -1..1; its first real cell is (0, 0).
    test::require(padded.indices.to_vector<std::int64_t>()[0] == 0, "first real cell of a window");

    // NaN: beats every number (including +inf) and the first NaN of the window wins.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const Tensor with_nan =
        Tensor::from_vector(std::vector<float>{inf, nan, 1, 2,     //
                                               nan, 3, 4, -inf},   //
                            {1, 1, 2, 4});
    const MaxPool2dResult r = max_pool2d(with_nan, {2, 2});
    const auto values = r.output.to_vector<float>();
    test::require(std::isnan(values[0]) && values[1] == 4.0f, "NaN propagates to its window only");
    require_indices(r.indices.to_vector<std::int64_t>(), {1, 6}, "first NaN wins");
    // A NaN in the first cell is kept even if a later cell is NaN too.
    const Tensor nan_first =
        Tensor::from_vector(std::vector<float>{nan, nan, 9, 1}, {1, 1, 2, 2});
    require_indices(max_pool2d(nan_first, {2, 2}).indices.to_vector<std::int64_t>(), {0},
                    "a leading NaN wins");
}

void test_overlap_accumulates() {
    // 3x3 input, 2x2 windows with stride 1: the centre is the maximum of all four windows, so its
    // gradient is the sum of the four upstream values.
    const Tensor x =
        Tensor::from_vector(std::vector<float>{0, 0, 0, 0, 9, 0, 0, 0, 0}, {1, 1, 3, 3});
    const MaxPool2dResult r = max_pool2d(x, {2, 2}, {1, 1});
    require_indices(r.indices.to_vector<std::int64_t>(), {4, 4, 4, 4}, "centre wins everywhere");
    const Tensor gy = Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {1, 1, 2, 2});
    test::require_close(max_pool2d_backward(gy, r.indices, x.shape()).to_vector<float>(),
                        std::vector<float>{0, 0, 0, 0, 10, 0, 0, 0, 0}, 0.0, 0.0,
                        "overlapping windows add up");
}

void test_autograd_saves_indices() {
    const Pool2dSizes s{1, 2, 4, 4, 2, 2, 2, 2, 0, 0};
    const auto x = distinct_values(s.x_size(), 9);
    const auto gy = test::random_values(s.y_size(), 10);
    Tensor leaf = Tensor::from_vector(x, x_shape(s));
    leaf.set_requires_grad();
    MaxPool2dResult r = max_pool2d(leaf, kernel_of(s), stride_of(s), padding_of(s));
    // Changing the returned indices must not move the gradient.
    r.indices.copy_from(Tensor::zeros(y_shape(s), DType::I64));
    test::weighted_loss(r.output, gy).backward();
    test::require_close(leaf.grad().to_vector<float>(),
                        reference::max_pool2d_backward(gy, reference::max_pool2d(x, s).indices, s),
                        0.0, 0.0, "autograd uses its own copy of the indices");
}

void test_views() {
    const Pool2dSizes s{2, 3, 6, 5, 3, 2, 2, 2, 1, 1};
    // Stored NHWC, permuted to NCHW.
    const Tensor nhwc = Tensor::from_vector(test::random_values(s.x_size(), 21),
                                            {s.n, s.h, s.w, s.c});
    const Tensor x = nhwc.permute({0, 3, 1, 2});
    test::require(!x.is_contiguous(), "permuted view");
    const auto want = reference::max_pool2d(x.to_vector<float>(), s);
    const MaxPool2dResult r = max_pool2d(x, kernel_of(s), stride_of(s), padding_of(s));
    test::require_close(r.output.to_vector<float>(), want.output, 0.0, 0.0, "view values");
    require_indices(r.indices.to_vector<std::int64_t>(), want.indices, "view indices");

    // A transposed gradient view goes through the backward op too.
    const auto gy = test::random_values(s.y_size(), 22);
    const Tensor gy_t = Tensor::from_vector(gy, {s.n, s.c, s.w_out(), s.h_out()}).transpose(2, 3);
    test::require_close(max_pool2d_backward(gy_t, r.indices, x_shape(s)).to_vector<float>(),
                        reference::max_pool2d_backward(gy_t.to_vector<float>(), want.indices, s),
                        1e-6, 1e-6, "backward of views");
}

void test_empty() {
    const MaxPool2dResult r = max_pool2d(Tensor::empty({0, 3, 4, 4}), {2, 2});
    test::require(r.output.shape() == (Shape{0, 3, 2, 2}) && r.indices.numel() == 0,
                  "empty batch");
    test::require(max_pool2d(Tensor::empty({2, 0, 4, 4}), {2, 2}).output.shape() ==
                      (Shape{2, 0, 2, 2}),
                  "no channels");
    test::require(max_pool2d_backward(Tensor::empty({0, 3, 2, 2}),
                                      Tensor::empty({0, 3, 2, 2}, DType::I64), Shape{0, 3, 4, 4})
                          .shape() == (Shape{0, 3, 4, 4}),
                  "empty backward");
}

void test_errors() {
    const Tensor x = Tensor::zeros({1, 2, 5, 5});
    using E = std::invalid_argument;
    test::require_throws<E>([&] { max_pool2d(Tensor(), {2, 2}); }, "undefined");
    test::require_throws<E>([&] { max_pool2d(Tensor::zeros({2, 5, 5}), {2, 2}); }, "3-D input");
    test::require_throws<E>([&] { max_pool2d(x, {0, 2}); }, "kernel 0");
    test::require_throws<E>([&] { max_pool2d(x, {2, 2}, {0, 1}); }, "one zero stride");
    test::require_throws<E>([&] { max_pool2d(x, {2, 2}, {-1, -1}); }, "negative stride");
    test::require_throws<E>([&] { max_pool2d(x, {2, 2}, {1, 1}, {-1, 0}); }, "negative padding");
    test::require_throws<E>([&] { max_pool2d(x, {3, 3}, {1, 1}, {2, 0}); },
                            "padding above half the kernel");
    test::require_throws<E>([&] { max_pool2d(x, {6, 2}); }, "kernel taller than the input");
    test::require_throws<E>([&] { max_pool2d(Tensor::zeros({1, 1, 0, 4}), {1, 1}); },
                            "zero height");

    const Tensor gy = Tensor::zeros({1, 2, 2, 2});
    const Tensor indices = Tensor::zeros({1, 2, 2, 2}, DType::I64);
    const Tensor float_indices = Tensor::zeros({1, 2, 2, 2});
    test::require_throws<E>([&] { max_pool2d_backward(gy, float_indices, x.shape()); },
                            "F32 indices");
    test::require_throws<E>(
        [&] { max_pool2d_backward(gy, Tensor::zeros({1, 2, 2, 3}, DType::I64), x.shape()); },
        "indices shape");
    test::require_throws<E>([&] { max_pool2d_backward(gy, indices, Shape{1, 3, 5, 5}); },
                            "input_shape channels");
    test::require_throws<E>([&] { max_pool2d_backward(gy, indices, Shape{1, 2, 5}); },
                            "input_shape rank");
    const Tensor out_of_range =
        Tensor::from_vector(std::vector<std::int64_t>{0, 1, 2, 25, 0, 0, 0, 0}, {1, 2, 2, 2});
    test::require_throws<E>([&] { max_pool2d_backward(gy, out_of_range, x.shape()); },
                            "index outside the plane");
    const Tensor negative_index =
        Tensor::from_vector(std::vector<std::int64_t>{0, -1, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 2});
    test::require_throws<E>([&] { max_pool2d_backward(gy, negative_index, x.shape()); },
                            "negative index");
}

cpu::Pool2dShape kernel_shape(const Pool2dSizes& p) {
    cpu::Pool2dShape s;
    s.n = p.n;
    s.c = p.c;
    s.h = p.h;
    s.w = p.w;
    s.kh = p.kh;
    s.kw = p.kw;
    s.stride_h = p.stride_h;
    s.stride_w = p.stride_w;
    s.pad_h = p.pad_h;
    s.pad_w = p.pad_w;
    s.h_out = p.h_out();
    s.w_out = p.w_out();
    return s;
}

// The parallel kernels give exactly the sequential results (values, indices and gradients) at
// every thread count, around the cutoff and for few large or many small planes.
void test_parallel() {
    const std::vector<Pool2dSizes> sizes = {
        // 2x2/2 on one plane: 127, 128 and 129 output rows of 128 windows of 4 cells, around
        // the point where the work first splits (2 * 2^15 cells).
        {1, 1, 254, 256, 2, 2, 2, 2, 0, 0},
        {1, 1, 256, 256, 2, 2, 2, 2, 0, 0},
        {1, 1, 258, 256, 2, 2, 2, 2, 0, 0},
        {1, 3, 224, 224, 3, 3, 2, 2, 1, 1}, // few large planes
        {2, 64, 57, 55, 3, 3, 2, 2, 1, 1},  // ResNet stem pool, odd sizes
        {32, 3, 30, 30, 2, 2, 2, 2, 0, 0},
        {2, 512, 7, 7, 7, 7, 7, 7, 0, 0},   // global pooling
        {3, 5, 17, 13, 4, 3, 1, 2, 2, 1},   // overlapping windows
    };
    const int original = cpu::max_threads();
    for (const Pool2dSizes& p : sizes) {
        const cpu::Pool2dShape s = kernel_shape(p);
        auto x = test::random_values(p.x_size(), 71);
        for (float& value : x) {
            value = std::round(value * 8.0f); // ties
        }
        x[x.size() / 3] = std::numeric_limits<float>::quiet_NaN();
        const auto gy = test::random_values(p.y_size(), 72);
        const auto outputs = static_cast<std::size_t>(p.y_size());
        std::vector<float> y_seq(outputs), gx_seq(static_cast<std::size_t>(p.x_size()));
        std::vector<std::int64_t> i_seq(outputs);
        cpu::max_pool2d_f32_sequential(x.data(), y_seq.data(), i_seq.data(), s);
        cpu::max_pool2d_backward_f32_sequential(gy.data(), i_seq.data(), gx_seq.data(), s);
        for (int threads : {1, 2, 3, 7, original}) {
            cpu::set_max_threads(threads);
            const std::string label = "parallel max_pool2d " + to_string(x_shape(p)) + ", " +
                                      std::to_string(threads) + " threads";
            std::vector<float> y(outputs), gx(gx_seq.size());
            std::vector<std::int64_t> idx(outputs);
            cpu::max_pool2d_f32_parallel(x.data(), y.data(), idx.data(), s);
            cpu::max_pool2d_backward_f32_parallel(gy.data(), idx.data(), gx.data(), s);
            test::require(test::matches_f32(y, y_seq), label + ": values");
            require_indices(idx, i_seq, label + ": indices");
            test::require(test::matches_f32(gx, gx_seq), label + ": gradient");
        }
    }
    cpu::set_max_threads(original);
}

} // namespace

int main() {
    return test::run("max_pool2d_test", [] {
        test_parallel();
        for (const Case& c : cases) {
            test_against_reference(c);
        }
        test_finite_differences({2, 2, 5, 6, 3, 2, 2, 1, 1, 1}, "stride 2x1, padding 1");
        test_finite_differences({1, 2, 4, 4, 2, 2, 1, 1, 0, 0}, "overlapping 2x2");
        test_finite_differences({1, 1, 3, 4, 3, 4, 1, 1, 0, 0}, "kernel as large as the input");
        test_ties_padding_and_nan();
        test_overlap_accumulates();
        test_autograd_saves_indices();
        test_views();
        test_empty();
        test_errors();
    });
}
