// avg_pool2d forward and backward against the direct reference in tests/reference/pooling.hpp:
// the divisor convention (padding counts, count_include_pad = true), a finite-difference check of
// the gradient, autograd, views, empty shapes and errors.
#include "gradient_check.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/pooling.hpp"
#include "test_util.hpp"

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
    {"ResNet stem shape, small", {1, 4, 15, 15, 3, 3, 2, 2, 1, 1}},
};

Shape x_shape(const Pool2dSizes& s) { return {s.n, s.c, s.h, s.w}; }

Shape y_shape(const Pool2dSizes& s) { return {s.n, s.c, s.h_out(), s.w_out()}; }

Size2d kernel_of(const Pool2dSizes& s) { return {s.kh, s.kw}; }

Size2d stride_of(const Pool2dSizes& s) { return {s.stride_h, s.stride_w}; }

Size2d padding_of(const Pool2dSizes& s) { return {s.pad_h, s.pad_w}; }

void test_against_reference(const Case& c) {
    const Pool2dSizes& s = c.s;
    const std::string label = std::string(c.name) + ": ";
    const auto x = test::random_values(s.x_size(), 1);
    const auto gy = test::random_values(s.y_size(), 2);

    const Tensor y = avg_pool2d(Tensor::from_vector(x, x_shape(s)), kernel_of(s), stride_of(s),
                                padding_of(s), Execution::CPU);
    test::require(y.shape() == y_shape(s), label + "output shape");
    test::require_close(y.to_vector<float>(), reference::avg_pool2d(x, s), 1e-6, 1e-6,
                        label + "forward");

    const auto want_gx = reference::avg_pool2d_backward(gy, s);
    test::require_close(avg_pool2d_backward(Tensor::from_vector(gy, y_shape(s)), x_shape(s),
                                            kernel_of(s), stride_of(s), padding_of(s))
                            .to_vector<float>(),
                        want_gx, 1e-6, 1e-5, label + "backward");

    Tensor leaf = Tensor::from_vector(x, x_shape(s));
    leaf.set_requires_grad();
    test::weighted_loss(avg_pool2d(leaf, kernel_of(s), stride_of(s), padding_of(s)), gy)
        .backward();
    test::require_close(leaf.grad().to_vector<float>(), want_gx, 1e-6, 1e-5, label + "autograd");
}

void test_finite_differences(const Pool2dSizes& s, const std::string& label) {
    const auto x = test::random_values(s.x_size(), 7);
    const auto gy = test::random_values(s.y_size(), 8);
    auto forward = [&](const std::vector<float>& v) {
        return avg_pool2d(Tensor::from_vector(v, x_shape(s)), kernel_of(s), stride_of(s),
                          padding_of(s))
            .to_vector<float>();
    };
    // Linear in x, so central differences are exact up to rounding.
    test::require_close(avg_pool2d_backward(Tensor::from_vector(gy, y_shape(s)), x_shape(s),
                                            kernel_of(s), stride_of(s), padding_of(s))
                            .to_vector<float>(),
                        test::numeric_gradient(forward, x, gy, 0.1f), 1e-3, 0.0,
                        label + ": finite differences");
}

// Padded cells count in the divisor (count_include_pad = true): a 3x3 window over all-ones with
// padding 1 averages to 4/9 in a corner, 6/9 on an edge and 1 in the middle, not 1 everywhere.
void test_divisor_includes_padding() {
    const Tensor ones = Tensor::ones({1, 1, 3, 3});
    const std::vector<float> want = {4.0f / 9, 6.0f / 9, 4.0f / 9, 6.0f / 9, 1.0f,
                                     6.0f / 9, 4.0f / 9, 6.0f / 9, 4.0f / 9};
    test::require_close(avg_pool2d(ones, {3, 3}, {1, 1}, {1, 1}).to_vector<float>(), want, 0.0,
                        1e-6, "padding counts in the divisor");
    // The gradient of the corner output reaches only its 4 input cells, each with 1/9.
    std::vector<float> gy(9, 0.0f);
    gy[0] = 1.0f;
    const std::vector<float> want_gx = {1.0f / 9, 1.0f / 9, 0, 1.0f / 9, 1.0f / 9, 0, 0, 0, 0};
    test::require_close(avg_pool2d_backward(Tensor::from_vector(gy, {1, 1, 3, 3}), ones.shape(),
                                            {3, 3}, {1, 1}, {1, 1})
                            .to_vector<float>(),
                        want_gx, 0.0, 1e-6, "padding takes no gradient");
    // NaN and inf propagate to every window that contains them.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor with_nan =
        Tensor::from_vector(std::vector<float>{nan, 1, 2, 3}, {1, 1, 1, 4});
    const auto pooled = avg_pool2d(with_nan, {1, 2}).to_vector<float>();
    test::require(std::isnan(pooled[0]) && pooled[1] == 2.5f, "NaN stays in its window");
}

void test_views() {
    const Pool2dSizes s{2, 3, 6, 5, 3, 2, 2, 2, 1, 1};
    const Tensor nhwc = Tensor::from_vector(test::random_values(s.x_size(), 21),
                                            {s.n, s.h, s.w, s.c});
    const Tensor x = nhwc.permute({0, 3, 1, 2});
    test::require(!x.is_contiguous(), "permuted view");
    test::require_close(avg_pool2d(x, kernel_of(s), stride_of(s), padding_of(s)).to_vector<float>(),
                        reference::avg_pool2d(x.to_vector<float>(), s), 1e-6, 1e-6, "view forward");
    const auto gy = test::random_values(s.y_size(), 22);
    const Tensor gy_t = Tensor::from_vector(gy, {s.n, s.c, s.w_out(), s.h_out()}).transpose(2, 3);
    test::require_close(avg_pool2d_backward(gy_t, x_shape(s), kernel_of(s), stride_of(s),
                                            padding_of(s))
                            .to_vector<float>(),
                        reference::avg_pool2d_backward(gy_t.to_vector<float>(), s), 1e-6, 1e-5,
                        "backward of a transposed gradient");
}

void test_empty() {
    test::require(avg_pool2d(Tensor::empty({0, 3, 4, 4}), {2, 2}).shape() == (Shape{0, 3, 2, 2}),
                  "empty batch");
    test::require(avg_pool2d(Tensor::empty({2, 0, 4, 4}), {2, 2}).shape() == (Shape{2, 0, 2, 2}),
                  "no channels");
    test::require(avg_pool2d_backward(Tensor::empty({0, 3, 2, 2}), Shape{0, 3, 4, 4}, {2, 2})
                          .shape() == (Shape{0, 3, 4, 4}),
                  "empty backward");
}

void test_errors() {
    const Tensor x = Tensor::zeros({1, 2, 5, 5});
    using E = std::invalid_argument;
    test::require_throws<E>([&] { avg_pool2d(Tensor(), {2, 2}); }, "undefined");
    test::require_throws<E>([&] { avg_pool2d(Tensor::zeros({2, 5, 5}), {2, 2}); }, "3-D input");
    test::require_throws<E>([&] { avg_pool2d(x, {2, 0}); }, "kernel 0");
    test::require_throws<E>([&] { avg_pool2d(x, {2, 2}, {1, 0}); }, "one zero stride");
    test::require_throws<E>([&] { avg_pool2d(x, {2, 2}, {1, 1}, {0, -1}); }, "negative padding");
    test::require_throws<E>([&] { avg_pool2d(x, {2, 2}, {1, 1}, {0, 2}); },
                            "padding above half the kernel");
    test::require_throws<E>([&] { avg_pool2d(x, {2, 6}); }, "kernel wider than the input");
    test::require_throws<E>([&] { avg_pool2d(Tensor::zeros({1, 1, 4, 0}), {1, 1}); },
                            "zero width");
    const Tensor gy = Tensor::zeros({1, 2, 2, 2});
    test::require_throws<E>([&] { avg_pool2d_backward(gy, x.shape(), {3, 3}); },
                            "gradient shape for another kernel");
    test::require_throws<E>([&] { avg_pool2d_backward(gy, Shape{1, 2, 5}, {2, 2}); },
                            "input_shape rank");
    test::require_throws<E>([&] { avg_pool2d_backward(Tensor(), x.shape(), {2, 2}); },
                            "undefined gradient");
}

} // namespace

int main() {
    return test::run("avg_pool2d_test", [] {
        for (const Case& c : cases) {
            test_against_reference(c);
        }
        test_finite_differences({2, 2, 5, 6, 3, 2, 2, 1, 1, 1}, "stride 2x1, padding 1");
        test_finite_differences({1, 2, 4, 4, 2, 2, 1, 1, 0, 0}, "overlapping 2x2");
        test_finite_differences({1, 1, 3, 4, 3, 4, 1, 1, 0, 0}, "kernel as large as the input");
        test_divisor_includes_padding();
        test_views();
        test_empty();
        test_errors();
    });
}
