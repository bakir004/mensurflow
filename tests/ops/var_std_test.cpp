// var (and later std) along an axis against the two-pass double reference in
// tests/reference/reduction.hpp: every axis, keepdim and correction, the count - correction <= 0
// rule, special values, a numerically hard case where the naive formulas lose precision,
// finite-difference and reference checks of the gradient, autograd, views, empty shapes and errors.
#include "gradient_check.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/reduction.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;

namespace {

const float nan = std::numeric_limits<float>::quiet_NaN();

Shape reduced_shape(const Shape& shape, int axis, bool keepdim) {
    Shape out = shape;
    if (keepdim) {
        out[static_cast<std::size_t>(axis)] = 1;
    } else {
        out.erase(out.begin() + axis);
    }
    return out;
}

std::string describe(const Shape& shape, int axis, bool keepdim, std::int64_t correction) {
    return to_string(shape) + " axis " + std::to_string(axis) + (keepdim ? " keepdim" : "") +
           " correction " + std::to_string(correction);
}

const std::vector<Shape> shapes = {{7},    {5, 3},  {3, 5},   {1, 9},      {4, 1},
                                   {2, 1}, {17, 31}, {2, 3, 4}, {2, 3, 4, 5}};

void test_var_against_reference() {
    std::uint32_t seed = 1;
    for (const Shape& shape : shapes) {
        const auto x = test::random_values(numel(shape), seed++, -2.0f, 3.0f);
        for (int axis = 0; axis < static_cast<int>(shape.size()); ++axis) {
            for (bool keepdim : {false, true}) {
                for (std::int64_t correction : {0, 1, 2}) {
                    const std::string label = "var " + describe(shape, axis, keepdim, correction);
                    const Shape out_shape = reduced_shape(shape, axis, keepdim);
                    const Tensor tx = Tensor::from_vector(x, shape);
                    const Tensor y = var(tx, axis, keepdim, correction, Execution::CPU);
                    test::require(y.shape() == out_shape, label + ": shape");
                    test::require_close(y.to_vector<float>(),
                                        reference::var_axis(x, shape, axis, correction), 1e-7,
                                        1e-5, label + ": forward");
                    // The negative spelling of the axis is the same reduction.
                    test::require_close(
                        var(tx, axis - static_cast<int>(shape.size()), keepdim, correction)
                            .to_vector<float>(),
                        y.to_vector<float>(), 0.0, 0.0, label + ": negative axis");

                    const auto g = test::random_values(numel(out_shape), seed++);
                    const auto want =
                        reference::var_axis_backward(x, shape, axis, correction, g);
                    test::require_close(var_backward(Tensor::from_vector(g, out_shape), tx, axis,
                                                     keepdim, correction)
                                            .to_vector<float>(),
                                        want, 1e-6, 1e-5, label + ": backward");
                    Tensor leaf = Tensor::from_vector(x, shape);
                    leaf.set_requires_grad();
                    test::weighted_loss(var(leaf, axis, keepdim, correction), g).backward();
                    test::require_close(leaf.grad().to_vector<float>(), want, 1e-6, 1e-5,
                                        label + ": autograd");
                }
            }
        }
    }
}

// count - correction <= 0 gives NaN (value and gradient); an empty axis is the extreme case.
void test_var_degenerate_counts() {
    const Tensor column = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3, 1});
    const auto one_sample = var(column, 1).to_vector<float>();
    test::require(std::isnan(one_sample[0]) && std::isnan(one_sample[2]),
                  "one element with correction 1 is NaN");
    test::require_close(var(column, 1, false, 0).to_vector<float>(), std::vector<float>(3, 0.0f),
                        0.0, 0.0, "one element with correction 0 is 0");
    const auto too_large = var(Tensor::ones({2, 3}), 1, false, 3).to_vector<float>();
    test::require(std::isnan(too_large[0]) && std::isnan(too_large[1]),
                  "correction equal to the count is NaN");
    test::require(std::isnan(var(Tensor::ones({2, 3}), 1, false, 7).to_vector<float>()[0]),
                  "correction above the count is NaN");
    const Tensor empty_axis = var(Tensor::empty({2, 0}), 1, false, 0);
    test::require(empty_axis.shape() == (Shape{2}) &&
                      std::isnan(empty_axis.to_vector<float>()[0]),
                  "an empty axis is NaN, even with correction 0");
    const auto grad = var_backward(Tensor::ones({3}), column, 1).to_vector<float>();
    test::require(std::isnan(grad[0]), "gradient of a NaN variance is NaN");
}

void test_var_special_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const Tensor x = Tensor::from_vector(
        std::vector<float>{5, 5, 5, 5, 1, nan, 2, 3, 1, inf, 2, 3, -1.5f, 2.5f, -1.5f, 2.5f},
        {4, 4});
    const auto v = var(x, 1, false, 0).to_vector<float>();
    test::require(v[0] == 0.0f, "a constant row has variance exactly 0");
    test::require(std::isnan(v[1]) && std::isnan(v[2]), "NaN and infinity give NaN");
    test::require(v[3] == 4.0f, "exact small case");
}

// Large mean, small spread: 1e8 + 8 * {0, 1, 2, 3} (exact in float32; the spacing of floats near
// 1e8 is 8). Population variance 80. The naive sum(x^2) - n * mean^2 formula subtracts two numbers
// near 1e19 and loses almost all digits even in double; Welford must not.
void test_var_large_mean() {
    const std::int64_t n = 1000;
    std::vector<float> x(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = 1e8f + 8.0f * static_cast<float>(i % 4);
    }
    const double want = 80.0 * static_cast<double>(n) / static_cast<double>(n - 1);
    test::require(std::abs(reference::var_axis(x, {n}, 0, 1)[0] - want) < 1e-9 * want,
                  "the two-pass reference is exact here");

    double sum = 0.0, squares = 0.0;
    float sum_f = 0.0f, squares_f = 0.0f;
    for (float value : x) {
        sum += value;
        squares += static_cast<double>(value) * value;
        sum_f += value;
        squares_f += value * value;
    }
    const double naive_double =
        (squares - sum * sum / static_cast<double>(n)) / static_cast<double>(n - 1);
    const double naive_float = static_cast<double>(
        (squares_f - sum_f * sum_f / static_cast<float>(n)) / static_cast<float>(n - 1));
    test::require(std::abs(naive_double - want) > 1e-3 * want,
                  "the case is hard: naive double sum of squares is off by more than 0.1% (" +
                      std::to_string(naive_double) + ")");
    test::require(!(std::abs(naive_float - want) < 0.5 * want),
                  "the case is hard: naive float sum of squares is far off (" +
                      std::to_string(naive_float) + ")");

    const float got = var(Tensor::from_vector(x, {n}), 0).item<float>();
    test::require(std::abs(static_cast<double>(got) - want) <= 1e-6 * want,
                  "Welford keeps the precision: got " + std::to_string(got) + ", expected " +
                      std::to_string(want));
    // The same along a strided axis, with a mean of -1e8 in the other column.
    std::vector<float> two_columns;
    for (float value : x) {
        two_columns.push_back(value);
        two_columns.push_back(-value);
    }
    const auto both = var(Tensor::from_vector(two_columns, {n, 2}), 0).to_vector<float>();
    test::require(std::abs(both[0] - want) <= 1e-6 * want &&
                      std::abs(both[1] - want) <= 1e-6 * want,
                  "Welford along a strided axis");

    // A mean of 1e4 with a spread of about 1e-2 (float spacing near 1e4 is about 1e-3).
    auto noise = test::random_values(4096, 3, -0.01f, 0.01f);
    for (float& value : noise) {
        value += 1e4f;
    }
    const double want_small = reference::var_axis(noise, {4096}, 0, 0)[0];
    test::require_close(std::vector<float>{var(Tensor::from_vector(noise, {4096}), 0, false, 0)
                                               .item<float>()},
                        std::vector<double>{want_small}, 0.0, 1e-5, "mean 1e4, spread 1e-2");
}

void test_var_finite_differences() {
    const Shape shape{3, 4, 5};
    const auto x = test::random_values(numel(shape), 21);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::int64_t correction : {0, 1}) {
            const Shape out_shape = reduced_shape(shape, axis, false);
            const auto g = test::random_values(numel(out_shape), 22);
            auto forward = [&](const std::vector<float>& v) {
                return var(Tensor::from_vector(v, shape), axis, false, correction)
                    .to_vector<float>();
            };
            // var is quadratic in x, so central differences are exact up to rounding.
            test::require_close(var_backward(Tensor::from_vector(g, out_shape),
                                             Tensor::from_vector(x, shape), axis, false,
                                             correction)
                                    .to_vector<float>(),
                                test::numeric_gradient(forward, x, g, 0.05f), 1e-3, 1e-3,
                                "var finite differences, axis " + std::to_string(axis));
        }
    }
}

void test_var_views_and_errors() {
    const auto values = test::random_values(6 * 10, 31);
    const Tensor base = Tensor::from_vector(values, {6, 10});
    const Tensor transposed = base.transpose(0, 1); // [10, 6]
    test::require_close(var(transposed, 1).to_vector<float>(),
                        reference::var_axis(transposed.to_vector<float>(), {10, 6}, 1, 1), 1e-7,
                        1e-5, "transposed view");
    const Tensor strided = base.slice(1, 1, 10, 3); // [6, 3]
    test::require_close(var(strided, 0, true).to_vector<float>(),
                        reference::var_axis(strided.to_vector<float>(), {6, 3}, 0, 1), 1e-7,
                        1e-5, "strided view");
    test::require(var(Tensor::empty({0, 4}), 1).shape() == (Shape{0}), "no output positions");
    const auto empty_rows = var(Tensor::empty({0, 4}), 0, false, 0).to_vector<float>();
    test::require(empty_rows.size() == 4 && std::isnan(empty_rows[3]), "reducing an empty axis 0");

    using E = std::invalid_argument;
    const Tensor x = Tensor::zeros({2, 3});
    test::require_throws<E>([&] { var(Tensor(), 0); }, "undefined");
    test::require_throws<E>([&] { var(Tensor::zeros({}), 0); }, "scalar");
    test::require_throws<E>([&] { var(x, 2); }, "axis too large");
    test::require_throws<E>([&] { var(x, -3); }, "axis too negative");
    test::require_throws<E>([&] { var(x, 1, false, -1); }, "negative correction");
    test::require_throws<E>([&] { var_backward(Tensor::zeros({3}), x, 1); },
                            "gradient of the wrong shape");
    test::require_throws<E>([&] { var_backward(Tensor::zeros({2}), x, 1, true); },
                            "gradient without the kept dimension");
    test::require_throws<E>([&] { var_backward(Tensor(), x, 1); }, "undefined gradient");
}

} // namespace

int main() {
    return test::run("var_std_test", [] {
        test_var_against_reference();
        test_var_degenerate_counts();
        test_var_special_values();
        test_var_large_mean();
        test_var_finite_differences();
        test_var_views_and_errors();
    });
}
