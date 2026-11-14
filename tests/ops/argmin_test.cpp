// argmin against the reference in tests/reference/reduction.hpp: odd and non-square shapes, the
// tie and NaN rules shared with argmax, views, empty shapes and errors.
#include "backends/cpu/ops/reduction/reduction.hpp"
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

void require_indices(const std::vector<std::int64_t>& got, const std::vector<std::int64_t>& want,
                     const std::string& label) {
    test::require(got.size() == want.size(), label + ": index count");
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(got[i] == want[i], label + ": row " + std::to_string(i) + " gives " +
                                             std::to_string(got[i]) + ", expected " +
                                             std::to_string(want[i]));
    }
}

// The reference applied to every row of a contiguous [rows, width] buffer.
std::vector<std::int64_t> reference_rows(const std::vector<float>& x, std::int64_t width) {
    std::vector<std::int64_t> out;
    for (std::size_t start = 0; start < x.size(); start += static_cast<std::size_t>(width)) {
        out.push_back(reference::argmin(x.data() + start, width));
    }
    return out;
}

void test_shapes() {
    const std::vector<Shape> shapes = {{1},       {7},       {1, 1},     {3, 1},   {1, 9},
                                       {5, 3},    {3, 5},    {17, 31},   {2, 3, 4}, {2, 3, 4, 5},
                                       {4, 1000}, {1, 4097}, {3, 2, 1}};
    std::uint32_t seed = 1;
    for (const Shape& shape : shapes) {
        // Values on a coarse grid so most rows contain ties.
        auto x = test::random_values(numel(shape), seed++);
        for (float& value : x) {
            value = std::round(value * 3.0f);
        }
        const Tensor result = argmin(Tensor::from_vector(x, shape), -1, Execution::CPU);
        test::require(result.dtype() == DType::I64 &&
                          result.shape() == Shape(shape.begin(), shape.end() - 1),
                      "argmin output dtype and shape for " + to_string(shape));
        require_indices(result.to_vector<std::int64_t>(), reference_rows(x, shape.back()),
                        "argmin " + to_string(shape));
        // The positive axis of the last dimension is the same call.
        require_indices(argmin(Tensor::from_vector(x, shape), static_cast<int>(shape.size()) - 1)
                            .to_vector<std::int64_t>(),
                        result.to_vector<std::int64_t>(), "argmin dim - 1 " + to_string(shape));
    }
}

void test_rules() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    // Each row of 4 checks one rule.
    const std::vector<float> x = {
        3,    1,    1,    2,    // tie: the first minimum
        5,    5,    5,    5,    // all equal: index 0
        2,    nan,  -inf, nan,  // NaN beats -inf; the first NaN
        nan,  -1,   -2,   -3,   // a leading NaN is kept
        -inf, 0,    -inf, 1,    // -inf, first of two
        0.0f, -0.0f, 1,   2,    // +0 and -0 compare equal: the first
        inf,  inf,  inf,  inf,  // all +inf: index 0
    };
    const std::vector<std::int64_t> want = {1, 0, 1, 0, 0, 0, 0};
    require_indices(argmin(Tensor::from_vector(x, {7, 4})).to_vector<std::int64_t>(), want,
                    "tie and NaN rules");
    require_indices(reference_rows(x, 4), want, "reference agrees with the stated rules");

    // The kernel directly, as the dispatcher calls it.
    std::vector<std::int64_t> out(7);
    cpu::argmin_f32_sequential(x.data(), out.data(), 7, 4);
    require_indices(out, want, "argmin_f32_sequential");

    // Without NaNs, argmin(x) is argmax(-x).
    const auto values = test::random_values(12 * 33, 5);
    const Tensor t = Tensor::from_vector(values, {12, 33});
    require_indices(argmin(t).to_vector<std::int64_t>(), argmax(neg(t)).to_vector<std::int64_t>(),
                    "argmin(x) == argmax(-x)");
}

void test_views() {
    const auto values = test::random_values(6 * 10, 9);
    const Tensor base = Tensor::from_vector(values, {6, 10});
    // Every second column: the last axis has stride 2.
    const Tensor strided = base.slice(1, 0, 10, 2);
    require_indices(argmin(strided).to_vector<std::int64_t>(),
                    reference_rows(strided.to_vector<float>(), 5), "strided last axis");
    // Transposed: the last axis is the original first axis.
    const Tensor transposed = base.transpose(0, 1);
    require_indices(argmin(transposed).to_vector<std::int64_t>(),
                    reference_rows(transposed.to_vector<float>(), 6), "transposed");
    // Expanded rows (stride 0) are all the same row.
    const Tensor row = Tensor::from_vector(std::vector<float>{4, 2, 9, 2}, {1, 4});
    require_indices(argmin(row.expand({3, 4})).to_vector<std::int64_t>(), {1, 1, 1},
                    "expanded rows");
}

void test_empty_and_errors() {
    // No rows: an empty result.
    const Tensor none = argmin(Tensor::empty({0, 3}));
    test::require(none.shape() == (Shape{0}) && none.numel() == 0, "no rows");
    test::require(argmin(Tensor::empty({2, 0, 5})).shape() == (Shape{2, 0}), "no rows, rank 3");

    using E = std::invalid_argument;
    const Tensor x = Tensor::zeros({2, 3});
    test::require_throws<E>([&] { argmin(Tensor()); }, "undefined");
    test::require_throws<E>([&] { argmin(Tensor::zeros({})); }, "scalar");
    test::require_throws<E>([&] { argmin(x, 0); }, "non-last axis");
    test::require_throws<E>([&] { argmin(x, -2); }, "non-last negative axis");
    test::require_throws<E>([&] { argmin(x, 2); }, "axis out of range");
    test::require_throws<E>([&] { argmin(Tensor::empty({2, 0})); }, "empty axis");
}

} // namespace

int main() {
    return test::run("argmin_test", [] {
        test_shapes();
        test_rules();
        test_views();
        test_empty_and_errors();
    });
}
