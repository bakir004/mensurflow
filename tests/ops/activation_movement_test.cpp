#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace mensurflow;

namespace {

void require_close(const std::vector<float>& actual, const std::vector<float>& expected,
                   const char* name) {
    test::require(actual.size() == expected.size(), std::string(name) + ": size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        test::require(std::abs(actual[i] - expected[i]) < 2e-5f,
                      std::string(name) + ": value " + std::to_string(i));
    }
}

void test_backward_ops() {
    const std::vector<float> x_values{-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    const std::vector<float> upstream{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor x = Tensor::from_vector(x_values, {5});
    const Tensor gradient = Tensor::from_vector(upstream, {5});

    require_close(relu_backward(x, gradient).to_vector<float>(), {0, 0, 0, 4, 5}, "relu backward");

    std::vector<float> expected(5);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-x_values[i]));
        expected[i] = sigmoid * (1.0f - sigmoid) * upstream[i];
    }
    require_close(sigmoid_backward(x, gradient).to_vector<float>(), expected, "sigmoid backward");

    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float t = std::tanh(x_values[i]);
        expected[i] = (1.0f - t * t) * upstream[i];
    }
    require_close(tanh_backward(x, gradient).to_vector<float>(), expected, "tanh backward");

    x.set_requires_grad();
    sum(gelu(x)).backward();
    test::require(x.grad().defined(), "gelu autograd");

    Tensor silu_input = Tensor::from_vector(x_values, {5});
    silu_input.set_requires_grad();
    sum(silu(silu_input)).backward();
    test::require(silu_input.grad().defined(), "silu autograd");
}

void test_gather_and_index_select() {
    const Tensor x = Tensor::from_vector(std::vector<float>{0, 1, 2, 3, 4, 5}, {2, 3});
    const Tensor columns = Tensor::from_vector(std::vector<std::int64_t>{2, 0}, {2});
    const Tensor selected = index_select(x, columns, 1);
    test::require(selected.shape() == Shape{2, 2}, "index_select shape");
    test::require(selected.to_vector<float>() == std::vector<float>{2, 0, 5, 3},
                  "index_select values");

    const Tensor indices =
        Tensor::from_vector(std::vector<std::int64_t>{2, 0, 1, 1, 0, 2}, {2, 3});
    const Tensor gathered = gather(x, indices, 1);
    test::require(gathered.to_vector<float>() == std::vector<float>{2, 0, 1, 4, 3, 5},
                  "gather values");

    const Tensor rows = Tensor::from_vector(std::vector<std::int64_t>{1, 0}, {2});
    const Tensor selected_rows = index_select(x, rows, 0);
    test::require(selected_rows.to_vector<float>() == std::vector<float>{3, 4, 5, 0, 1, 2},
                  "index_select rows");
    const Tensor row_indices =
        Tensor::from_vector(std::vector<std::int64_t>{1, 0, 0, 1, 1, 0}, {2, 3});
    const Tensor gathered_rows = gather(x, row_indices, 0);
    test::require(gathered_rows.to_vector<float>() == std::vector<float>{3, 1, 2, 3, 4, 2},
                  "gather rows");

    const Tensor narrow_indices =
        Tensor::from_vector(std::vector<std::int64_t>{1, 0, 1}, {1, 3});
    const Tensor narrow = gather(x, narrow_indices, 0);
    test::require(narrow.to_vector<float>() == std::vector<float>{3, 1, 5},
                  "gather shape-mismatch fallback");

    test::require_throws<std::invalid_argument>(
        [&] { index_select(x, Tensor::from_vector(std::vector<std::int64_t>{3}, {1}), 1); },
        "index_select bounds");
    test::require_throws<std::invalid_argument>(
        [&] { gather(x, Tensor::from_vector(std::vector<std::int64_t>{0, 1}, {2}), 1); },
        "gather rank");
}

} // namespace

int main() {
    return test::run("activation_movement_test", [] {
        test_backward_ops();
        test_gather_and_index_select();
    });
}
