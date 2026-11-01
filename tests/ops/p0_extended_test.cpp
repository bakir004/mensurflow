#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cmath>
#include <vector>

using namespace mensurflow;

int main() {
    return test::run("p0_extended_test", [] {
        const Tensor x = Tensor::from_vector<float>({1.0f, 4.0f, 9.0f}, {3});
        test::require(test::matches_f32(
                          pow(x, Tensor::full({3}, 0.5)).to_vector<float>(),
                          {1.0f, 2.0f, 3.0f}),
                      "power");
        test::require(test::matches_f32(reciprocal(x).to_vector<float>(),
                                        {1.0f, 0.25f, 1.0f / 9.0f}, 1e-6f),
                      "reciprocal");
        test::require(test::matches_f32(sin(Tensor::from_vector<float>({0.0f, 1.0f}, {2}))
                                            .to_vector<float>(),
                                        {0.0f, std::sin(1.0f)}, 1e-6f),
                      "sin");

        const Tensor matrix =
            Tensor::from_vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
        const Tensor vector = Tensor::from_vector<float>({1.0f, 1.0f, 1.0f}, {3});
        test::require(test::matches_f32(gemv(matrix, vector).to_vector<float>(), {6.0f, 15.0f}),
                      "gemv");
        const Tensor initial = Tensor::from_vector<float>({2.0f, 3.0f}, {2});
        test::require(test::matches_f32(gemv(matrix, vector, 2.0f, &initial, 0.5f)
                                            .to_vector<float>(),
                                        {13.0f, 31.5f}),
                      "gemv alpha beta");
        const Tensor transposed_input = Tensor::from_vector<float>({1.0f, 1.0f}, {2});
        test::require(test::matches_f32(gemv(matrix, transposed_input, 1.0f, nullptr, 0.0f, true)
                                            .to_vector<float>(),
                                        {5.0f, 7.0f, 9.0f}),
                      "gemv transpose");

        const Tensor view = matrix.transpose(0, 1);
        test::require(copy(view).to_vector<float>() == view.to_vector<float>(), "copy");
        test::require(min(x).item<float>() == 1.0f, "min");
        test::require(prod(x).item<float>() == 36.0f, "prod");
        test::require(min(matrix, 1).to_vector<float>() == std::vector<float>({1.0f, 4.0f}),
                      "axis min");
        test::require(prod(matrix, 1).to_vector<float>() == std::vector<float>({6.0f, 120.0f}),
                      "axis prod");
    });
}
