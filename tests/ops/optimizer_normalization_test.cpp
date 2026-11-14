#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cmath>
#include <vector>

using namespace mensurflow;

namespace {
void near(float actual, float expected, float tolerance = 2e-4f) {
    test::require(std::abs(actual - expected) <= tolerance, "values differ");
}

void optimizer_tests() {
    Tensor p = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3});
    const Tensor g = Tensor::ones({3});
    sgd(p, g, 0.1f);
    test::require(p.to_vector<float>()[0] == 0.9f, "sgd update");
    Tensor m = Tensor::zeros({3}), v = Tensor::zeros({3});
    adam(p, g, m, v, 0.1f, 1);
    near(p.to_vector<float>()[0], 0.8f, 1e-5f);
    adamw(p, g, m, v, 0.1f, 2);
}

void normalization_tests() {
    const Tensor x = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor w = Tensor::ones({3});
    const Tensor y = layer_norm(x, {3}, &w);
    auto values = y.to_vector<float>();
    near(values[0] + values[1] + values[2], 0.0f);
    auto backward = layer_norm_backward(x, Tensor::ones({2, 3}), {3}, &w);
    near(backward.weight.to_vector<float>()[0], -2.44947f, 1e-3f);
    const Tensor rms = rms_norm(x, {3});
    test::require(rms.numel() == 6, "rms output shape");
    auto rb = rms_norm_backward(x, Tensor::ones({2, 3}), {3});
    std::cerr << "rb\n";
    test::require(rb.input.shape() == x.shape(), "rms backward shape");

    Tensor running_mean = Tensor::zeros({3}), running_var = Tensor::ones({3});
    Tensor bn = batch_norm(x, running_mean, running_var, &w, nullptr, true);
    auto b = bn.to_vector<float>();
    near(b[0] + b[1] + b[2], 0.0f);
    auto bb = batch_norm_backward(x, Tensor::ones({2, 3}), running_mean, running_var, &w, true);
    test::require(bb.input.shape() == x.shape(), "batch backward shape");
}
}

int main() {
    optimizer_tests();
    normalization_tests();
    return 0;
}
