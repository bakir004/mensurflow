// Trains a 2-2-1 network on XOR with autograd and plain gradient descent.
#include "mensurflow/mensurflow.hpp"

#include <iostream>
#include <vector>

using namespace mensurflow;

namespace {

Tensor forward(const Tensor& inputs, const Tensor& w1, const Tensor& b1, const Tensor& w2,
               const Tensor& b2) {
    const Tensor hidden = relu(bias_add(matmul(inputs, w1), b1));
    return bias_add(matmul(hidden, w2), b2);
}

Tensor mse(const Tensor& prediction, const Tensor& target) {
    return mean(square(sub(prediction, target)));
}

// parameter -= learning_rate * gradient, done on host copies so the update is not recorded.
void step(Tensor& parameter, float learning_rate) {
    std::vector<float> values = parameter.to_vector<float>();
    const std::vector<float> gradient = parameter.grad().to_vector<float>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] -= learning_rate * gradient[i];
    }
    parameter.copy_from(Tensor::from_vector(values, parameter.shape()));
}

} // namespace

int main() {
    const Tensor inputs = Tensor::from_vector<float>({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
    const Tensor target = Tensor::from_vector<float>({0, 1, 1, 0}, {4, 1});
    Tensor w1 = Tensor::from_vector<float>({0.5f, -0.6f, 0.7f, 0.4f}, {2, 2});
    Tensor b1 = Tensor::from_vector<float>({-0.2f, 0.1f}, {2});
    Tensor w2 = Tensor::from_vector<float>({0.6f, 0.5f}, {2, 1});
    Tensor b2 = Tensor::from_vector<float>({0.1f}, {1});
    std::vector<Tensor*> parameters{&w1, &b1, &w2, &b2};
    for (Tensor* parameter : parameters) {
        parameter->set_requires_grad();
    }

    for (int epoch = 0; epoch <= 2000; ++epoch) {
        for (Tensor* parameter : parameters) {
            parameter->zero_grad();
        }
        const Tensor loss = mse(forward(inputs, w1, b1, w2, b2), target);
        loss.backward();
        for (Tensor* parameter : parameters) {
            step(*parameter, 0.1f);
        }
        if (epoch % 500 == 0) {
            std::cout << "epoch " << epoch << " loss " << loss.item<float>() << '\n';
        }
    }

    const auto prediction = forward(inputs, w1, b1, w2, b2).to_vector<float>();
    std::cout << "predictions for 00 01 10 11:";
    for (float value : prediction) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}
