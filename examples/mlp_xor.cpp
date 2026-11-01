#include "mensurflow/mensurflow.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    using namespace mensurflow;
    Tensor inputs = Tensor::from_vector<float>({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
    Tensor w1 = Tensor::from_vector<float>({0.4f, -0.2f, -0.3f, 0.5f}, {2, 2});
    Tensor b1 = Tensor::from_vector<float>({0.1f, 0.1f}, {2});
    Tensor w2 = Tensor::from_vector<float>({0.6f, -0.4f}, {2, 1});
    Tensor b2 = Tensor::from_vector<float>({0.1f}, {1});
    for (Tensor* parameter : {&w1, &b1, &w2, &b2}) {
        parameter->set_requires_grad();
    }
    SGD optimizer({w1, b1, w2, b2}, 0.2f, 0.9f, 0.0f);

    for (int epoch = 0; epoch < 2000; ++epoch) {
        optimizer.zero_grad();
        Tensor hidden = relu(add(matmul(inputs, w1), b1.view({1, 2}).expand({4, 2})));
        Tensor logits = add(matmul(hidden, w2), b2.view({1, 1}).expand({4, 1}));
        Tensor loss = mse_loss(logits, Tensor::from_vector<float>({0, 1, 1, 1}, {4, 1}));
        loss.backward();
        optimizer.step();
        if (epoch % 500 == 0) {
            std::cout << "epoch " << epoch << " loss " << loss.item<float>() << '\n';
        }
    }
    std::cout << "final loss " << mse_loss(
        add(matmul(relu(add(matmul(inputs, w1), b1.view({1, 2}).expand({4, 2}))), w2),
            b2.view({1, 1}).expand({4, 1})),
        Tensor::from_vector<float>({0, 1, 1, 1}, {4, 1})).item<float>()
              << '\n';
}
