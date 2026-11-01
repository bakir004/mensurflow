#pragma once

#include "mensurflow/tensor.hpp"

#include <vector>

namespace mensurflow {

class SGD {
  public:
    SGD(std::vector<Tensor> parameters, float learning_rate, float momentum = 0.0f,
        float weight_decay = 0.0f);

    void zero_grad();
    void step();

  private:
    std::vector<Tensor> parameters_;
    std::vector<Tensor> velocity_;
    float learning_rate_;
    float momentum_;
    float weight_decay_;
};

} // namespace mensurflow
