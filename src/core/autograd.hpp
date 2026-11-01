#pragma once

#include "mensurflow/tensor.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace mensurflow::detail {

struct AutogradNode {
    std::vector<Tensor> parents;
    std::function<void(const Tensor&)> backward;
};

struct AutogradState {
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<AutogradNode> node;
};

struct AutogradAccess {
    static std::shared_ptr<AutogradState>& state(Tensor& tensor) { return tensor.autograd_; }
    static std::shared_ptr<AutogradState> state(const Tensor& tensor) {
        return tensor.autograd_;
    }

    static void record(Tensor& output, std::vector<Tensor> parents,
                       std::function<void(const Tensor&)> backward);
};

namespace autograd {

bool enabled();

class NoGradGuard {
  public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

void propagate(const Tensor& tensor, const Tensor& gradient);
void backward(const Tensor& output);
Tensor reduce_gradient(const Tensor& gradient, const Shape& target_shape);

} // namespace autograd
} // namespace mensurflow::detail
