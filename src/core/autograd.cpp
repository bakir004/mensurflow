#include "core/autograd.hpp"

#include "mensurflow/ops/elementwise.hpp"

#include <stdexcept>

namespace mensurflow::detail {

namespace {
thread_local int no_grad_depth = 0;
}

void AutogradAccess::record(Tensor& output, std::vector<Tensor> parents,
                            std::function<void(const Tensor&)> backward_fn) {
    if (!autograd::enabled()) {
        return;
    }
    bool needs_grad = false;
    for (const Tensor& parent : parents) {
        needs_grad = needs_grad || parent.requires_grad();
    }
    if (!needs_grad) {
        return;
    }
    if (output.dtype() != DType::F32 || output.device().type != DeviceType::CPU) {
        throw std::runtime_error("autograd: only CPU f32 tensors are supported");
    }
    auto& state_ref = state(output);
    if (state_ref == nullptr) {
        state_ref = std::make_shared<AutogradState>();
    }
    state_ref->requires_grad = true;
    state_ref->node = std::make_shared<AutogradNode>(
        AutogradNode{std::move(parents), std::move(backward_fn)});
}

namespace autograd {

bool enabled() { return no_grad_depth == 0; }

NoGradGuard::NoGradGuard() { ++no_grad_depth; }

NoGradGuard::~NoGradGuard() { --no_grad_depth; }

void propagate(const Tensor& tensor, const Tensor& gradient) {
    if (!tensor.requires_grad()) {
        return;
    }
    if (tensor.shape() != gradient.shape() || tensor.device() != gradient.device() ||
        gradient.dtype() != DType::F32) {
        throw std::invalid_argument("autograd: gradient shape, device, or dtype mismatch");
    }
    auto& state = AutogradAccess::state(tensor);
    if (state->grad == nullptr) {
        state->grad = std::make_shared<Tensor>(gradient);
    } else {
        NoGradGuard guard;
        state->grad = std::make_shared<Tensor>(add(*state->grad, gradient));
    }
    if (state->node != nullptr) {
        const auto node = state->node;
        state->node.reset();
        node->backward(gradient);
    }
}

void backward(const Tensor& output) {
    if (!output.requires_grad()) {
        throw std::logic_error("backward: tensor does not require gradients");
    }
    if (output.numel() != 1) {
        throw std::invalid_argument("backward: implicit gradient requires a scalar tensor");
    }
    const Tensor seed = Tensor::ones({}, DType::F32, output.device());
    propagate(output, seed);
}

} // namespace autograd
} // namespace mensurflow::detail
