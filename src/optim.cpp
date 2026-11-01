#include "mensurflow/optim.hpp"

#include <stdexcept>
#include <utility>

namespace mensurflow {

SGD::SGD(std::vector<Tensor> parameters, float learning_rate, float momentum, float weight_decay)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), momentum_(momentum),
      weight_decay_(weight_decay) {
    if (learning_rate <= 0.0f || momentum < 0.0f || momentum >= 1.0f || weight_decay < 0.0f) {
        throw std::invalid_argument("SGD: invalid hyperparameters");
    }
    for (const Tensor& parameter : parameters_) {
        if (!parameter.defined() || parameter.dtype() != DType::F32 ||
            parameter.device().type != DeviceType::CPU || !parameter.requires_grad()) {
            throw std::invalid_argument("SGD: parameters must be CPU f32 tensors requiring grad");
        }
        velocity_.push_back(Tensor::zeros(parameter.shape(), DType::F32, parameter.device()));
    }
}

void SGD::zero_grad() {
    for (Tensor& parameter : parameters_) {
        parameter.zero_grad();
    }
}

void SGD::step() {
    for (std::size_t p = 0; p < parameters_.size(); ++p) {
        Tensor& parameter = parameters_[p];
        if (!parameter.has_grad()) {
            throw std::logic_error("SGD::step: parameter has no gradient");
        }
        const auto values = parameter.to_vector<float>();
        const auto gradients = parameter.grad().to_vector<float>();
        auto velocity = velocity_[p].to_vector<float>();
        std::vector<float> updated(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            const float gradient = gradients[i] + weight_decay_ * values[i];
            velocity[i] = momentum_ * velocity[i] + gradient;
            updated[i] = values[i] - learning_rate_ * velocity[i];
        }
        velocity_[p].copy_from(Tensor::from_vector(velocity, velocity_[p].shape()));
        parameter.copy_from(Tensor::from_vector(updated, parameter.shape()));
    }
}

} // namespace mensurflow
