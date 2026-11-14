// Validation and dispatch for the in-place optimizer updates (see ops/optimizers.hpp).
#include "mensurflow/ops/optimizers.hpp"

#include "core/dispatcher.hpp"

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace mensurflow {

namespace {

[[noreturn]] void fail(const char* op, const std::string& message) {
    throw std::invalid_argument(std::string(op) + ": " + message);
}

void require_f32_cpu(const Tensor& t, const char* op, const char* what) {
    if (!t.defined()) {
        fail(op, std::string("undefined ") + what);
    }
    if (t.dtype() != DType::F32) {
        fail(op, std::string(what) + " must be F32, got " + dtype_name(t.dtype()));
    }
    if (t.device().type != DeviceType::CPU) {
        fail(op, std::string(what) + " must be on the CPU (optimizers are CPU only so far)");
    }
}

// The tensors that are updated: contiguous, parameter-shaped, and not sharing storage.
void check_updated(const Tensor& parameter, const Tensor& state, const char* op, const char* what) {
    require_f32_cpu(state, op, what);
    if (state.shape() != parameter.shape()) {
        fail(op, std::string(what) + " shape " + to_string(state.shape()) +
                     " does not match the parameter's " + to_string(parameter.shape()));
    }
    if (!state.is_contiguous()) {
        fail(op, std::string(what) + " must be contiguous");
    }
    if (state.shares_storage(parameter)) {
        fail(op, std::string(what) + " must not share storage with the parameter");
    }
}

Tensor check_parameter_and_gradient(const Tensor& parameter, const Tensor& gradient,
                                    const char* op) {
    require_f32_cpu(parameter, op, "parameter");
    require_f32_cpu(gradient, op, "gradient");
    if (gradient.shape() != parameter.shape()) {
        fail(op, "gradient shape " + to_string(gradient.shape()) +
                     " does not match the parameter's " + to_string(parameter.shape()));
    }
    if (!parameter.is_contiguous()) {
        fail(op, "parameter must be contiguous");
    }
    return gradient.contiguous();
}

bool finite_at_least_zero(float value) { return std::isfinite(value) && value >= 0.0f; }

void check_adam_hyperparameters(const char* op, float lr, std::int64_t step, float beta1,
                                float beta2, float eps, float weight_decay) {
    if (step < 1) {
        fail(op, "step counts from 1, got " + std::to_string(step));
    }
    if (!finite_at_least_zero(lr)) {
        fail(op, "learning rate must be finite and >= 0");
    }
    if (!(beta1 >= 0.0f && beta1 < 1.0f) || !(beta2 >= 0.0f && beta2 < 1.0f)) {
        fail(op, "beta1 and beta2 must be in [0, 1)");
    }
    if (!(std::isfinite(eps) && eps > 0.0f)) {
        fail(op, "epsilon must be finite and > 0");
    }
    if (!finite_at_least_zero(weight_decay)) {
        fail(op, "weight decay must be finite and >= 0");
    }
}

void run_adam(detail::OpId op, Tensor& parameter, const Tensor& gradient, Tensor& first_moment,
              Tensor& second_moment, float lr, std::int64_t step, float beta1, float beta2,
              float eps, float weight_decay, Execution execution) {
    const char* name = detail::op_name(op);
    const Tensor grad = check_parameter_and_gradient(parameter, gradient, name);
    check_updated(parameter, first_moment, name, "first moment");
    check_updated(parameter, second_moment, name, "second moment");
    if (first_moment.shares_storage(second_moment)) {
        fail(name, "the two moments must not share storage");
    }
    check_adam_hyperparameters(name, lr, step, beta1, beta2, eps, weight_decay);
    detail::OpArgs args{{grad}, {parameter, first_moment, second_moment}};
    args.alpha = lr;
    args.beta1 = beta1;
    args.beta2 = beta2;
    args.epsilon = eps;
    args.weight_decay = weight_decay;
    args.step = step;
    detail::dispatch(op, args, execution);
}

} // namespace

void sgd(Tensor& parameter, const Tensor& gradient, float learning_rate, float momentum,
         float weight_decay, Tensor* momentum_buffer, Execution execution) {
    const Tensor grad = check_parameter_and_gradient(parameter, gradient, "sgd");
    if (!finite_at_least_zero(learning_rate)) {
        fail("sgd", "learning rate must be finite and >= 0");
    }
    if (!(std::isfinite(momentum) && momentum >= 0.0f && momentum < 1.0f)) {
        fail("sgd", "momentum must be in [0, 1)");
    }
    if (!finite_at_least_zero(weight_decay)) {
        fail("sgd", "weight decay must be finite and >= 0");
    }
    if (momentum > 0.0f && momentum_buffer == nullptr) {
        fail("sgd", "momentum > 0 needs a momentum buffer");
    }
    if (momentum_buffer != nullptr) {
        check_updated(parameter, *momentum_buffer, "sgd", "momentum buffer");
    }
    detail::OpArgs args{{grad}, {parameter}};
    if (momentum > 0.0f) {
        args.outputs.push_back(*momentum_buffer);
    }
    args.alpha = learning_rate;
    args.momentum = momentum;
    args.weight_decay = weight_decay;
    detail::dispatch(detail::OpId::Sgd, args, execution);
}

void adam(Tensor& parameter, const Tensor& gradient, Tensor& first_moment, Tensor& second_moment,
          float learning_rate, std::int64_t step, float beta1, float beta2, float epsilon,
          float weight_decay, Execution execution) {
    run_adam(detail::OpId::Adam, parameter, gradient, first_moment, second_moment, learning_rate,
             step, beta1, beta2, epsilon, weight_decay, execution);
}

void adamw(Tensor& parameter, const Tensor& gradient, Tensor& first_moment, Tensor& second_moment,
           float learning_rate, std::int64_t step, float beta1, float beta2, float epsilon,
           float weight_decay, Execution execution) {
    run_adam(detail::OpId::AdamW, parameter, gradient, first_moment, second_moment, learning_rate,
             step, beta1, beta2, epsilon, weight_decay, execution);
}

} // namespace mensurflow
