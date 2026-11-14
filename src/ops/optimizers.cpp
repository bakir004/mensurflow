#include "mensurflow/ops/optimizers.hpp"

#include "core/dispatcher.hpp"

#include <cmath>
#include <stdexcept>

namespace mensurflow {
namespace {
void check(Tensor& p, const Tensor& g, const char* name) {
    if (!p.defined() || !g.defined() || p.dtype() != DType::F32 || g.dtype() != DType::F32 ||
        p.shape() != g.shape() || p.device() != g.device() || !p.is_contiguous()) {
        throw std::invalid_argument(std::string(name) +
                                    ": parameter and gradient must be matching contiguous F32 tensors");
    }
}
void check_state(const Tensor& p, const Tensor& state, const char* name) {
    if (!state.defined() || state.dtype() != DType::F32 || state.shape() != p.shape() ||
        state.device() != p.device() || !state.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + ": state must match parameter");
    }
}
}

void sgd(Tensor& p, const Tensor& g, float lr, float momentum, float decay, Tensor* buffer,
         Execution execution) {
    check(p, g, "sgd");
    if (!(std::isfinite(lr) && lr >= 0.0f) || !(std::isfinite(momentum) && momentum >= 0.0f &&
        momentum < 1.0f) || !(std::isfinite(decay) && decay >= 0.0f))
        throw std::invalid_argument("sgd: invalid hyperparameter");
    if (momentum > 0.0f) {
        if (buffer == nullptr) throw std::invalid_argument("sgd: momentum requires a buffer");
        check_state(p, *buffer, "sgd");
    } else if (buffer != nullptr) check_state(p, *buffer, "sgd");
    detail::OpArgs args{{p, g}, {p}};
    if (buffer != nullptr) args.inputs.push_back(*buffer);
    args.alpha = lr; args.gamma = momentum; args.weight_decay = decay;
    detail::dispatch(detail::OpId::Sgd, args, execution);
}

void adam(Tensor& p, const Tensor& g, Tensor& m, Tensor& v, float lr, std::int64_t step,
          float beta1, float beta2, float eps, float decay, Execution execution) {
    check(p, g, "adam"); check_state(p, m, "adam"); check_state(p, v, "adam");
    if (step < 1 || !(std::isfinite(lr) && lr >= 0.0f) || !(beta1 >= 0 && beta1 < 1) ||
        !(beta2 >= 0 && beta2 < 1) || !(eps > 0 && std::isfinite(eps)) ||
        !(decay >= 0 && std::isfinite(decay))) throw std::invalid_argument("adam: invalid hyperparameter");
    detail::OpArgs args{{p, g, m, v}, {p}};
    args.alpha = lr; args.gamma = beta1; args.delta = beta2; args.epsilon = eps;
    args.weight_decay = decay; args.step = step;
    detail::dispatch(detail::OpId::Adam, args, execution);
}

void adamw(Tensor& p, const Tensor& g, Tensor& m, Tensor& v, float lr, std::int64_t step,
           float beta1, float beta2, float eps, float decay, Execution execution) {
    check(p, g, "adamw"); check_state(p, m, "adamw"); check_state(p, v, "adamw");
    if (step < 1 || !(std::isfinite(lr) && lr >= 0.0f) || !(beta1 >= 0 && beta1 < 1) ||
        !(beta2 >= 0 && beta2 < 1) || !(eps > 0 && std::isfinite(eps)) ||
        !(decay >= 0 && std::isfinite(decay))) throw std::invalid_argument("adamw: invalid hyperparameter");
    detail::OpArgs args{{p, g, m, v}, {p}};
    args.alpha = lr; args.gamma = beta1; args.delta = beta2; args.epsilon = eps;
    args.weight_decay = decay; args.step = step;
    detail::dispatch(detail::OpId::AdamW, args, execution);
}
} // namespace mensurflow
