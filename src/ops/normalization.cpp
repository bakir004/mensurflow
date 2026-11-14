// Validation, dispatch and autograd for layer_norm, rms_norm and batch_norm (see
// ops/normalization.hpp).
#include "mensurflow/ops/normalization.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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
        fail(op, std::string(what) + " must be on the CPU (normalization is CPU only so far)");
    }
}

void require_epsilon(float epsilon, const char* op) {
    if (!(std::isfinite(epsilon) && epsilon > 0.0f)) {
        fail(op, "epsilon must be finite and > 0");
    }
}

// x, and the number of elements in one normalized line.
std::int64_t check_lines(const Tensor& x, const Shape& normalized, const char* op) {
    require_f32_cpu(x, op, "input");
    if (normalized.empty() || normalized.size() > x.shape().size()) {
        fail(op, "normalized shape " + to_string(normalized) + " needs 1 to " +
                     std::to_string(x.dim()) + " dimensions for input " + to_string(x.shape()));
    }
    std::int64_t width = 1;
    const std::size_t offset = x.shape().size() - normalized.size();
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        if (normalized[i] != x.shape()[offset + i]) {
            fail(op, "normalized shape " + to_string(normalized) +
                         " does not match the trailing dimensions of " + to_string(x.shape()));
        }
        width *= normalized[i];
    }
    if (width == 0) {
        fail(op, "cannot normalize over an empty dimension");
    }
    return width;
}

void check_parameter(const Tensor* t, const Shape& shape, const char* op, const char* what) {
    if (t == nullptr) {
        return;
    }
    require_f32_cpu(*t, op, what);
    if (t->shape() != shape) {
        fail(op, std::string(what) + " shape " + to_string(t->shape()) + " must be " +
                     to_string(shape));
    }
}

// The caller's parameter packed, or a tensor of the default (1 for scales, 0 for biases).
Tensor parameter_or_default(const Tensor* t, const Shape& shape, float fill, Device device) {
    return t != nullptr ? t->contiguous() : Tensor::full(shape, fill, DType::F32, device);
}

void check_gradient(const Tensor& x, const Tensor& gradient, const char* op) {
    require_f32_cpu(gradient, op, "gradient");
    if (gradient.shape() != x.shape()) {
        fail(op, "gradient " + to_string(gradient.shape()) + " and input " + to_string(x.shape()) +
                     " differ in shape");
    }
}

bool any_requires_grad(const Tensor& x, const Tensor* weight, const Tensor* bias) {
    return x.requires_grad() || (weight != nullptr && weight->requires_grad()) ||
           (bias != nullptr && bias->requires_grad());
}

std::vector<Tensor> autograd_parents(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    std::vector<Tensor> parents{x};
    for (const Tensor* t : {&weight, &bias}) {
        if (t->defined()) {
            parents.push_back(*t);
        }
    }
    return parents;
}

Tensor copy_of(const Tensor& t) {
    Tensor copy = Tensor::empty(t.shape(), t.dtype(), t.device());
    copy.copy_from(t);
    return copy;
}

} // namespace

// ------------------------------------------------------------------------------ layer norm

Tensor layer_norm(const Tensor& x, const Shape& normalized_shape, const Tensor* weight,
                  const Tensor* bias, float epsilon, Execution execution) {
    const char* op = "layer_norm";
    check_lines(x, normalized_shape, op);
    check_parameter(weight, normalized_shape, op, "weight");
    check_parameter(bias, normalized_shape, op, "bias");
    require_epsilon(epsilon, op);
    Tensor out = Tensor::empty(x.shape(), DType::F32, x.device());
    if (x.numel() > 0) {
        detail::OpArgs args{{x.contiguous(),
                             parameter_or_default(weight, normalized_shape, 1.0f, x.device()),
                             parameter_or_default(bias, normalized_shape, 0.0f, x.device())},
                            {out}};
        args.epsilon = epsilon;
        detail::dispatch(detail::OpId::LayerNorm, args, execution);
    }
    if (detail::autograd::enabled() && any_requires_grad(x, weight, bias)) {
        const Tensor w = weight != nullptr ? *weight : Tensor();
        const Tensor b = bias != nullptr ? *bias : Tensor();
        const Shape shape = normalized_shape;
        detail::AutogradAccess::record(
            out, autograd_parents(x, w, b),
            [x, w, b, shape, epsilon, execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const NormalizationBackward g = layer_norm_backward(
                    x, gradient, shape, w.defined() ? &w : nullptr, epsilon, execution);
                detail::autograd::propagate(x, g.input);
                if (w.defined()) {
                    detail::autograd::propagate(w, g.weight);
                }
                if (b.defined()) {
                    detail::autograd::propagate(b, g.bias);
                }
            });
    }
    return out;
}

NormalizationBackward layer_norm_backward(const Tensor& x, const Tensor& gradient,
                                          const Shape& normalized_shape, const Tensor* weight,
                                          float epsilon, Execution execution) {
    const char* op = "layer_norm_backward";
    check_lines(x, normalized_shape, op);
    check_gradient(x, gradient, op);
    check_parameter(weight, normalized_shape, op, "weight");
    require_epsilon(epsilon, op);
    NormalizationBackward result{Tensor::empty(x.shape(), DType::F32, x.device()),
                                 Tensor::zeros(normalized_shape, DType::F32, x.device()),
                                 Tensor::zeros(normalized_shape, DType::F32, x.device())};
    if (x.numel() > 0) {
        detail::OpArgs args{{x.contiguous(), gradient.contiguous(),
                             parameter_or_default(weight, normalized_shape, 1.0f, x.device())},
                            {result.input, result.weight, result.bias}};
        args.epsilon = epsilon;
        detail::dispatch(detail::OpId::LayerNormBackward, args, execution);
    }
    return result;
}

// -------------------------------------------------------------------------------- RMS norm

Tensor rms_norm(const Tensor& x, const Shape& normalized_shape, const Tensor* weight, float epsilon,
                Execution execution) {
    const char* op = "rms_norm";
    check_lines(x, normalized_shape, op);
    check_parameter(weight, normalized_shape, op, "weight");
    require_epsilon(epsilon, op);
    Tensor out = Tensor::empty(x.shape(), DType::F32, x.device());
    if (x.numel() > 0) {
        detail::OpArgs args{
            {x.contiguous(), parameter_or_default(weight, normalized_shape, 1.0f, x.device())},
            {out}};
        args.epsilon = epsilon;
        detail::dispatch(detail::OpId::RmsNorm, args, execution);
    }
    if (detail::autograd::enabled() && any_requires_grad(x, weight, nullptr)) {
        const Tensor w = weight != nullptr ? *weight : Tensor();
        const Shape shape = normalized_shape;
        detail::AutogradAccess::record(out, autograd_parents(x, w, Tensor()),
                                       [x, w, shape, epsilon, execution](const Tensor& gradient) {
                                           detail::autograd::NoGradGuard guard;
                                           const NormalizationBackward g = rms_norm_backward(
                                               x, gradient, shape, w.defined() ? &w : nullptr,
                                               epsilon, execution);
                                           detail::autograd::propagate(x, g.input);
                                           if (w.defined()) {
                                               detail::autograd::propagate(w, g.weight);
                                           }
                                       });
    }
    return out;
}

NormalizationBackward rms_norm_backward(const Tensor& x, const Tensor& gradient,
                                        const Shape& normalized_shape, const Tensor* weight,
                                        float epsilon, Execution execution) {
    const char* op = "rms_norm_backward";
    check_lines(x, normalized_shape, op);
    check_gradient(x, gradient, op);
    check_parameter(weight, normalized_shape, op, "weight");
    require_epsilon(epsilon, op);
    NormalizationBackward result{Tensor::empty(x.shape(), DType::F32, x.device()),
                                 Tensor::zeros(normalized_shape, DType::F32, x.device()), Tensor()};
    if (x.numel() > 0) {
        detail::OpArgs args{{x.contiguous(), gradient.contiguous(),
                             parameter_or_default(weight, normalized_shape, 1.0f, x.device())},
                            {result.input, result.weight}};
        args.epsilon = epsilon;
        detail::dispatch(detail::OpId::RmsNormBackward, args, execution);
    }
    return result;
}

// ------------------------------------------------------------------------------ batch norm

namespace {

// x = [N, C, ...]; the running statistics are [C] contiguous F32. Returns C.
std::int64_t check_batch(const Tensor& x, const Tensor& running_mean,
                         const Tensor& running_variance, const char* op) {
    require_f32_cpu(x, op, "input");
    if (x.dim() < 2) {
        fail(op, "input must be [N, C, ...], got " + to_string(x.shape()));
    }
    const std::int64_t channels = x.size(1);
    if (channels < 1) {
        fail(op, "input has no channels: " + to_string(x.shape()));
    }
    const Shape stats_shape{channels};
    for (const Tensor* t : {&running_mean, &running_variance}) {
        require_f32_cpu(*t, op, t == &running_mean ? "running mean" : "running variance");
        if (t->shape() != stats_shape) {
            fail(op, std::string(t == &running_mean ? "running mean" : "running variance") +
                         " shape " + to_string(t->shape()) + " must be " + to_string(stats_shape));
        }
        if (!t->is_contiguous()) {
            fail(op, "running statistics must be contiguous");
        }
    }
    return channels;
}

} // namespace

Tensor batch_norm(const Tensor& x, Tensor& running_mean, Tensor& running_variance,
                  const Tensor* weight, const Tensor* bias, bool training, float momentum,
                  float epsilon, Execution execution) {
    const char* op = "batch_norm";
    const std::int64_t channels = check_batch(x, running_mean, running_variance, op);
    const Shape channel_shape{channels};
    check_parameter(weight, channel_shape, op, "weight");
    check_parameter(bias, channel_shape, op, "bias");
    require_epsilon(epsilon, op);
    if (training && !(std::isfinite(momentum) && momentum >= 0.0f && momentum <= 1.0f)) {
        fail(op, "momentum must be in [0, 1]");
    }
    if (training && x.numel() / channels < 2) {
        fail(op,
             "training needs more than one value per channel, got input " + to_string(x.shape()));
    }
    // Eval-mode backward needs the statistics as they are now, and the caller may keep updating
    // them, so the recorded node keeps its own copy.
    const bool record_graph = detail::autograd::enabled() && any_requires_grad(x, weight, bias);
    const Tensor saved_mean = record_graph && !training ? copy_of(running_mean) : running_mean;
    const Tensor saved_variance =
        record_graph && !training ? copy_of(running_variance) : running_variance;

    Tensor out = Tensor::empty(x.shape(), DType::F32, x.device());
    if (x.numel() > 0) {
        detail::OpArgs args{{x.contiguous(),
                             parameter_or_default(weight, channel_shape, 1.0f, x.device()),
                             parameter_or_default(bias, channel_shape, 0.0f, x.device())},
                            {out, running_mean, running_variance}};
        args.training = training;
        args.momentum = momentum;
        args.epsilon = epsilon;
        detail::dispatch(detail::OpId::BatchNorm, args, execution);
    }
    if (record_graph) {
        const Tensor w = weight != nullptr ? *weight : Tensor();
        const Tensor b = bias != nullptr ? *bias : Tensor();
        detail::AutogradAccess::record(
            out, autograd_parents(x, w, b),
            [x, w, b, saved_mean, saved_variance, training, epsilon,
             execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const NormalizationBackward g =
                    batch_norm_backward(x, gradient, saved_mean, saved_variance,
                                        w.defined() ? &w : nullptr, training, epsilon, execution);
                detail::autograd::propagate(x, g.input);
                if (w.defined()) {
                    detail::autograd::propagate(w, g.weight);
                }
                if (b.defined()) {
                    detail::autograd::propagate(b, g.bias);
                }
            });
    }
    return out;
}

NormalizationBackward batch_norm_backward(const Tensor& x, const Tensor& gradient,
                                          const Tensor& running_mean,
                                          const Tensor& running_variance, const Tensor* weight,
                                          bool training, float epsilon, Execution execution) {
    const char* op = "batch_norm_backward";
    const std::int64_t channels = check_batch(x, running_mean, running_variance, op);
    const Shape channel_shape{channels};
    check_gradient(x, gradient, op);
    check_parameter(weight, channel_shape, op, "weight");
    require_epsilon(epsilon, op);
    if (training && x.numel() / channels < 2) {
        fail(op,
             "training needs more than one value per channel, got input " + to_string(x.shape()));
    }
    NormalizationBackward result{Tensor::empty(x.shape(), DType::F32, x.device()),
                                 Tensor::zeros(channel_shape, DType::F32, x.device()),
                                 Tensor::zeros(channel_shape, DType::F32, x.device())};
    if (x.numel() > 0) {
        detail::OpArgs args{{x.contiguous(), gradient.contiguous(),
                             parameter_or_default(weight, channel_shape, 1.0f, x.device()),
                             running_mean, running_variance},
                            {result.input, result.weight, result.bias}};
        args.training = training;
        args.epsilon = epsilon;
        detail::dispatch(detail::OpId::BatchNormBackward, args, execution);
    }
    return result;
}

} // namespace mensurflow
