// Validation and dispatch for the losses, forward and backward, and their autograd.
#include "mensurflow/ops/loss.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"
#include "mensurflow/ops/elementwise.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

void require_f32(const Tensor& t, const char* op, const char* what) {
    if (!t.defined()) {
        throw std::invalid_argument(std::string(op) + ": undefined " + what);
    }
    if (t.dtype() != DType::F32) {
        throw std::invalid_argument(std::string(op) + ": " + what + " must be F32, got " +
                                    dtype_name(t.dtype()));
    }
}

void require_same_device(const Tensor& a, const Tensor& b, const char* op) {
    if (a.device() != b.device()) {
        throw std::invalid_argument(std::string(op) + ": tensors are on " + to_string(a.device()) +
                                    " and " + to_string(b.device()) +
                                    "; move one with Tensor::to()");
    }
}

// The shape a loss produces: one value per row or element for None, a scalar otherwise.
Shape loss_shape(Reduction reduction, Shape per_item) {
    return reduction == Reduction::None ? per_item : Shape{};
}

// The upstream gradient a backward op accepts: the loss's own shape (a 1-element tensor is also
// fine for Sum and Mean).
void require_upstream(const Tensor& gradient, Reduction reduction, const Shape& per_item,
                      const char* op) {
    require_f32(gradient, op, "gradient");
    const bool ok =
        reduction == Reduction::None ? gradient.shape() == per_item : gradient.numel() == 1;
    if (!ok) {
        throw std::invalid_argument(
            std::string(op) + ": gradient has shape " + to_string(gradient.shape()) +
            ", expected " +
            (reduction == Reduction::None ? to_string(per_item) : std::string("one element")));
    }
}

// logits [N, C] (C >= 1) and labels [N] (I64, on the same device). Returns the labels, read once.
std::vector<std::int64_t> check_classification(const Tensor& logits, const Tensor& labels,
                                               const char* op) {
    require_f32(logits, op, "logits");
    if (!labels.defined()) {
        throw std::invalid_argument(std::string(op) + ": undefined labels");
    }
    if (labels.dtype() != DType::I64) {
        throw std::invalid_argument(std::string(op) + ": labels must be I64, got " +
                                    dtype_name(labels.dtype()));
    }
    require_same_device(logits, labels, op);
    if (logits.dim() != 2 || labels.dim() != 1 || labels.size(0) != logits.size(0)) {
        throw std::invalid_argument(
            std::string(op) + ": expected logits [N, C] and labels [N], got " +
            to_string(logits.shape()) + " and " + to_string(labels.shape()));
    }
    const std::int64_t classes = logits.size(1);
    if (classes < 1) {
        throw std::invalid_argument(std::string(op) + ": logits need at least one class");
    }
    // Checked here: a kernel cannot throw from inside an OpenMP loop.
    std::vector<std::int64_t> values = labels.to_vector<std::int64_t>();
    for (const std::int64_t label : values) {
        if (label < 0 || label >= classes) {
            throw std::invalid_argument(std::string(op) + ": label " + std::to_string(label) +
                                        " is out of range [0, " + std::to_string(classes) + ")");
        }
    }
    return values;
}

// x and target of the same shape and device, both F32.
void check_pair(const Tensor& x, const Tensor& target, const char* op) {
    require_f32(x, op, "input");
    require_f32(target, op, "target");
    require_same_device(x, target, op);
    if (x.shape() != target.shape()) {
        throw std::invalid_argument(std::string(op) + ": input " + to_string(x.shape()) +
                                    " and target " + to_string(target.shape()) +
                                    " differ in shape");
    }
}

Tensor run_pair(detail::OpId op, const Tensor& x, const Tensor& target, Reduction reduction,
                Execution execution) {
    check_pair(x, target, detail::op_name(op));
    Tensor out = Tensor::empty(loss_shape(reduction, x.shape()), DType::F32, x.device());
    detail::OpArgs args{{x.contiguous(), target.contiguous()}, {out}};
    args.reduction = static_cast<int>(reduction);
    detail::dispatch(op, args, execution);
    return out;
}

Tensor run_pair_backward(detail::OpId op, const Tensor& x, const Tensor& target,
                         const Tensor& gradient, Reduction reduction, Execution execution) {
    const char* name = detail::op_name(op);
    check_pair(x, target, name);
    require_upstream(gradient, reduction, x.shape(), name);
    require_same_device(x, gradient, name);
    Tensor out = Tensor::empty(x.shape(), DType::F32, x.device());
    detail::OpArgs args{{x.contiguous(), target.contiguous(), gradient.contiguous()}, {out}};
    args.reduction = static_cast<int>(reduction);
    detail::dispatch(op, args, execution);
    return out;
}

} // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& labels, Reduction reduction,
                     Execution execution) {
    const std::vector<std::int64_t> values = check_classification(logits, labels, "cross_entropy");
    Tensor out =
        Tensor::empty(loss_shape(reduction, {logits.size(0)}), DType::F32, logits.device());
    detail::OpArgs args{{logits.contiguous(), labels.contiguous()}, {out}};
    args.reduction = static_cast<int>(reduction);
    detail::dispatch(detail::OpId::CrossEntropy, args, execution);
    if (logits.requires_grad() && detail::autograd::enabled()) {
        // An independent copy, so changing the caller's labels after forward cannot change the
        // gradient.
        const Tensor saved = Tensor::from_vector(values, labels.shape(), labels.device());
        detail::AutogradAccess::record(
            out, {logits}, [logits, saved, reduction, execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                detail::autograd::propagate(
                    logits, cross_entropy_backward(logits, saved, gradient, reduction, execution));
            });
    }
    return out;
}

Tensor cross_entropy_backward(const Tensor& logits, const Tensor& labels, const Tensor& gradient,
                              Reduction reduction, Execution execution) {
    check_classification(logits, labels, "cross_entropy_backward");
    require_upstream(gradient, reduction, {logits.size(0)}, "cross_entropy_backward");
    require_same_device(logits, gradient, "cross_entropy_backward");
    Tensor out = Tensor::empty(logits.shape(), DType::F32, logits.device());
    detail::OpArgs args{{logits.contiguous(), labels.contiguous(), gradient.contiguous()}, {out}};
    args.reduction = static_cast<int>(reduction);
    detail::dispatch(detail::OpId::CrossEntropyBackward, args, execution);
    return out;
}

Tensor mse_loss(const Tensor& x, const Tensor& target, Reduction reduction, Execution execution) {
    Tensor out = run_pair(detail::OpId::MseLoss, x, target, reduction, execution);
    if ((x.requires_grad() || target.requires_grad()) && detail::autograd::enabled()) {
        detail::AutogradAccess::record(
            out, {x, target}, [x, target, reduction, execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const Tensor dx = mse_loss_backward(x, target, gradient, reduction, execution);
                if (x.requires_grad()) {
                    detail::autograd::propagate(x, dx);
                }
                if (target.requires_grad()) {
                    detail::autograd::propagate(target, neg(dx, execution));
                }
            });
    }
    return out;
}

Tensor mse_loss_backward(const Tensor& x, const Tensor& target, const Tensor& gradient,
                         Reduction reduction, Execution execution) {
    return run_pair_backward(detail::OpId::MseLossBackward, x, target, gradient, reduction,
                             execution);
}

Tensor binary_cross_entropy_with_logits(const Tensor& x, const Tensor& target, Reduction reduction,
                                        Execution execution) {
    Tensor out = run_pair(detail::OpId::BceWithLogits, x, target, reduction, execution);
    if ((x.requires_grad() || target.requires_grad()) && detail::autograd::enabled()) {
        detail::AutogradAccess::record(
            out, {x, target}, [x, target, reduction, execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                if (x.requires_grad()) {
                    detail::autograd::propagate(x, binary_cross_entropy_with_logits_backward(
                                                       x, target, gradient, reduction, execution));
                }
                if (target.requires_grad()) {
                    // d loss / d target = -x * upstream (divided by the count for Mean).
                    Tensor dt = mul(neg(x, execution), gradient, execution);
                    if (reduction == Reduction::Mean && x.numel() > 0) {
                        const float count = static_cast<float>(x.numel());
                        dt = mul(dt, Tensor::full({}, 1.0 / count, DType::F32, x.device()),
                                 execution);
                    }
                    detail::autograd::propagate(target, dt);
                }
            });
    }
    return out;
}

Tensor binary_cross_entropy_with_logits_backward(const Tensor& x, const Tensor& target,
                                                 const Tensor& gradient, Reduction reduction,
                                                 Execution execution) {
    return run_pair_backward(detail::OpId::BceWithLogitsBackward, x, target, gradient, reduction,
                             execution);
}

} // namespace mensurflow
