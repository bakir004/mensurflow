// Validation and dispatch for softmax and log_softmax, forward and backward, and their autograd.
#include "mensurflow/ops/softmax.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"

#include <stdexcept>
#include <string>

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

// The axis in [0, rank). A 0-d tensor is one line of one element, reachable as axis 0 or -1.
int checked_axis(int axis, int rank, const char* op) {
    const int extent = rank == 0 ? 1 : rank;
    const int out = axis < 0 ? axis + extent : axis;
    if (out < 0 || out >= extent) {
        throw std::invalid_argument(std::string(op) + ": axis " + std::to_string(axis) +
                                    " is out of range for a " + std::to_string(rank) + "-d tensor");
    }
    return out;
}

Tensor forward(detail::OpId op, const Tensor& x, int axis, Execution execution) {
    const char* name = detail::op_name(op);
    require_f32(x, name, "input");
    const int dim = checked_axis(axis, x.dim(), name);
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    detail::OpArgs args{{x.contiguous()}, {out}};
    args.axis = dim;
    detail::dispatch(op, args, execution);
    return out;
}

Tensor backward(detail::OpId op, const Tensor& output, const Tensor& gradient, int axis,
                Execution execution) {
    const char* name = detail::op_name(op);
    require_f32(output, name, "output");
    require_f32(gradient, name, "gradient");
    if (output.shape() != gradient.shape()) {
        throw std::invalid_argument(std::string(name) + ": output " + to_string(output.shape()) +
                                    " and gradient " + to_string(gradient.shape()) +
                                    " differ in shape");
    }
    if (output.device() != gradient.device()) {
        throw std::invalid_argument(std::string(name) + ": tensors must be on the same device");
    }
    const int dim = checked_axis(axis, output.dim(), name);
    Tensor out = Tensor::empty(output.shape(), output.dtype(), output.device());
    detail::OpArgs args{{output.contiguous(), gradient.contiguous()}, {out}};
    args.axis = dim;
    detail::dispatch(op, args, execution);
    return out;
}

} // namespace

Tensor softmax(const Tensor& x, int axis, Execution execution) {
    Tensor out = forward(detail::OpId::Softmax, x, axis, execution);
    if (x.requires_grad() && detail::autograd::enabled()) {
        // Taken before record(): it shares out's storage but not its autograd state, so the node
        // does not keep itself alive through its own output.
        const Tensor saved = out;
        detail::AutogradAccess::record(out, {x}, [x, saved, axis, execution](const Tensor& g) {
            detail::autograd::NoGradGuard guard;
            detail::autograd::propagate(x, softmax_backward(saved, g, axis, execution));
        });
    }
    return out;
}

Tensor log_softmax(const Tensor& x, int axis, Execution execution) {
    Tensor out = forward(detail::OpId::LogSoftmax, x, axis, execution);
    if (x.requires_grad() && detail::autograd::enabled()) {
        const Tensor saved = out; // see softmax
        detail::AutogradAccess::record(out, {x}, [x, saved, axis, execution](const Tensor& g) {
            detail::autograd::NoGradGuard guard;
            detail::autograd::propagate(x, log_softmax_backward(saved, g, axis, execution));
        });
    }
    return out;
}

Tensor softmax_backward(const Tensor& output, const Tensor& gradient, int axis,
                        Execution execution) {
    return backward(detail::OpId::SoftmaxBackward, output, gradient, axis, execution);
}

Tensor log_softmax_backward(const Tensor& output, const Tensor& gradient, int axis,
                            Execution execution) {
    return backward(detail::OpId::LogSoftmaxBackward, output, gradient, axis, execution);
}

} // namespace mensurflow
