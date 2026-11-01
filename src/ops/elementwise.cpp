// Validation, broadcasting, and dispatch for elementwise ops. Kernels receive contiguous
// inputs; binary arithmetic packs broadcast inputs to the output's shape, while bias_add
// keeps its bias vector compact.
#include "mensurflow/ops/elementwise.hpp"

#include "core/dispatcher.hpp"
#include "core/autograd.hpp"

#include <stdexcept>
#include <string>

namespace mensurflow {

namespace {

Tensor unary_op(detail::OpId op, const Tensor& x) {
    if (!x.defined()) {
        throw std::invalid_argument(std::string(detail::op_name(op)) + ": undefined tensor");
    }
    const Tensor input = x.contiguous();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    detail::dispatch(op, {{input}, {out}});
    if (x.requires_grad()) {
        if (op == detail::OpId::Neg) {
            detail::AutogradAccess::record(
                out, {x}, [x](const Tensor& gradient) {
                    detail::autograd::NoGradGuard guard;
                    detail::autograd::propagate(x, neg(gradient));
                });
        } else if (op == detail::OpId::Relu) {
            detail::AutogradAccess::record(
                out, {x}, [x](const Tensor& gradient) {
                    const auto input = x.to_vector<float>();
                    auto values = gradient.to_vector<float>();
                    for (std::size_t i = 0; i < values.size(); ++i) {
                        values[i] = input[i] > 0.0f ? values[i] : 0.0f;
                    }
                    detail::autograd::propagate(
                        x, Tensor::from_vector(values, x.shape(), x.device()));
                });
        }
    }
    return out;
}

void require_compatible(const Tensor& a, const Tensor& b, const std::string& name) {
    if (!a.defined() || !b.defined()) {
        throw std::invalid_argument(name + ": undefined tensor");
    }
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument(name + ": dtypes differ (" + dtype_name(a.dtype()) + " and " +
                                    dtype_name(b.dtype()) + ")");
    }
    if (a.device() != b.device()) {
        throw std::invalid_argument(name + ": tensors are on " + to_string(a.device()) + " and " +
                                    to_string(b.device()) + "; move one with Tensor::to()");
    }
}

Tensor binary_op(detail::OpId op, const Tensor& a, const Tensor& b, float alpha = 1.0f) {
    const std::string name = detail::op_name(op);
    require_compatible(a, b, name);
    Shape shape;
    try {
        shape = broadcast_shapes(a.shape(), b.shape());
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(name + ": " + e.what());
    }
    if (numel(shape) == 0) {
        Tensor out = Tensor::empty(shape, a.dtype(), a.device());
        if (op == detail::OpId::Axpy) {
            // New operations check kernel support even when there is no work to launch.
            detail::dispatch(op, {{a, b}, {out}, alpha});
        }
        return out;
    }

    // Kernels read flat memory, so an input that is smaller than the output, or not packed
    // row-major, is first expanded into a contiguous tensor of the output shape.
    auto packed = [&shape](const Tensor& t) {
        return t.shape() == shape && t.is_contiguous() ? t : t.expand(shape).contiguous();
    };
    const Tensor x = packed(a);
    const Tensor y = packed(b);
    Tensor out = Tensor::empty(shape, a.dtype(), a.device());
    detail::dispatch(op, {{x, y}, {out}, alpha});
    if (a.requires_grad() || b.requires_grad()) {
        if (a.shape() != b.shape()) {
            throw std::logic_error("autograd: broadcasted gradients are not implemented yet");
        }
        if (op == detail::OpId::Add || op == detail::OpId::Mul) {
            detail::AutogradAccess::record(
                out, {a, b}, [a, b, op](const Tensor& gradient) {
                    detail::autograd::NoGradGuard guard;
                    if (op == detail::OpId::Add) {
                        detail::autograd::propagate(a, gradient);
                        detail::autograd::propagate(b, gradient);
                    } else {
                        detail::autograd::propagate(a, mul(gradient, b));
                        detail::autograd::propagate(b, mul(gradient, a));
                    }
                });
        }
    }
    return out;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Add, a, b); }

Tensor sub(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Sub, a, b); }

Tensor mul(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Mul, a, b); }

Tensor div(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Div, a, b); }

Tensor axpy(float alpha, const Tensor& x, const Tensor& y) {
    return binary_op(detail::OpId::Axpy, x, y, alpha);
}

Tensor bias_add(const Tensor& x, const Tensor& bias) {
    require_compatible(x, bias, "bias_add");
    if (x.dim() != 2 || bias.dim() != 1) {
        throw std::invalid_argument("bias_add: expected [batch, features] and [features], got " +
                                    to_string(x.shape()) + " and " + to_string(bias.shape()));
    }
    if (x.size(1) != bias.size(0)) {
        throw std::invalid_argument("bias_add: bias length must match the matrix's feature count");
    }
    const Tensor input = x.contiguous();
    const Tensor packed_bias = bias.contiguous();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    detail::dispatch(detail::OpId::BiasAdd, {{input, packed_bias}, {out}});
    return out;
}

Tensor neg(const Tensor& x) { return unary_op(detail::OpId::Neg, x); }

Tensor abs(const Tensor& x) { return unary_op(detail::OpId::Abs, x); }

Tensor square(const Tensor& x) { return unary_op(detail::OpId::Square, x); }

Tensor relu(const Tensor& x) { return unary_op(detail::OpId::Relu, x); }

Tensor sqrt(const Tensor& x) { return unary_op(detail::OpId::Sqrt, x); }

Tensor exp(const Tensor& x) { return unary_op(detail::OpId::Exp, x); }

Tensor sigmoid(const Tensor& x) { return unary_op(detail::OpId::Sigmoid, x); }

Tensor log(const Tensor& x) { return unary_op(detail::OpId::Log, x); }

Tensor rsqrt(const Tensor& x) { return unary_op(detail::OpId::Rsqrt, x); }

Tensor tanh(const Tensor& x) { return unary_op(detail::OpId::Tanh, x); }

Tensor minimum(const Tensor& a, const Tensor& b) {
    return binary_op(detail::OpId::Minimum, a, b);
}

Tensor maximum(const Tensor& a, const Tensor& b) {
    return binary_op(detail::OpId::Maximum, a, b);
}

Tensor clamp(const Tensor& x, float lower, float upper) {
    if (lower > upper) {
        throw std::invalid_argument("clamp: lower bound must not exceed upper bound");
    }
    if (!x.defined()) {
        throw std::invalid_argument("clamp: undefined tensor");
    }
    const Tensor input = x.contiguous();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    detail::OpArgs args{{input}, {out}};
    args.alpha = lower;
    args.beta = upper;
    detail::dispatch(detail::OpId::Clamp, args);
    return out;
}

} // namespace mensurflow
