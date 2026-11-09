// Validation, broadcasting, and dispatch for elementwise ops. Kernels receive contiguous
// inputs; binary arithmetic packs broadcast inputs to the output's shape, while bias_add
// keeps its bias vector compact.
#include "mensurflow/ops/elementwise.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

Tensor unary_op(detail::OpId op, const Tensor& x, Execution execution, float alpha = 1.0f,
                float beta = 20.0f, bool approximate = false) {
    if (!x.defined()) {
        throw std::invalid_argument(std::string(detail::op_name(op)) + ": undefined tensor");
    }
    const Tensor input = x.contiguous();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    detail::OpArgs args{{input}, {out}};
    args.alpha = alpha;
    args.beta = beta;
    args.approximate = approximate;
    detail::dispatch(op, args, execution);
    if (x.requires_grad()) {
        detail::AutogradAccess::record(out, {x}, [x, op, approximate](const Tensor& gradient) {
            const auto input = x.to_vector<float>();
            const auto upstream = gradient.to_vector<float>();
            std::vector<float> values(upstream.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                const float value = input[i];
                switch (op) {
                case detail::OpId::Neg:
                    values[i] = -upstream[i];
                    break;
                case detail::OpId::Abs:
                    values[i] = value < 0.0f ? -upstream[i] : upstream[i];
                    break;
                case detail::OpId::Square:
                    values[i] = 2.0f * value * upstream[i];
                    break;
                case detail::OpId::Relu:
                    values[i] = value > 0.0f ? upstream[i] : 0.0f;
                    break;
                case detail::OpId::Sqrt:
                    values[i] = upstream[i] / (2.0f * std::sqrt(value));
                    break;
                case detail::OpId::Exp:
                    values[i] = std::exp(value) * upstream[i];
                    break;
                case detail::OpId::Sigmoid: {
                    const float sigmoid_value = value >= 0.0f
                                                    ? 1.0f / (1.0f + std::exp(-value))
                                                    : (std::exp(value) / (1.0f + std::exp(value)));
                    values[i] = sigmoid_value * (1.0f - sigmoid_value) * upstream[i];
                    break;
                }
                case detail::OpId::Log:
                    values[i] = upstream[i] / value;
                    break;
                case detail::OpId::Rsqrt:
                    values[i] = -0.5f * std::pow(value, -1.5f) * upstream[i];
                    break;
                case detail::OpId::Tanh: {
                    const float tanh_value = std::tanh(value);
                    values[i] = (1.0f - tanh_value * tanh_value) * upstream[i];
                    break;
                }
                case detail::OpId::Gelu: {
                    const double v = value;
                    double derivative;
                    if (approximate) {
                        const double u =
                            0.79788456080286535588 * v * (1.0 + 0.044715 * v * v);
                        const double t = std::tanh(u);
                        derivative = 0.5 * (1.0 + t) +
                                     0.5 * v * (1.0 - t * t) *
                                         0.79788456080286535588 *
                                         (1.0 + 3.0 * 0.044715 * v * v);
                    } else {
                        derivative =
                            0.5 * std::erfc(-v * 0.70710678118654752440) +
                            v * 0.39894228040143267794 * std::exp(-0.5 * v * v);
                    }
                    values[i] = static_cast<float>(derivative) * upstream[i];
                    break;
                }
                case detail::OpId::Silu: {
                    const double v = value;
                    const double z = std::exp(-std::fabs(v));
                    const double normalized = v < 0.0 ? z / (1.0 + z) : 1.0 / (1.0 + z);
                    values[i] = static_cast<float>(
                                    normalized + static_cast<double>(value) * normalized *
                                                                (1.0 - normalized)) *
                                upstream[i];
                    break;
                }
                case detail::OpId::Sin:
                    values[i] = std::cos(value) * upstream[i];
                    break;
                case detail::OpId::Cos:
                    values[i] = -std::sin(value) * upstream[i];
                    break;
                case detail::OpId::Reciprocal:
                    values[i] = -upstream[i] / (value * value);
                    break;
                default:
                    throw std::logic_error("autograd: unsupported unary operation");
                }
            }
            detail::autograd::propagate(x, Tensor::from_vector(values, x.shape(), x.device()));
        });
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

Tensor comparison_op(detail::OpId op, const Tensor& a, const Tensor& b, Execution execution) {
    const std::string name = detail::op_name(op);
    require_compatible(a, b, name);
    Shape shape;
    try {
        shape = broadcast_shapes(a.shape(), b.shape());
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(name + ": " + e.what());
    }
    const auto packed = [&shape](const Tensor& input) {
        return input.shape() == shape && input.is_contiguous()
                   ? input
                   : input.expand(shape).contiguous();
    };
    const Tensor x = packed(a);
    const Tensor y = packed(b);
    Tensor out = Tensor::empty(shape, DType::Bool, a.device());
    // Always dispatch, including empty results, to check dtype/backend support.
    detail::dispatch(op, {{x, y}, {out}}, execution);
    return out;
}

Tensor binary_op(detail::OpId op, const Tensor& a, const Tensor& b, float alpha,
                 Execution execution) {
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
            detail::dispatch(op, {{a, b}, {out}, alpha}, execution);
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
    detail::dispatch(op, {{x, y}, {out}, alpha}, execution);
    if (a.requires_grad() || b.requires_grad()) {
        detail::AutogradAccess::record(out, {a, b}, [a, b, op, alpha](const Tensor& gradient) {
            detail::autograd::NoGradGuard guard;
            Tensor grad_a;
            Tensor grad_b;
            if (op == detail::OpId::Add) {
                grad_a = gradient;
                grad_b = gradient;
            } else if (op == detail::OpId::Sub) {
                grad_a = gradient;
                grad_b = neg(gradient);
            } else if (op == detail::OpId::Mul) {
                grad_a = mul(gradient, b);
                grad_b = mul(gradient, a);
            } else if (op == detail::OpId::Div) {
                grad_a = div(gradient, b);
                grad_b = neg(div(mul(gradient, a), square(b)));
            } else if (op == detail::OpId::Pow) {
                grad_a = mul(gradient, mul(b, pow(a, sub(b, Tensor::ones_like(b)))));
                grad_b = mul(gradient, mul(pow(a, b), log(a)));
            } else if (op == detail::OpId::Axpy) {
                grad_a = mul(gradient, Tensor::full({}, alpha, DType::F32, a.device()));
                grad_b = gradient;
            } else {
                throw std::logic_error("autograd: unsupported binary operation");
            }
            if (a.requires_grad()) {
                detail::autograd::propagate(a,
                                            detail::autograd::reduce_gradient(grad_a, a.shape()));
            }
            if (b.requires_grad()) {
                detail::autograd::propagate(b,
                                            detail::autograd::reduce_gradient(grad_b, b.shape()));
            }
        });
    }
    return out;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b, Execution execution) {
    return binary_op(detail::OpId::Add, a, b, 1.0f, execution);
}

Tensor sub(const Tensor& a, const Tensor& b, Execution execution) {
    return binary_op(detail::OpId::Sub, a, b, 1.0f, execution);
}

Tensor mul(const Tensor& a, const Tensor& b, Execution execution) {
    return binary_op(detail::OpId::Mul, a, b, 1.0f, execution);
}

Tensor div(const Tensor& a, const Tensor& b, Execution execution) {
    return binary_op(detail::OpId::Div, a, b, 1.0f, execution);
}

Tensor pow(const Tensor& a, const Tensor& b, Execution execution) {
    return binary_op(detail::OpId::Pow, a, b, 1.0f, execution);
}

Tensor axpy(float alpha, const Tensor& x, const Tensor& y, Execution execution) {
    return binary_op(detail::OpId::Axpy, x, y, alpha, execution);
}

Tensor bias_add(const Tensor& x, const Tensor& bias, Execution execution) {
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
    detail::dispatch(detail::OpId::BiasAdd, {{input, packed_bias}, {out}}, execution);
    if (x.requires_grad() || bias.requires_grad()) {
        detail::AutogradAccess::record(out, {x, bias}, [x, bias](const Tensor& gradient) {
            detail::autograd::NoGradGuard guard;
            if (x.requires_grad()) {
                detail::autograd::propagate(x, gradient);
            }
            if (bias.requires_grad()) {
                const auto values = gradient.to_vector<float>();
                std::vector<float> reduced(static_cast<std::size_t>(bias.numel()), 0.0f);
                for (std::int64_t row = 0; row < x.size(0); ++row) {
                    for (std::int64_t feature = 0; feature < x.size(1); ++feature) {
                        reduced[static_cast<std::size_t>(feature)] +=
                            values[static_cast<std::size_t>(row * x.size(1) + feature)];
                    }
                }
                detail::autograd::propagate(
                    bias, Tensor::from_vector(reduced, bias.shape(), bias.device()));
            }
        });
    }
    return out;
}

Tensor neg(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Neg, x, execution);
}

Tensor abs(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Abs, x, execution);
}

Tensor square(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Square, x, execution);
}

Tensor relu(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Relu, x, execution);
}

Tensor leaky_relu(const Tensor& x, float negative_slope, Execution execution) {
    return unary_op(detail::OpId::LeakyRelu, x, execution, negative_slope);
}

Tensor gelu(const Tensor& x, bool approximate, Execution execution) {
    return unary_op(detail::OpId::Gelu, x, execution, 1.0f, 20.0f, approximate);
}

Tensor silu(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Silu, x, execution);
}

Tensor elu(const Tensor& x, float alpha, Execution execution) {
    return unary_op(detail::OpId::Elu, x, execution, alpha);
}

Tensor softplus(const Tensor& x, float beta, float threshold, Execution execution) {
    if (!std::isfinite(beta) || beta <= 0.0f) {
        throw std::invalid_argument("softplus: beta must be positive and finite");
    }
    if (!std::isfinite(threshold)) {
        throw std::invalid_argument("softplus: threshold must be finite");
    }
    return unary_op(detail::OpId::Softplus, x, execution, beta, threshold);
}

namespace {

Tensor activation_backward(detail::OpId op, const Tensor& x, const Tensor& gradient,
                           bool approximate, Execution execution) {
    require_compatible(x, gradient, detail::op_name(op));
    if (x.shape() != gradient.shape()) {
        throw std::invalid_argument(std::string(detail::op_name(op)) +
                                    ": input and gradient shapes differ");
    }
    const Tensor input = x.contiguous();
    const Tensor upstream = gradient.contiguous();
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    detail::OpArgs args{{input, upstream}, {out}};
    args.approximate = approximate;
    detail::dispatch(op, args, execution);
    return out;
}

} // namespace

Tensor relu_backward(const Tensor& x, const Tensor& gradient, Execution execution) {
    return activation_backward(detail::OpId::ReluBackward, x, gradient, false, execution);
}

Tensor sigmoid_backward(const Tensor& x, const Tensor& gradient, Execution execution) {
    return activation_backward(detail::OpId::SigmoidBackward, x, gradient, false, execution);
}

Tensor tanh_backward(const Tensor& x, const Tensor& gradient, Execution execution) {
    return activation_backward(detail::OpId::TanhBackward, x, gradient, false, execution);
}

Tensor gelu_backward(const Tensor& x, const Tensor& gradient, bool approximate,
                     Execution execution) {
    return activation_backward(detail::OpId::GeluBackward, x, gradient, approximate, execution);
}

Tensor silu_backward(const Tensor& x, const Tensor& gradient, Execution execution) {
    return activation_backward(detail::OpId::SiluBackward, x, gradient, false, execution);
}

Tensor sqrt(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Sqrt, x, execution);
}

Tensor exp(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Exp, x, execution);
}

Tensor sigmoid(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Sigmoid, x, execution);
}

Tensor log(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Log, x, execution);
}

Tensor rsqrt(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Rsqrt, x, execution);
}

Tensor tanh(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Tanh, x, execution);
}

Tensor sin(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Sin, x, execution);
}

Tensor cos(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Cos, x, execution);
}

Tensor reciprocal(const Tensor& x, Execution execution) {
    return unary_op(detail::OpId::Reciprocal, x, execution);
}

Tensor minimum(const Tensor& a, const Tensor& b, Execution execution) {
    return binary_op(detail::OpId::Minimum, a, b, 1.0f, execution);
}

Tensor maximum(const Tensor& a, const Tensor& b, Execution execution) {
    return binary_op(detail::OpId::Maximum, a, b, 1.0f, execution);
}

Tensor clamp(const Tensor& x, float lower, float upper, Execution execution) {
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
    detail::dispatch(detail::OpId::Clamp, args, execution);
    return out;
}

Tensor eq(const Tensor& a, const Tensor& b, Execution execution) {
    return comparison_op(detail::OpId::Eq, a, b, execution);
}

Tensor ne(const Tensor& a, const Tensor& b, Execution execution) {
    return comparison_op(detail::OpId::Ne, a, b, execution);
}

Tensor lt(const Tensor& a, const Tensor& b, Execution execution) {
    return comparison_op(detail::OpId::Lt, a, b, execution);
}

Tensor le(const Tensor& a, const Tensor& b, Execution execution) {
    return comparison_op(detail::OpId::Le, a, b, execution);
}

Tensor gt(const Tensor& a, const Tensor& b, Execution execution) {
    return comparison_op(detail::OpId::Gt, a, b, execution);
}

Tensor ge(const Tensor& a, const Tensor& b, Execution execution) {
    return comparison_op(detail::OpId::Ge, a, b, execution);
}

Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y, Execution execution) {
    require_compatible(x, y, "where");
    if (!condition.defined()) {
        throw std::invalid_argument("where: undefined condition tensor");
    }
    if (condition.dtype() != DType::Bool) {
        throw std::invalid_argument("where: condition must have Bool dtype");
    }
    if (condition.device() != x.device()) {
        throw std::invalid_argument("where: condition and values must be on the same device");
    }
    Shape shape;
    try {
        shape = broadcast_shapes(broadcast_shapes(x.shape(), y.shape()), condition.shape());
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("where: ") + e.what());
    }
    const auto packed = [&shape](const Tensor& input) {
        return input.shape() == shape && input.is_contiguous()
                   ? input
                   : input.expand(shape).contiguous();
    };
    const Tensor a = packed(x);
    const Tensor b = packed(y);
    const Tensor mask = packed(condition);
    Tensor out = Tensor::empty(shape, x.dtype(), x.device());
    // The dispatch key uses the first input dtype, so keep values before the Bool condition.
    detail::dispatch(detail::OpId::Where, {{a, b, mask}, {out}}, execution);
    return out;
}

Tensor cast(const Tensor& x, DType dtype, Execution execution) {
    if (!x.defined()) {
        throw std::invalid_argument("cast: undefined tensor");
    }
    // Validate enum values before formatting errors or allocating the target.
    (void)dtype_name(dtype);
    const auto supported = [](DType type) {
        return type == DType::F32 || type == DType::I32 || type == DType::F16 ||
               type == DType::BF16;
    };
    if (!supported(x.dtype()) || !supported(dtype)) {
        throw detail::NotImplemented(std::string("cast: unsupported conversion from ") +
                                     dtype_name(x.dtype()) + " to " + dtype_name(dtype));
    }
    const Tensor input = x.contiguous();
    Tensor out = Tensor::empty(x.shape(), dtype, x.device());
    // The existing key selects the source dtype; the adapter reads the destination from out.
    detail::dispatch(detail::OpId::Cast, {{input}, {out}}, execution);
    return out;
}

} // namespace mensurflow
