#include "mensurflow/ops/reduction.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

Tensor reduce(detail::OpId op, const Tensor& x, bool allow_empty, Execution execution) {
    const std::string name = detail::op_name(op);
    if (!x.defined()) {
        throw std::invalid_argument(name + ": undefined tensor");
    }
    if (!allow_empty && x.numel() == 0) {
        throw std::invalid_argument(name + ": cannot reduce an empty tensor");
    }
    const Tensor input = x.contiguous();
    Tensor out = Tensor::empty({}, x.dtype(), x.device());
    detail::dispatch(op, {{input}, {out}}, execution);
    if ((op == detail::OpId::Sum || op == detail::OpId::Mean) && x.requires_grad()) {
        detail::AutogradAccess::record(out, {x}, [x, op](const Tensor& gradient) {
            detail::autograd::NoGradGuard guard;
            const float scale =
                op == detail::OpId::Mean ? 1.0f / static_cast<float>(x.numel()) : 1.0f;
            detail::autograd::propagate(
                x, Tensor::full(x.shape(), gradient.item<float>() * scale, DType::F32, x.device()));
        });
    }
    return out;
}

} // namespace

Tensor sum(const Tensor& x, Execution execution) {
    return reduce(detail::OpId::Sum, x, true, execution);
}

Tensor mean(const Tensor& x, Execution execution) {
    return reduce(detail::OpId::Mean, x, false, execution);
}

Tensor max(const Tensor& x, Execution execution) {
    return reduce(detail::OpId::Max, x, false, execution);
}

Tensor min(const Tensor& x, Execution execution) {
    return reduce(detail::OpId::Min, x, false, execution);
}

Tensor prod(const Tensor& x, Execution execution) {
    return reduce(detail::OpId::Prod, x, true, execution);
}

Tensor reduce_axis(detail::OpId op, const Tensor& x, int axis, bool keepdim, Execution execution) {
    if (!x.defined()) {
        throw std::invalid_argument(std::string(detail::op_name(op)) + ": undefined tensor");
    }
    if (x.dim() == 0) {
        throw std::invalid_argument(std::string(detail::op_name(op)) +
                                    ": input must have a dimension");
    }
    const int normalized_axis = axis < 0 ? axis + x.dim() : axis;
    if (normalized_axis < 0 || normalized_axis >= x.dim()) {
        throw std::invalid_argument(std::string(detail::op_name(op)) + ": axis out of range");
    }
    if (x.size(normalized_axis) == 0 && op != detail::OpId::AxisSum &&
        op != detail::OpId::AxisProd) {
        throw std::invalid_argument(std::string(detail::op_name(op)) + ": axis is empty");
    }
    Shape output_shape = x.shape();
    if (keepdim) {
        output_shape[static_cast<std::size_t>(normalized_axis)] = 1;
    } else {
        output_shape.erase(output_shape.begin() + normalized_axis);
    }
    Tensor out = Tensor::empty(output_shape, x.dtype(), x.device());
    const Tensor input = x.contiguous();
    detail::OpArgs args{{input}, {out}};
    args.axis = normalized_axis;
    args.keepdim = keepdim;
    detail::dispatch(op, args, execution);
    if ((op == detail::OpId::AxisSum || op == detail::OpId::AxisMean ||
         op == detail::OpId::AxisMax) &&
        x.requires_grad()) {
        detail::AutogradAccess::record(
            out, {x}, [x, op, normalized_axis, keepdim](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const auto input = x.to_vector<float>();
                const auto upstream = gradient.to_vector<float>();
                std::vector<float> result(input.size(), 0.0f);
                const std::int64_t width = x.size(normalized_axis);
                const std::int64_t inner = [&] {
                    std::int64_t value = 1;
                    for (int d = normalized_axis + 1; d < x.dim(); ++d) {
                        value *= x.size(d);
                    }
                    return value;
                }();
                const std::int64_t outer = x.numel() / (width * inner);
                for (std::int64_t o = 0; o < outer; ++o) {
                    for (std::int64_t i = 0; i < inner; ++i) {
                        const std::size_t output_index = static_cast<std::size_t>(o * inner + i);
                        const float upstream_value =
                            upstream[keepdim ? static_cast<std::size_t>((o * 1 + 0) * inner + i)
                                             : output_index];
                        std::int64_t selected = -1;
                        if (op == detail::OpId::AxisMax) {
                            float maximum = -std::numeric_limits<float>::infinity();
                            for (std::int64_t r = 0; r < width; ++r) {
                                const float value =
                                    input[static_cast<std::size_t>((o * width + r) * inner + i)];
                                if (value > maximum) {
                                    maximum = value;
                                    selected = r;
                                }
                            }
                        }
                        for (std::int64_t r = 0; r < width; ++r) {
                            if (op == detail::OpId::AxisMax && r != selected) {
                                continue;
                            }
                            const float scale = op == detail::OpId::AxisMean
                                                    ? 1.0f / static_cast<float>(width)
                                                    : 1.0f;
                            result[static_cast<std::size_t>((o * width + r) * inner + i)] =
                                upstream_value * scale;
                        }
                    }
                }
                detail::autograd::propagate(x, Tensor::from_vector(result, x.shape(), x.device()));
            });
    }
    return out;
}

Tensor sum(const Tensor& x, int axis, bool keepdim, Execution execution) {
    return reduce_axis(detail::OpId::AxisSum, x, axis, keepdim, execution);
}

Tensor mean(const Tensor& x, int axis, bool keepdim, Execution execution) {
    return reduce_axis(detail::OpId::AxisMean, x, axis, keepdim, execution);
}

Tensor max(const Tensor& x, int axis, bool keepdim, Execution execution) {
    return reduce_axis(detail::OpId::AxisMax, x, axis, keepdim, execution);
}

Tensor min(const Tensor& x, int axis, bool keepdim, Execution execution) {
    return reduce_axis(detail::OpId::AxisMin, x, axis, keepdim, execution);
}

Tensor prod(const Tensor& x, int axis, bool keepdim, Execution execution) {
    return reduce_axis(detail::OpId::AxisProd, x, axis, keepdim, execution);
}

namespace {

// argmax and argmin: the same validation (last axis only, non-empty) and an I64 result.
Tensor arg_reduce(detail::OpId op, const Tensor& x, int axis, Execution execution) {
    const std::string name = detail::op_name(op);
    if (!x.defined()) {
        throw std::invalid_argument(name + ": undefined tensor");
    }
    if (x.dim() == 0) {
        throw std::invalid_argument(name + ": input must have at least one dimension");
    }
    const int normalized_axis = axis < 0 ? axis + x.dim() : axis;
    if (normalized_axis != x.dim() - 1) {
        throw std::invalid_argument(name + ": only the last axis is supported");
    }
    const std::int64_t width = x.size(x.dim() - 1);
    if (width == 0) {
        throw std::invalid_argument(name + ": empty axis");
    }
    Shape output_shape(x.shape().begin(), x.shape().end() - 1);
    Tensor out = Tensor::empty(std::move(output_shape), DType::I64, x.device());
    const Tensor input = x.is_contiguous() ? x : x.contiguous();
    detail::dispatch(op, {{input}, {out}}, execution);
    return out;
}

} // namespace

Tensor argmax(const Tensor& x, int axis, Execution execution) {
    return arg_reduce(detail::OpId::Argmax, x, axis, execution);
}

Tensor argmin(const Tensor& x, int axis, Execution execution) {
    return arg_reduce(detail::OpId::Argmin, x, axis, execution);
}

namespace {

// Validation shared by var and std and their backward passes: returns the normalized axis and
// sets out_shape to the reduced shape.
int moment_axis(const std::string& name, const Tensor& x, int axis, bool keepdim,
                std::int64_t correction, Shape& out_shape) {
    if (!x.defined()) {
        throw std::invalid_argument(name + ": undefined tensor");
    }
    if (x.dim() == 0) {
        throw std::invalid_argument(name + ": input must have at least one dimension");
    }
    const int normalized_axis = axis < 0 ? axis + x.dim() : axis;
    if (normalized_axis < 0 || normalized_axis >= x.dim()) {
        throw std::invalid_argument(name + ": axis " + std::to_string(axis) +
                                    " out of range for " + std::to_string(x.dim()) +
                                    " dimensions");
    }
    if (correction < 0) {
        throw std::invalid_argument(name + ": correction must be non-negative, got " +
                                    std::to_string(correction));
    }
    out_shape = x.shape();
    if (keepdim) {
        out_shape[static_cast<std::size_t>(normalized_axis)] = 1;
    } else {
        out_shape.erase(out_shape.begin() + normalized_axis);
    }
    return normalized_axis;
}

detail::OpArgs moment_args(std::vector<Tensor> inputs, Tensor out, int axis,
                           std::int64_t correction) {
    detail::OpArgs args{std::move(inputs), {std::move(out)}};
    args.axis = axis;
    args.correction = correction;
    return args;
}

// The gradient of var or std (backward_op) with respect to x.
Tensor moment_backward(detail::OpId backward_op, const Tensor& grad_output, const Tensor& x,
                       int axis, bool keepdim, std::int64_t correction, Execution execution) {
    const std::string name = detail::op_name(backward_op);
    Shape out_shape;
    const int normalized_axis = moment_axis(name, x, axis, keepdim, correction, out_shape);
    if (!grad_output.defined()) {
        throw std::invalid_argument(name + ": undefined grad_output");
    }
    if (grad_output.shape() != out_shape) {
        throw std::invalid_argument(name + ": grad_output has shape " +
                                    to_string(grad_output.shape()) + ", expected " +
                                    to_string(out_shape));
    }
    if (grad_output.dtype() != x.dtype() || grad_output.device() != x.device()) {
        throw std::invalid_argument(name + ": grad_output and x differ in dtype or device");
    }
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    detail::dispatch(backward_op,
                     moment_args({x.contiguous(), grad_output.contiguous()}, out,
                                 normalized_axis, correction),
                     execution);
    return out;
}

// var or std (op) along one axis, recording autograd through `backward`.
template <typename Backward>
Tensor moment(detail::OpId op, const Tensor& x, int axis, bool keepdim, std::int64_t correction,
              Execution execution, Backward backward) {
    Shape out_shape;
    const int normalized_axis =
        moment_axis(detail::op_name(op), x, axis, keepdim, correction, out_shape);
    Tensor out = Tensor::empty(out_shape, x.dtype(), x.device());
    detail::dispatch(op, moment_args({x.contiguous()}, out, normalized_axis, correction),
                     execution);
    if (x.requires_grad()) {
        detail::AutogradAccess::record(
            out, {x}, [x, axis, keepdim, correction, execution, backward](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                detail::autograd::propagate(
                    x, backward(gradient, x, axis, keepdim, correction, execution));
            });
    }
    return out;
}

} // namespace

Tensor var(const Tensor& x, int axis, bool keepdim, std::int64_t correction, Execution execution) {
    return moment(detail::OpId::Var, x, axis, keepdim, correction, execution, var_backward);
}

Tensor var_backward(const Tensor& grad_output, const Tensor& x, int axis, bool keepdim,
                    std::int64_t correction, Execution execution) {
    return moment_backward(detail::OpId::VarBackward, grad_output, x, axis, keepdim, correction,
                           execution);
}

} // namespace mensurflow
