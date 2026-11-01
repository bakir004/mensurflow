#include "mensurflow/ops/reduction.hpp"

#include "core/dispatcher.hpp"
#include "core/autograd.hpp"

#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

Tensor reduce(detail::OpId op, const Tensor& x, bool allow_empty) {
    const std::string name = detail::op_name(op);
    if (!x.defined()) {
        throw std::invalid_argument(name + ": undefined tensor");
    }
    if (!allow_empty && x.numel() == 0) {
        throw std::invalid_argument(name + ": cannot reduce an empty tensor");
    }
    const Tensor input = x.contiguous();
    Tensor out = Tensor::empty({}, x.dtype(), x.device());
    detail::dispatch(op, {{input}, {out}});
    if ((op == detail::OpId::Sum || op == detail::OpId::Mean) && x.requires_grad()) {
        detail::AutogradAccess::record(
            out, {x}, [x, op](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const float scale = op == detail::OpId::Mean
                                        ? 1.0f / static_cast<float>(x.numel())
                                        : 1.0f;
                detail::autograd::propagate(
                    x, Tensor::full(x.shape(), gradient.item<float>() * scale, DType::F32,
                                    x.device()));
            });
    }
    return out;
}

} // namespace

Tensor sum(const Tensor& x) { return reduce(detail::OpId::Sum, x, true); }

Tensor mean(const Tensor& x) { return reduce(detail::OpId::Mean, x, false); }

Tensor max(const Tensor& x) { return reduce(detail::OpId::Max, x, false); }

Tensor min(const Tensor& x) { return reduce(detail::OpId::Min, x, false); }

Tensor prod(const Tensor& x) { return reduce(detail::OpId::Prod, x, true); }

Tensor reduce_axis(detail::OpId op, const Tensor& x, int axis, bool keepdim) {
    if (!x.defined()) {
        throw std::invalid_argument(std::string(detail::op_name(op)) + ": undefined tensor");
    }
    if (x.dim() == 0) {
        throw std::invalid_argument(std::string(detail::op_name(op)) + ": input must have a dimension");
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
    detail::dispatch(op, args);
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
                        const std::size_t output_index =
                            static_cast<std::size_t>(o * inner + i);
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
                detail::autograd::propagate(
                    x, Tensor::from_vector(result, x.shape(), x.device()));
            });
    }
    return out;
}

Tensor sum(const Tensor& x, int axis, bool keepdim) {
    return reduce_axis(detail::OpId::AxisSum, x, axis, keepdim);
}

Tensor mean(const Tensor& x, int axis, bool keepdim) {
    return reduce_axis(detail::OpId::AxisMean, x, axis, keepdim);
}

Tensor max(const Tensor& x, int axis, bool keepdim) {
    return reduce_axis(detail::OpId::AxisMax, x, axis, keepdim);
}

Tensor min(const Tensor& x, int axis, bool keepdim) {
    return reduce_axis(detail::OpId::AxisMin, x, axis, keepdim);
}

Tensor prod(const Tensor& x, int axis, bool keepdim) {
    return reduce_axis(detail::OpId::AxisProd, x, axis, keepdim);
}

Tensor argmax(const Tensor& x, int axis) {
    if (!x.defined()) {
        throw std::invalid_argument("argmax: undefined tensor");
    }
    if (x.dim() == 0) {
        throw std::invalid_argument("argmax: input must have at least one dimension");
    }
    const int normalized_axis = axis < 0 ? axis + x.dim() : axis;
    if (normalized_axis != x.dim() - 1) {
        throw std::invalid_argument("argmax: only the last axis is supported");
    }
    const std::int64_t width = x.size(x.dim() - 1);
    if (width == 0) {
        throw std::invalid_argument("argmax: empty axis");
    }
    Shape output_shape(x.shape().begin(), x.shape().end() - 1);
    Tensor out = Tensor::empty(std::move(output_shape), DType::I64, x.device());
    const Tensor input = x.is_contiguous() ? x : x.contiguous();
    detail::dispatch(detail::OpId::Argmax, {{input}, {out}});
    return out;
}

} // namespace mensurflow
