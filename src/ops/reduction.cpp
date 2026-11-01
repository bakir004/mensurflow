#include "mensurflow/ops/reduction.hpp"

#include "core/dispatcher.hpp"
#include "core/autograd.hpp"

#include <stdexcept>
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
    if (op == detail::OpId::Sum && x.requires_grad()) {
        detail::AutogradAccess::record(
            out, {x}, [x](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                detail::autograd::propagate(
                    x, Tensor::full(x.shape(), gradient.item<float>(), DType::F32, x.device()));
            });
    }
    return out;
}

} // namespace

Tensor sum(const Tensor& x) { return reduce(detail::OpId::Sum, x, true); }

Tensor mean(const Tensor& x) { return reduce(detail::OpId::Mean, x, false); }

Tensor max(const Tensor& x) { return reduce(detail::OpId::Max, x, false); }

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
    if (x.size(normalized_axis) == 0 && op != detail::OpId::AxisSum) {
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
