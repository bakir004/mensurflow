#include "mensurflow/ops/reduction.hpp"

#include "core/dispatcher.hpp"

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
    return out;
}

} // namespace

Tensor sum(const Tensor& x) { return reduce(detail::OpId::Sum, x, true); }

Tensor mean(const Tensor& x) { return reduce(detail::OpId::Mean, x, false); }

Tensor max(const Tensor& x) { return reduce(detail::OpId::Max, x, false); }

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
