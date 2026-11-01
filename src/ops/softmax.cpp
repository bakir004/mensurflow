#include "mensurflow/ops/softmax.hpp"

#include "core/dispatcher.hpp"

#include <stdexcept>

namespace mensurflow {

Tensor softmax(const Tensor& x, int axis) {
    if (!x.defined()) {
        throw std::invalid_argument("softmax: undefined tensor");
    }
    if (x.dim() == 0) {
        throw std::invalid_argument("softmax: input must have at least one dimension");
    }
    const int normalized_axis = axis < 0 ? axis + x.dim() : axis;
    if (normalized_axis != x.dim() - 1) {
        throw std::invalid_argument("softmax: only the last axis is supported");
    }
    const std::int64_t width = x.size(x.dim() - 1);
    const std::int64_t rows = width == 0 ? 0 : x.numel() / width;
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    if (rows == 0 || width == 0) {
        return out;
    }
    const Tensor input = x.is_contiguous() ? x : x.contiguous();
    detail::dispatch(detail::OpId::Softmax, {{input}, {out}});
    return out;
}

} // namespace mensurflow
