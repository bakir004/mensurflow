#include "mensurflow/ops/softmax.hpp"

#include "core/dispatcher.hpp"
#include "core/autograd.hpp"
#include "mensurflow/ops/elementwise.hpp"
#include "mensurflow/ops/reduction.hpp"

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
    if (normalized_axis < 0 || normalized_axis >= x.dim()) {
        throw std::invalid_argument("softmax: axis out of range");
    }
    if (x.size(normalized_axis) == 0) {
        throw std::invalid_argument("softmax: axis is empty");
    }
    Tensor out;
    if (normalized_axis != x.dim() - 1) {
        const Tensor moved = x.transpose(normalized_axis, x.dim() - 1).contiguous();
        Tensor moved_out = Tensor::empty(moved.shape(), moved.dtype(), moved.device());
        detail::dispatch(detail::OpId::Softmax, {{moved}, {moved_out}});
        out = moved_out.transpose(normalized_axis, x.dim() - 1).contiguous();
    } else {
        const std::int64_t width = x.size(x.dim() - 1);
        const std::int64_t rows = width == 0 ? 0 : x.numel() / width;
        out = Tensor::empty(x.shape(), x.dtype(), x.device());
        if (rows == 0 || width == 0) {
            return out;
        }
        const Tensor input = x.is_contiguous() ? x : x.contiguous();
        detail::dispatch(detail::OpId::Softmax, {{input}, {out}});
    }
    if (x.requires_grad()) {
        detail::AutogradAccess::record(
            out, {x}, [x, out, normalized_axis](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const Tensor weighted = mul(gradient, out);
                const Tensor summed = sum(weighted, normalized_axis, true);
                detail::autograd::propagate(x, mul(out, sub(gradient, summed)));
            });
    }
    return out;
}

} // namespace mensurflow
