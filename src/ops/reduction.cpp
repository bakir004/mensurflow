#include "mensurflow/ops/reduction.hpp"

#include "core/dispatcher.hpp"

#include <stdexcept>
#include <string>

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

} // namespace mensurflow
