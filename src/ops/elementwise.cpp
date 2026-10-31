// Validation, broadcasting, and dispatch for elementwise ops. Kernels receive contiguous
// inputs that already have the output's shape.
#include "mensurflow/ops/elementwise.hpp"

#include "core/dispatcher.hpp"

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
    return out;
}

Tensor binary_op(detail::OpId op, const Tensor& a, const Tensor& b) {
    const std::string name = detail::op_name(op);
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
    Shape shape;
    try {
        shape = broadcast_shapes(a.shape(), b.shape());
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(name + ": " + e.what());
    }
    if (numel(shape) == 0) {
        // Nothing to compute; the result is an empty tensor of the broadcast shape.
        return Tensor::empty(shape, a.dtype(), a.device());
    }

    // Kernels read flat memory, so an input that is smaller than the output, or not packed
    // row-major, is first expanded into a contiguous tensor of the output shape.
    auto packed = [&shape](const Tensor& t) {
        return t.shape() == shape && t.is_contiguous() ? t : t.expand(shape).contiguous();
    };
    const Tensor x = packed(a);
    const Tensor y = packed(b);
    Tensor out = Tensor::empty(shape, a.dtype(), a.device());
    detail::dispatch(op, {{x, y}, {out}});
    return out;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Add, a, b); }

Tensor sub(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Sub, a, b); }

Tensor mul(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Mul, a, b); }

Tensor div(const Tensor& a, const Tensor& b) { return binary_op(detail::OpId::Div, a, b); }

Tensor neg(const Tensor& x) { return unary_op(detail::OpId::Neg, x); }

Tensor abs(const Tensor& x) { return unary_op(detail::OpId::Abs, x); }

Tensor square(const Tensor& x) { return unary_op(detail::OpId::Square, x); }

Tensor relu(const Tensor& x) { return unary_op(detail::OpId::Relu, x); }

} // namespace mensurflow
