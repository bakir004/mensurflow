// Validation and dispatch for BLAS operations. Kernels receive checked, contiguous inputs.
#include "mensurflow/ops/blas.hpp"

#include "core/dispatcher.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

void require_vector(const Tensor& t, const char* op, const char* name) {
    if (!t.defined()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " is an undefined tensor");
    }
    if (t.dim() != 1) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be 1-D, got shape " +
                                    to_string(t.shape()));
    }
}

} // namespace

Tensor dot(const Tensor& a, const Tensor& b) {
    require_vector(a, "dot", "a");
    require_vector(b, "dot", "b");
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument(std::string("dot: dtypes differ (") + dtype_name(a.dtype()) +
                                    " and " + dtype_name(b.dtype()) + ")");
    }
    if (a.device() != b.device()) {
        throw std::invalid_argument("dot: tensors are on " + to_string(a.device()) + " and " +
                                    to_string(b.device()) + "; move one with Tensor::to()");
    }
    if (a.size(0) != b.size(0)) {
        throw std::invalid_argument("dot: lengths differ (" + std::to_string(a.size(0)) + " and " +
                                    std::to_string(b.size(0)) + ")");
    }
    if (a.size(0) == 0) {
        // Nothing to launch; the sum of no products is zero.
        const std::vector<std::byte> zero(dtype_size(a.dtype()));
        return Tensor::from_host(zero.data(), {}, a.dtype(), a.device());
    }

    // Kernels assume stride 1 so every backend, including MPSGraph, can take the buffer as is.
    const Tensor x = a.strides()[0] == 1 ? a : a.contiguous();
    const Tensor y = b.strides()[0] == 1 ? b : b.contiguous();
    Tensor out = Tensor::empty({}, a.dtype(), a.device());
    detail::dispatch(detail::OpId::Dot, {{x, y}, {out}});
    return out;
}

} // namespace mensurflow
