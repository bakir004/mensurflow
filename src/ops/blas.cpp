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

void require_matrix(const Tensor& t, const char* op, const char* name) {
    if (!t.defined()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " is an undefined tensor");
    }
    if (t.dim() != 2) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be 2-D, got shape " +
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

Tensor matvec(const Tensor& a, const Tensor& x) {
    require_matrix(a, "matvec", "a");
    require_vector(x, "matvec", "x");
    if (a.dtype() != x.dtype()) {
        throw std::invalid_argument(std::string("matvec: dtypes differ (") + dtype_name(a.dtype()) +
                                    " and " + dtype_name(x.dtype()) + ")");
    }
    if (a.device() != x.device()) {
        throw std::invalid_argument("matvec: tensors are on " + to_string(a.device()) + " and " +
                                    to_string(x.device()) + "; move one with Tensor::to()");
    }
    const std::int64_t m = a.size(0), k = a.size(1);
    if (x.size(0) != k) {
        throw std::invalid_argument("matvec: inner dimensions differ (" + to_string(a.shape()) +
                                    " times " + to_string(x.shape()) + ")");
    }
    Tensor out = Tensor::empty({m}, a.dtype(), a.device());
    if (m == 0) {
        return out;
    }
    if (k == 0) {
        const std::vector<std::byte> zeros(out.nbytes());
        return Tensor::from_host(zeros.data(), {m}, a.dtype(), a.device());
    }
    const Tensor matrix = a.is_contiguous() ? a : a.contiguous();
    const Tensor vector = x.strides()[0] == 1 ? x : x.contiguous();
    detail::dispatch(detail::OpId::Matvec, {{matrix, vector}, {out}});
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    require_matrix(a, "matmul", "a");
    require_matrix(b, "matmul", "b");
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument(std::string("matmul: dtypes differ (") + dtype_name(a.dtype()) +
                                    " and " + dtype_name(b.dtype()) + ")");
    }
    if (a.device() != b.device()) {
        throw std::invalid_argument("matmul: tensors are on " + to_string(a.device()) + " and " +
                                    to_string(b.device()) + "; move one with Tensor::to()");
    }
    const std::int64_t m = a.size(0), k = a.size(1), n = b.size(1);
    if (b.size(0) != k) {
        throw std::invalid_argument("matmul: inner dimensions differ (" + to_string(a.shape()) +
                                    " times " + to_string(b.shape()) + ")");
    }
    Tensor out = Tensor::empty({m, n}, a.dtype(), a.device());
    if (m == 0 || n == 0) {
        return out; // empty result, nothing to compute
    }
    if (k == 0) {
        // The sum over no products is zero everywhere.
        const std::vector<std::byte> zeros(out.nbytes());
        return Tensor::from_host(zeros.data(), {m, n}, a.dtype(), a.device());
    }

    // Kernels read flat row-major memory, so a transposed or sliced input is packed first.
    const Tensor x = a.is_contiguous() ? a : a.contiguous();
    const Tensor y = b.is_contiguous() ? b : b.contiguous();
    detail::dispatch(detail::OpId::Matmul, {{x, y}, {out}});
    return out;
}

} // namespace mensurflow
