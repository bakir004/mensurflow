// Validation and dispatch for BLAS operations. Kernels receive checked, contiguous inputs.
#include "mensurflow/ops/blas.hpp"

#include "core/dispatcher.hpp"
#include "core/autograd.hpp"
#include "mensurflow/ops/elementwise.hpp"

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

Tensor scal(float alpha, const Tensor& x) {
    require_vector(x, "scal", "x");
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    if (x.size(0) == 0) {
        return out;
    }
    const Tensor input = x.strides()[0] == 1 ? x : x.contiguous();
    detail::dispatch(detail::OpId::Scal, {{input}, {out}, alpha});
    return out;
}

Tensor asum(const Tensor& x) {
    require_vector(x, "asum", "x");
    if (x.size(0) == 0) {
        const std::vector<std::byte> zero(dtype_size(x.dtype()));
        return Tensor::from_host(zero.data(), {}, x.dtype(), x.device());
    }
    Tensor out = Tensor::empty({}, x.dtype(), x.device());
    const Tensor input = x.strides()[0] == 1 ? x : x.contiguous();
    detail::dispatch(detail::OpId::Asum, {{input}, {out}});
    return out;
}

Tensor nrm2(const Tensor& x) {
    require_vector(x, "nrm2", "x");
    if (x.size(0) == 0) {
        const std::vector<std::byte> zero(dtype_size(x.dtype()));
        return Tensor::from_host(zero.data(), {}, x.dtype(), x.device());
    }

    Tensor out = Tensor::empty({}, x.dtype(), x.device());
    const Tensor input = x.strides()[0] == 1 ? x : x.contiguous();
    detail::dispatch(detail::OpId::Nrm2, {{input}, {out}});
    return out;
}

Tensor copy(const Tensor& x) {
    if (!x.defined()) {
        throw std::invalid_argument("copy: undefined tensor");
    }
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    if (x.numel() == 0) {
        return out;
    }
    const Tensor input = x.contiguous();
    detail::dispatch(detail::OpId::Copy, {{input}, {out}});
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

Tensor gemv(const Tensor& a, const Tensor& x, float alpha, const Tensor* y, float beta,
            bool transpose_a) {
    require_matrix(a, "gemv", "a");
    require_vector(x, "gemv", "x");
    if (y == nullptr && beta != 0.0f) {
        throw std::invalid_argument("gemv: beta requires an output vector");
    }
    if (a.dtype() != x.dtype() || (y != nullptr && a.dtype() != y->dtype())) {
        throw std::invalid_argument("gemv: dtypes differ");
    }
    if (a.device() != x.device() || (y != nullptr && a.device() != y->device())) {
        throw std::invalid_argument("gemv: tensors must share a device");
    }
    const std::int64_t rows = a.size(0);
    const std::int64_t cols = a.size(1);
    const std::int64_t input = transpose_a ? rows : cols;
    const std::int64_t output = transpose_a ? cols : rows;
    if (x.size(0) != input) {
        throw std::invalid_argument("gemv: input length does not match matrix dimensions");
    }
    if (y != nullptr && y->size(0) != output) {
        throw std::invalid_argument("gemv: output length does not match matrix dimensions");
    }
    Tensor out = Tensor::empty({output}, a.dtype(), a.device());
    detail::OpArgs args{{a.contiguous(), x.contiguous()}, {out}};
    if (y != nullptr) {
        args.inputs.push_back(y->contiguous());
    }
    args.alpha = alpha;
    args.beta = beta;
    args.transpose = transpose_a;
    detail::dispatch(detail::OpId::Gemv, args);
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
    if (a.requires_grad() || b.requires_grad()) {
        detail::AutogradAccess::record(
            out, {a, b}, [a, b](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                if (a.requires_grad()) {
                    detail::autograd::propagate(a, matmul(gradient, b.transpose(0, 1)));
                }
                if (b.requires_grad()) {
                    detail::autograd::propagate(b, matmul(a.transpose(0, 1), gradient));
                }
            });
    }
    return out;
}

Tensor gemm(const Tensor& a, const Tensor& b, float alpha, const Tensor* c, float beta,
            bool transpose_a, bool transpose_b) {
    require_matrix(a, "gemm", "a");
    require_matrix(b, "gemm", "b");
    if (a.dtype() != b.dtype() || (c != nullptr && c->dtype() != a.dtype())) {
        throw std::invalid_argument("gemm: dtypes differ");
    }
    if (a.device() != b.device() || (c != nullptr && c->device() != a.device())) {
        throw std::invalid_argument("gemm: tensors are on different devices");
    }
    const Tensor left = transpose_a ? a.transpose(0, 1) : a;
    const Tensor right = transpose_b ? b.transpose(0, 1) : b;
    Tensor result = matmul(left, right);
    if (alpha != 1.0f) {
        result = mul(result, Tensor::full({}, alpha, DType::F32, result.device()));
    }
    if (c != nullptr) {
        if (c->shape() != result.shape()) {
            throw std::invalid_argument("gemm: c shape does not match result");
        }
        if (beta != 0.0f) {
            result = add(result, scal(beta, c->view({c->numel()})).view(c->shape()));
        }
    } else if (beta != 0.0f) {
        throw std::invalid_argument("gemm: beta requires c");
    }
    return result;
}

} // namespace mensurflow
