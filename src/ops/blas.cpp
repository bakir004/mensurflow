// Validation and dispatch for BLAS operations. Kernels receive checked, contiguous inputs, except the
// CPU matmul, which reads strided views in place.
#include "mensurflow/ops/blas.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"
#include "mensurflow/ops/elementwise.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
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

Tensor dot(const Tensor& a, const Tensor& b, Execution execution) {
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
    detail::dispatch(detail::OpId::Dot, {{x, y}, {out}}, execution);
    return out;
}

Tensor scal(float alpha, const Tensor& x, Execution execution) {
    require_vector(x, "scal", "x");
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    if (x.size(0) == 0) {
        return out;
    }
    const Tensor input = x.strides()[0] == 1 ? x : x.contiguous();
    detail::dispatch(detail::OpId::Scal, {{input}, {out}, alpha}, execution);
    return out;
}

Tensor asum(const Tensor& x, Execution execution) {
    require_vector(x, "asum", "x");
    if (x.size(0) == 0) {
        const std::vector<std::byte> zero(dtype_size(x.dtype()));
        return Tensor::from_host(zero.data(), {}, x.dtype(), x.device());
    }
    Tensor out = Tensor::empty({}, x.dtype(), x.device());
    const Tensor input = x.strides()[0] == 1 ? x : x.contiguous();
    detail::dispatch(detail::OpId::Asum, {{input}, {out}}, execution);
    return out;
}

Tensor nrm2(const Tensor& x, Execution execution) {
    require_vector(x, "nrm2", "x");
    if (x.size(0) == 0) {
        const std::vector<std::byte> zero(dtype_size(x.dtype()));
        return Tensor::from_host(zero.data(), {}, x.dtype(), x.device());
    }

    Tensor out = Tensor::empty({}, x.dtype(), x.device());
    const Tensor input = x.strides()[0] == 1 ? x : x.contiguous();
    detail::dispatch(detail::OpId::Nrm2, {{input}, {out}}, execution);
    return out;
}

Tensor copy(const Tensor& x, Execution execution) {
    if (!x.defined()) {
        throw std::invalid_argument("copy: undefined tensor");
    }
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    if (x.numel() == 0) {
        return out;
    }
    const Tensor input = x.contiguous();
    detail::dispatch(detail::OpId::Copy, {{input}, {out}}, execution);
    return out;
}

Tensor matvec(const Tensor& a, const Tensor& x, Execution execution) {
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
    detail::dispatch(detail::OpId::Matvec, {{matrix, vector}, {out}}, execution);
    return out;
}

Tensor gemv(const Tensor& a, const Tensor& x, float alpha, const Tensor* y, float beta,
            bool transpose_a, Execution execution) {
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
    detail::dispatch(detail::OpId::Gemv, args, execution);
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b, Execution execution) {
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

    // The CPU kernel reads its inputs with any strides (its packing step walks every element once
    // anyway), so a transposed or sliced view goes through as it is; the backward pass below is made
    // of such views, and copying them would cost as much as the multiplication. The kernels of other
    // devices read flat row-major memory, so those inputs are made contiguous first.
    const bool reads_strides = a.device().type == DeviceType::CPU;
    const Tensor x = reads_strides || a.is_contiguous() ? a : a.contiguous();
    const Tensor y = reads_strides || b.is_contiguous() ? b : b.contiguous();
    detail::dispatch(detail::OpId::Matmul, {{x, y}, {out}}, execution);
    if (a.requires_grad() || b.requires_grad()) {
        detail::AutogradAccess::record(out, {a, b}, [a, b](const Tensor& gradient) {
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
            bool transpose_a, bool transpose_b, Execution execution) {
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
    Tensor result = matmul(left, right, execution);
    if (alpha != 1.0f) {
        result = mul(result, Tensor::full({}, alpha, DType::F32, result.device()), execution);
    }
    if (c != nullptr) {
        if (c->shape() != result.shape()) {
            throw std::invalid_argument("gemm: c shape does not match result");
        }
        if (beta != 0.0f) {
            result = add(result, scal(beta, c->view({c->numel()}), execution).view(c->shape()),
                         execution);
        }
    } else if (beta != 0.0f) {
        throw std::invalid_argument("gemm: beta requires c");
    }
    return result;
}

Tensor ger(float alpha, const Tensor& x, const Tensor& y, const Tensor* a, Execution execution) {
    require_vector(x, "ger", "x");
    require_vector(y, "ger", "y");
    if (a != nullptr) {
        require_matrix(*a, "ger", "a");
    }
    if (x.dtype() != y.dtype() || (a != nullptr && a->dtype() != x.dtype())) {
        throw std::invalid_argument("ger: dtypes differ");
    }
    if (x.device() != y.device() || (a != nullptr && a->device() != x.device())) {
        throw std::invalid_argument("ger: tensors are on different devices; move one with "
                                    "Tensor::to()");
    }
    const std::int64_t m = x.size(0), n = y.size(0);
    if (a != nullptr && a->shape() != Shape{m, n}) {
        throw std::invalid_argument("ger: a has shape " + to_string(a->shape()) + ", expected " +
                                    to_string(Shape{m, n}));
    }
    Tensor out = Tensor::empty({m, n}, x.dtype(), x.device());
    if (m == 0 || n == 0) {
        return out;
    }
    detail::OpArgs args{
        {x.strides()[0] == 1 ? x : x.contiguous(), y.strides()[0] == 1 ? y : y.contiguous()},
        {out}};
    if (a != nullptr) {
        args.inputs.push_back(a->contiguous());
    }
    args.alpha = alpha;
    detail::dispatch(detail::OpId::Ger, args, execution);
    return out;
}

Tensor gemm_strided_batched(const Tensor& a, const Tensor& b, float alpha, const Tensor* c,
                            float beta, bool transpose_a, bool transpose_b, Execution execution) {
    for (const auto& [t, name] : {std::pair<const Tensor*, const char*>{&a, "a"}, {&b, "b"}}) {
        if (!t->defined()) {
            throw std::invalid_argument(std::string("gemm_strided_batched: ") + name +
                                        " is an undefined tensor");
        }
        if (t->dim() < 2) {
            throw std::invalid_argument(std::string("gemm_strided_batched: ") + name +
                                        " must be at least 2-D, got shape " +
                                        to_string(t->shape()));
        }
    }
    if (a.dtype() != b.dtype() || (c != nullptr && c->dtype() != a.dtype())) {
        throw std::invalid_argument("gemm_strided_batched: dtypes differ");
    }
    if (a.device() != b.device() || (c != nullptr && c->device() != a.device())) {
        throw std::invalid_argument("gemm_strided_batched: tensors are on different devices; move "
                                    "one with Tensor::to()");
    }
    if (c == nullptr && beta != 0.0f) {
        throw std::invalid_argument("gemm_strided_batched: beta requires c");
    }

    // op(a) and op(b) as views: a transpose only swaps the last two strides.
    const Tensor left = transpose_a ? a.transpose(-2, -1) : a;
    const Tensor right = transpose_b ? b.transpose(-2, -1) : b;
    const std::int64_t m = left.size(-2), k = left.size(-1), n = right.size(-1);
    if (right.size(-2) != k) {
        throw std::invalid_argument("gemm_strided_batched: inner dimensions differ (" +
                                    to_string(left.shape()) + " times " + to_string(right.shape()) +
                                    ")");
    }
    const Shape left_batch(left.shape().begin(), left.shape().end() - 2);
    const Shape right_batch(right.shape().begin(), right.shape().end() - 2);
    Shape batch;
    try {
        batch = broadcast_shapes(left_batch, right_batch);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("gemm_strided_batched: batch shapes " + to_string(left_batch) +
                                    " and " + to_string(right_batch) + " do not broadcast");
    }
    Shape out_shape = batch;
    out_shape.push_back(m);
    out_shape.push_back(n);

    // c is broadcast to the result (a shared bias matrix needs no copy of its own) and packed, so
    // the kernel reads it with the result's layout.
    Tensor initial;
    if (c != nullptr) {
        bool broadcasts = c->dim() <= static_cast<int>(out_shape.size());
        if (broadcasts) {
            try {
                broadcasts = broadcast_shapes(c->shape(), out_shape) == out_shape;
            } catch (const std::invalid_argument&) {
                broadcasts = false;
            }
        }
        if (!broadcasts) {
            throw std::invalid_argument("gemm_strided_batched: c has shape " +
                                        to_string(c->shape()) + ", which does not broadcast to " +
                                        to_string(out_shape));
        }
        if (beta != 0.0f) {
            initial = c->expand(out_shape).contiguous();
        }
    }

    Tensor out = Tensor::empty(out_shape, a.dtype(), a.device());
    if (out.numel() == 0) {
        return out;
    }
    if (k == 0) {
        // No products: the result is beta * c, or zeros.
        if (initial.defined()) {
            return scal(beta, initial.view({initial.numel()}), execution).view(out_shape);
        }
        const std::vector<std::byte> zeros(out.nbytes());
        return Tensor::from_host(zeros.data(), out_shape, a.dtype(), a.device());
    }

    // Every matrix of the batch as a view with the full batch shape: broadcast batch dimensions get
    // stride 0, so the kernel reaches each matrix through its strides without copying anything.
    Shape left_shape = batch;
    left_shape.push_back(m);
    left_shape.push_back(k);
    Shape right_shape = batch;
    right_shape.push_back(k);
    right_shape.push_back(n);
    detail::OpArgs args{{left.expand(left_shape), right.expand(right_shape)}, {out}};
    if (initial.defined()) {
        args.inputs.push_back(initial);
    }
    args.alpha = alpha;
    args.beta = initial.defined() ? beta : 0.0f;
    detail::dispatch(detail::OpId::GemmBatched, args, execution);
    return out;
}

} // namespace mensurflow
