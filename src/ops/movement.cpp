// Validation and dispatch for data movement. concat and pad hand their kernels contiguous inputs;
// split only makes views.
#include "mensurflow/ops/movement.hpp"

#include "core/dispatcher.hpp"
#include "mensurflow/ops/blas.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

// normalize_dim throws out_of_range; these ops document invalid_argument for a bad axis.
int checked_axis(int axis, int ndim, const char* op) {
    if (ndim == 0) {
        throw std::invalid_argument(std::string(op) + ": a 0-d tensor has no axis");
    }

    const int out = axis < 0 ? axis + ndim : axis;
    if (out < 0 || out >= ndim) {
        throw std::invalid_argument(std::string(op) + ": axis " + std::to_string(axis) +
                                    " is out of range for a " + std::to_string(ndim) + "-d tensor");
    }
    return out;
}

void require_indices(const Tensor& indices, const Tensor& x, const char* op) {
    if (!indices.defined()) {
        throw std::invalid_argument(std::string(op) + ": undefined indices");
    }
    if (indices.dtype() != DType::I64) {
        throw std::invalid_argument(std::string(op) + ": indices must have I64 dtype");
    }
    if (indices.device() != x.device()) {
        throw std::invalid_argument(std::string(op) + ": tensors must be on the same device");
    }
}

void validate_indices(const Tensor& indices, std::int64_t limit, const char* op) {
    for (std::int64_t value : indices.to_vector<std::int64_t>()) {
        if (value < 0 || value >= limit) {
            throw std::invalid_argument(std::string(op) + ": index " + std::to_string(value) +
                                        " is out of range");
        }
    }
}

} // namespace

Tensor concat(const std::vector<Tensor>& tensors, int axis, Execution execution) {
    if (tensors.empty()) {
        throw std::invalid_argument("concat: no tensors");
    }
    for (const Tensor& t : tensors) {
        if (!t.defined()) {
            throw std::invalid_argument("concat: undefined tensor");
        }
    }
    const Tensor& first = tensors[0];
    const int dim = checked_axis(axis, first.dim(), "concat");
    Shape out_shape = first.shape();
    out_shape[static_cast<std::size_t>(dim)] = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const Tensor& t = tensors[i];
        if (t.dtype() != first.dtype()) {
            throw std::invalid_argument(std::string("concat: dtypes differ (") +
                                        dtype_name(first.dtype()) + " and " +
                                        dtype_name(t.dtype()) + ")");
        }
        if (t.device() != first.device()) {
            throw std::invalid_argument("concat: tensors are on " + to_string(first.device()) +
                                        " and " + to_string(t.device()) +
                                        "; move one with Tensor::to()");
        }
        bool same = t.dim() == first.dim();
        for (int d = 0; same && d < t.dim(); ++d) {
            same = d == dim || t.size(d) == first.size(d);
        }
        if (!same) {
            throw std::invalid_argument("concat: tensor " + std::to_string(i) + " has shape " +
                                        to_string(t.shape()) + ", which does not match " +
                                        to_string(first.shape()) + " outside axis " +
                                        std::to_string(dim));
        }
        out_shape[static_cast<std::size_t>(dim)] += t.size(dim);
    }

    Tensor out = Tensor::empty(out_shape, first.dtype(), first.device());
    if (out.numel() == 0) {
        return out;
    }
    detail::OpArgs args{{}, {out}};
    for (const Tensor& t : tensors) {
        args.inputs.push_back(t.contiguous());
    }
    args.axis = dim;
    detail::dispatch(detail::OpId::Concat, args, execution);
    return out;
}

std::vector<Tensor> split(const Tensor& x, const std::vector<std::int64_t>& sizes, int axis) {
    if (!x.defined()) {
        throw std::invalid_argument("split: undefined tensor");
    }
    const int dim = checked_axis(axis, x.dim(), "split");
    std::int64_t total = 0;
    for (std::int64_t size : sizes) {
        if (size < 0) {
            throw std::invalid_argument("split: sizes must be non-negative, got " +
                                        std::to_string(size));
        }
        total += size;
    }
    if (total != x.size(dim)) {
        throw std::invalid_argument("split: sizes add up to " + std::to_string(total) +
                                    ", but axis " + std::to_string(dim) + " has " +
                                    std::to_string(x.size(dim)) + " elements");
    }
    std::vector<Tensor> pieces;
    pieces.reserve(sizes.size());
    std::int64_t start = 0;
    for (std::int64_t size : sizes) {
        pieces.push_back(x.slice(dim, start, start + size));
        start += size;
    }
    return pieces;
}

std::vector<Tensor> split(const Tensor& x, std::int64_t size, int axis) {
    if (!x.defined()) {
        throw std::invalid_argument("split: undefined tensor");
    }
    const int dim = checked_axis(axis, x.dim(), "split");
    if (size < 1) {
        throw std::invalid_argument("split: size must be at least 1, got " + std::to_string(size));
    }
    std::vector<std::int64_t> sizes;
    for (std::int64_t left = x.size(dim); left > 0; left -= size) {
        sizes.push_back(left < size ? left : size);
    }
    if (sizes.empty()) {
        sizes.push_back(0); // an empty axis gives one empty piece, like torch.split
    }
    return split(x, sizes, dim);
}

Tensor pad(const Tensor& x, const std::vector<std::int64_t>& padding, float value,
           Execution execution) {
    if (!x.defined()) {
        throw std::invalid_argument("pad: undefined tensor");
    }
    if (padding.size() % 2 != 0) {
        throw std::invalid_argument("pad: padding needs (before, after) pairs, got " +
                                    std::to_string(padding.size()) + " values");
    }
    const int ndim = x.dim();
    if (padding.size() / 2 > static_cast<std::size_t>(ndim)) {
        throw std::invalid_argument("pad: " + std::to_string(padding.size() / 2) +
                                    " padded dimensions for a " + std::to_string(ndim) +
                                    "-d tensor");
    }
    for (std::int64_t amount : padding) {
        if (amount < 0) {
            throw std::invalid_argument("pad: negative padding (cropping) is not supported, got " +
                                        std::to_string(amount));
        }
    }
    if (padding.empty()) {
        return copy(x, execution); // nothing to add, but the result is still a new tensor
    }

    // padding starts at the last dimension; the kernel wants (before, after) in dimension order.
    std::vector<std::int64_t> per_dim(static_cast<std::size_t>(2 * ndim), 0);
    Shape out_shape = x.shape();
    for (std::size_t pair = 0; pair < padding.size() / 2; ++pair) {
        const std::size_t d = static_cast<std::size_t>(ndim) - 1 - pair;
        per_dim[2 * d] = padding[2 * pair];
        per_dim[2 * d + 1] = padding[2 * pair + 1];
        out_shape[d] += padding[2 * pair] + padding[2 * pair + 1];
    }
    Tensor out = Tensor::empty(out_shape, x.dtype(), x.device());
    if (out.numel() == 0) {
        return out;
    }
    detail::OpArgs args{{x.contiguous()}, {out}};
    args.alpha = value;
    args.padding = std::move(per_dim);
    detail::dispatch(detail::OpId::Pad, args, execution);
    return out;
}

Tensor gather(const Tensor& x, const Tensor& indices, int axis, Execution execution) {
    if (!x.defined()) {
        throw std::invalid_argument("gather: undefined input");
    }
    require_indices(indices, x, "gather");
    const int dim = checked_axis(axis, x.dim(), "gather");
    if (indices.dim() != x.dim()) {
        throw std::invalid_argument("gather: indices must have the same rank as input");
    }
    for (int d = 0; d < x.dim(); ++d) {
        if (d != dim && indices.size(d) > x.size(d)) {
            throw std::invalid_argument("gather: indices exceed input size on dimension " +
                                        std::to_string(d));
        }
    }
    validate_indices(indices, x.size(dim), "gather");
    Tensor out = Tensor::empty(indices.shape(), x.dtype(), x.device());
    detail::OpArgs args{{x.contiguous(), indices.contiguous()}, {out}};
    args.axis = dim;
    detail::dispatch(detail::OpId::Gather, args, execution);
    return out;
}

Tensor index_select(const Tensor& x, const Tensor& indices, int axis, Execution execution) {
    if (!x.defined()) {
        throw std::invalid_argument("index_select: undefined input");
    }
    require_indices(indices, x, "index_select");
    const int dim = checked_axis(axis, x.dim(), "index_select");
    if (indices.dim() != 1) {
        throw std::invalid_argument("index_select: indices must be one-dimensional");
    }
    validate_indices(indices, x.size(dim), "index_select");
    Shape output_shape = x.shape();
    output_shape[static_cast<std::size_t>(dim)] = indices.size(0);
    Tensor out = Tensor::empty(output_shape, x.dtype(), x.device());
    detail::OpArgs args{{x.contiguous(), indices.contiguous()}, {out}};
    args.axis = dim;
    detail::dispatch(detail::OpId::IndexSelect, args, execution);
    return out;
}

} // namespace mensurflow
