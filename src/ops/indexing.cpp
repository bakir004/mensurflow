// Validation and dispatch for one-hot labels and embedding lookup/scatter-add.
#include "mensurflow/ops/indexing.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

void require_indices(const Tensor& indices, const char* op) {
    if (!indices.defined()) {
        throw std::invalid_argument(std::string(op) + ": undefined indices");
    }
    if (indices.dtype() != DType::I64) {
        throw std::invalid_argument(std::string(op) + ": indices must have I64 dtype");
    }
}

void require_float(const Tensor& values, const Tensor& indices, const char* op) {
    if (!values.defined()) {
        throw std::invalid_argument(std::string(op) + ": undefined input");
    }
    if (values.dtype() != DType::F32) {
        throw std::invalid_argument(std::string(op) + ": only F32 values are supported");
    }
    if (values.device() != indices.device()) {
        throw std::invalid_argument(std::string(op) + ": tensors must be on the same device");
    }
}

void validate_indices(const Tensor& indices, std::int64_t limit, const char* op) {
    // Validate outside the kernel: throwing from inside an OpenMP loop is not supported.
    for (const std::int64_t index : indices.to_vector<std::int64_t>()) {
        if (index < 0 || index >= limit) {
            throw std::invalid_argument(std::string(op) + ": index " + std::to_string(index) +
                                        " is out of range [0, " + std::to_string(limit) + ")");
        }
    }
}

} // namespace

Tensor one_hot(const Tensor& indices, std::int64_t num_classes, Execution execution) {
    require_indices(indices, "one_hot");
    if (num_classes <= 0) {
        throw std::invalid_argument("one_hot: num_classes must be positive");
    }
    validate_indices(indices, num_classes, "one_hot");
    Shape shape = indices.shape();
    shape.push_back(num_classes);
    Tensor out = Tensor::empty(shape, DType::I64, indices.device());
    // Dispatch even an empty tensor so unsupported execution requests still fail.
    detail::dispatch(detail::OpId::OneHot, {{indices.contiguous()}, {out}}, execution);
    return out;
}

Tensor embedding(const Tensor& weight, const Tensor& indices, Execution execution) {
    require_indices(indices, "embedding");
    require_float(weight, indices, "embedding");
    if (weight.dim() != 2) {
        throw std::invalid_argument("embedding: weight must be a two-dimensional table");
    }
    validate_indices(indices, weight.size(0), "embedding");
    Shape shape = indices.shape();
    shape.push_back(weight.size(1));
    Tensor out = Tensor::empty(shape, weight.dtype(), weight.device());
    detail::dispatch(detail::OpId::Embedding,
                     {{weight.contiguous(), indices.contiguous()}, {out}}, execution);
    if (weight.requires_grad() && detail::autograd::enabled()) {
        // Tensor copies share storage. Save independent IDs so edits after forward cannot move
        // gradients into different rows when backward eventually runs.
        const Tensor saved_indices = Tensor::from_vector(indices.to_vector<std::int64_t>(),
                                                         indices.shape(), indices.device());
        detail::AutogradAccess::record(
            out, {weight}, [weight, saved_indices, execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                detail::autograd::propagate(
                    weight, embedding_backward(gradient, saved_indices, weight.size(0), execution));
            });
    }
    return out;
}

Tensor embedding_backward(const Tensor& gradient, const Tensor& indices,
                          std::int64_t num_embeddings, Execution execution) {
    require_indices(indices, "embedding_backward");
    require_float(gradient, indices, "embedding_backward");
    if (num_embeddings < 0) {
        throw std::invalid_argument("embedding_backward: num_embeddings must be non-negative");
    }
    if (gradient.dim() != indices.dim() + 1) {
        throw std::invalid_argument("embedding_backward: gradient rank must be indices rank + 1");
    }
    for (int d = 0; d < indices.dim(); ++d) {
        if (gradient.size(d) != indices.size(d)) {
            throw std::invalid_argument(
                "embedding_backward: gradient shape does not match indices");
        }
    }
    validate_indices(indices, num_embeddings, "embedding_backward");
    Tensor out = Tensor::empty({num_embeddings, gradient.size(-1)}, DType::F32, gradient.device());
    detail::dispatch(detail::OpId::EmbeddingBackward,
                     {{gradient.contiguous(), indices.contiguous()}, {out}}, execution);
    return out;
}

} // namespace mensurflow
