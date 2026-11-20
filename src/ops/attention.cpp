// Naive scaled dot-product attention from existing ops: gemm_strided_batched for q k^T, where()
// to mask scores with -inf, softmax over the keys, and gemm_strided_batched again for the
// weighted sum of v. No new kernel: the cost is the two GEMMs, two or three passes over the
// [..., Lq, Lk] scores, and the memory to hold them.
#include "mensurflow/ops/attention.hpp"

#include "mensurflow/ops/blas.hpp"
#include "mensurflow/ops/elementwise.hpp"
#include "mensurflow/ops/softmax.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

void require_operand(const Tensor& t, const char* name, const Tensor& q) {
    if (!t.defined()) {
        throw std::invalid_argument(std::string("attention: undefined ") + name);
    }
    if (t.dtype() != DType::F32) {
        throw std::invalid_argument(std::string("attention: ") + name + " must be F32, got " +
                                    dtype_name(t.dtype()));
    }
    if (t.dim() < 2) {
        throw std::invalid_argument(std::string("attention: ") + name +
                                    " needs at least 2 dimensions, got " + to_string(t.shape()));
    }
    if (t.device() != q.device()) {
        throw std::invalid_argument("attention: q, k and v must be on the same device");
    }
}

// Scores outside `keep` become -inf, so softmax gives them zero weight.
Tensor masked(const Tensor& scores, const Tensor& keep, Execution execution) {
    const Tensor minus_infinity =
        Tensor::full({}, -std::numeric_limits<double>::infinity(), DType::F32, scores.device());
    return where(keep, scores, minus_infinity, execution);
}

} // namespace

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, bool causal,
                 float scale, Execution execution) {
    require_operand(q, "q", q);
    require_operand(k, "k", q);
    require_operand(v, "v", q);
    const std::int64_t depth = q.size(-1);
    const std::int64_t queries = q.size(-2);
    const std::int64_t keys = k.size(-2);
    if (k.size(-1) != depth) {
        throw std::invalid_argument("attention: q " + to_string(q.shape()) + " and k " +
                                    to_string(k.shape()) + " differ in the last dimension");
    }
    if (v.size(-2) != keys) {
        throw std::invalid_argument("attention: k " + to_string(k.shape()) + " and v " +
                                    to_string(v.shape()) + " differ in the number of keys");
    }
    if (scale == 0.0f) {
        scale = depth > 0 ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(depth))) : 1.0f;
    }

    // [..., Lq, Lk], already scaled: the GEMM's alpha does the multiply.
    Tensor scores = gemm_strided_batched(q, k, scale, nullptr, 0.0f, false, true, execution);
    if (mask != nullptr) {
        if (!mask->defined() || mask->dtype() != DType::Bool) {
            throw std::invalid_argument("attention: mask must be a defined Bool tensor");
        }
        if (mask->device() != q.device()) {
            throw std::invalid_argument("attention: mask must be on the inputs' device");
        }
        if (broadcast_shapes(mask->shape(), scores.shape()) != scores.shape()) {
            throw std::invalid_argument("attention: mask " + to_string(mask->shape()) +
                                        " does not broadcast to the scores " +
                                        to_string(scores.shape()));
        }
        scores = masked(scores, *mask, execution);
    }
    if (causal) {
        // [Lq, Lk], true on and below the diagonal; it broadcasts over the batch dimensions.
        std::vector<std::uint8_t> keep(static_cast<std::size_t>(queries * keys));
        for (std::int64_t i = 0; i < queries; ++i) {
            for (std::int64_t j = 0; j < keys; ++j) {
                keep[static_cast<std::size_t>(i * keys + j)] = j <= i ? 1 : 0;
            }
        }
        const Tensor lower =
            Tensor::from_host(keep.data(), {queries, keys}, DType::Bool, q.device());
        scores = masked(scores, lower, execution);
    }
    const Tensor weights = softmax(scores, -1, execution);
    return gemm_strided_batched(weights, v, 1.0f, nullptr, 0.0f, false, false, execution);
}

} // namespace mensurflow
