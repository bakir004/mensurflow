// Scaled dot-product attention, written as three existing ops: a batched GEMM for the scores, a
// masked softmax, and a batched GEMM for the weighted sum. The last argument says where the op
// runs (see execution.hpp). Only float32, and only on the CPU so far (it inherits that from
// gemm_strided_batched). Not differentiable yet: gemm_strided_batched has no backward.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow {

// softmax(q * k^T * scale + mask) * v. q is [..., Lq, D], k is [..., Lk, D] and v is [..., Lk, Dv];
// the leading (batch, head) dimensions broadcast like gemm_strided_batched, and the result is
// [..., Lq, Dv]. scale = 0 means 1 / sqrt(D). mask is optional: a Bool tensor that broadcasts to
// the scores [..., Lq, Lk], where true keeps a position and false removes it (like PyTorch's
// boolean attn_mask). causal = true also removes every key after the query's own position: query
// i sees keys 0..i (top-left aligned, as PyTorch's is_causal). A query whose keys are all removed
// gets zeros, not NaN. The full [..., Lq, Lk] score matrix is materialized: this is the naive
// algorithm, not a fused one. Throws invalid_argument for wrong ranks, mismatched D or Lk, a
// non-Bool mask, or mixed dtypes or devices.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask = nullptr,
                 bool causal = false, float scale = 0.0f, Execution execution = Execution::Auto);

} // namespace mensurflow
