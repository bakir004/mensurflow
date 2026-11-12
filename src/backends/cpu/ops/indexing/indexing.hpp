// Contiguous CPU indexing kernels. Public ops validate shapes and indices before dispatch.
// The same loops run sequentially or with OpenMP through MENSURFLOW_PARALLEL_FOR.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

void one_hot_i64(const std::int64_t* indices, std::int64_t* out, std::int64_t count,
                  std::int64_t classes);

void embedding_f32(const float* weight, const std::int64_t* indices, float* out,
                   std::int64_t count, std::int64_t width);

// Stable grouping gives each output tile one writer. Repeated rows sum in input order, so the
// result does not depend on thread count. Scratch memory is O(rows + count), not per-thread tables.
void embedding_backward_f32(const float* gradient, const std::int64_t* indices, float* out,
                            std::int64_t count, std::int64_t rows, std::int64_t width);

} // namespace mensurflow::cpu
