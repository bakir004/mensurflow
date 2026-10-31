// Parallel, cache-blocked CPU matrix multiplication over contiguous float32 matrices.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// Row-major C[m,n] = A[m,k] * B[k,n]. C is overwritten and must not overlap A or B. k == 0 fills C
// with zeros; m == 0 or n == 0 does nothing. A negative dimension, or a null pointer where memory
// would be touched, throws invalid_argument.
//
// The matrices are processed in blocks small enough to stay in cache, and the rows of C are split
// between threads, so no two threads write the same entry. Every entry of C sums its products in
// the same order as the sequential i-k-j loop (p = 0, 1, ..., k-1), whatever the thread count, so
// results do not depend on it; they can still differ in the last bits from another compiler or
// from a build that fuses multiply-adds (MENSURFLOW_NATIVE).
void matmul_f32(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                std::int64_t k);

} // namespace mensurflow::cpu
