// CPU BLAS kernels over contiguous float32 buffers. Each is one loop marked with
// MENSURFLOW_PARALLEL_FOR, which splits it across threads in an OpenMP build. matmul alone has a
// separate sequential and parallel version, because its parallel loop is cache-blocked.
//
// Callers (src/ops) have already validated shapes, so kernels do no checking of their own: sizes
// are non-negative and pointers are valid wherever a size is non-zero. Sums are accumulated in
// double and rounded to float once at the end.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// sum(x[i] * y[i]). n == 0 gives 0.
float dot_f32(const float* x, const float* y, std::int64_t n);

// out[i] = alpha * x[i]. out may be x.
void scal_f32(float alpha, const float* x, float* out, std::int64_t n);

// sum(|x[i]|). n == 0 gives 0.
float asum_f32(const float* x, std::int64_t n);

// sqrt(sum(x[i]^2)). n == 0 gives 0.
float nrm2_f32(const float* x, std::int64_t n);

// out[i] = x[i].
void copy_f32(const float* x, float* out, std::int64_t n);

// Row-major y[m] = A[m,k] * x[k]. k == 0 fills y with zeros.
void matvec_f32(const float* a, const float* x, float* y, std::int64_t m, std::int64_t k);

// y = alpha * op(A) * x + beta * initial, where A is row-major [rows, cols] and op(A) is A or its
// transpose. initial is only read when beta != 0.
void gemv_f32(const float* a, const float* x, float* y, std::int64_t rows, std::int64_t cols,
              float alpha, float beta, const float* initial, bool transpose);

// Row-major C[m,n] = A[m,k] * B[k,n]. C is overwritten and must not overlap A or B. k == 0 fills
// C with zeros. Every C[i,j] adds its products in increasing p in both versions, so they agree.
// The parallel version also works through B in cache-sized blocks.
void matmul_f32_sequential(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                           std::int64_t k);
void matmul_f32_parallel(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                         std::int64_t k);

} // namespace mensurflow::cpu
