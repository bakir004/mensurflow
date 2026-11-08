// Batched GEMM on the CPU: C[i] = alpha * A[i] * B[i] + beta * C0[i] for every matrix i of a batch,
// built on the single packed GEMM (gemm_packed.hpp). Each A[i] and B[i] is found through an offset,
// so a batch dimension that was broadcast (stride 0) or any other strided batch layout needs no
// copy, and within one matrix any row and column strides are read in place (a transposed operand is
// a swapped pair of strides).
//
// There are two ways to use several threads, and which one wins depends on the shapes:
//   over_batch     each thread takes whole matrices and multiplies them on one thread. Best when
//                  there are at least as many matrices as threads, or the matrices are small.
//   within_matrix  the matrices go one after another and each is split over all threads by the
//                  parallel packed GEMM. Best for a few large matrices.
// gemm_batched_f32_parallel picks one from the shapes.
//
// Callers (src/ops) have already validated the shapes, so nothing here checks them.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

struct BatchedGemm {
    // Matrix i of A starts at a + a_offsets[i]; its element (r, p) is at r * rsa + p * csa.
    const float* a = nullptr;
    const std::int64_t* a_offsets = nullptr;
    std::int64_t rsa = 0, csa = 0;
    // Matrix i of B starts at b + b_offsets[i]; its element (p, j) is at p * rsb + j * csb.
    const float* b = nullptr;
    const std::int64_t* b_offsets = nullptr;
    std::int64_t rsb = 0, csb = 0;
    // C is contiguous: matrix i is the row-major m x n block at c + i * m * n. It must not overlap
    // A or B. c0 has the same layout and is only read when beta != 0 (it may then equal c).
    float* c = nullptr;
    const float* c0 = nullptr;
    std::int64_t batch = 0, m = 0, n = 0, k = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
};

// One thread, one matrix after another: the benchmark baseline.
void gemm_batched_f32_sequential(const BatchedGemm& g);

// Threads split the batch; every matrix is multiplied on one thread.
void gemm_batched_f32_over_batch(const BatchedGemm& g);

// One matrix after another, each split over all threads.
void gemm_batched_f32_within_matrix(const BatchedGemm& g);

// over_batch when there are at least as many matrices as threads or the matrices are too small to
// split well, within_matrix otherwise.
void gemm_batched_f32_parallel(const BatchedGemm& g);

} // namespace mensurflow::cpu
