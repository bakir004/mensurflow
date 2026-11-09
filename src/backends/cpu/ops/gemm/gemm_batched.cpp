// Batched GEMM: a loop over the matrices of the batch around the packed single GEMM. The packed
// kernel keeps its packing buffers per thread, so several threads may run it at once.
#include "backends/cpu/ops/gemm/gemm_batched.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"

namespace mensurflow::cpu {

// C[i] = alpha * C[i] + beta * C0[i], after the product has been written into C[i]. Skipped for the
// plain product (alpha 1, beta 0), which is the common case.
static void finish_matrix(const BatchedGemm& g, std::int64_t i) {
    if (g.alpha == 1.0f && g.beta == 0.0f) {
        return;
    }
    const std::int64_t size = g.m * g.n;
    float* c = g.c + i * size;
    if (g.beta == 0.0f) {
        for (std::int64_t j = 0; j < size; ++j) {
            c[j] = g.alpha * c[j];
        }
    } else {
        const float* c0 = g.c0 + i * size;
        for (std::int64_t j = 0; j < size; ++j) {
            c[j] = g.alpha * c[j] + g.beta * c0[j];
        }
    }
}

void gemm_batched_f32_sequential(const BatchedGemm& g) {
    for (std::int64_t i = 0; i < g.batch; ++i) {
        matmul_f32_packed_strided(g.a + g.a_offsets[i], g.rsa, g.csa, g.b + g.b_offsets[i], g.rsb,
                                  g.csb, g.c + i * g.m * g.n, g.n, g.m, g.n, g.k);
        finish_matrix(g, i);
    }
}

void gemm_batched_f32_over_batch(const BatchedGemm& g) {
    [[maybe_unused]] const double work = static_cast<double>(g.batch) * static_cast<double>(g.m) *
                                         static_cast<double>(g.n) * static_cast<double>(g.k);
    // dynamic: matrices finish at different speeds when threads share a core or a cache.
    MENSURFLOW_PARALLEL_FOR(if (g.batch > 1 && work > parallel_work_threshold) schedule(dynamic))
    for (std::int64_t i = 0; i < g.batch; ++i) {
        matmul_f32_packed_strided(g.a + g.a_offsets[i], g.rsa, g.csa, g.b + g.b_offsets[i], g.rsb,
                                  g.csb, g.c + i * g.m * g.n, g.n, g.m, g.n, g.k);
        finish_matrix(g, i);
    }
}

void gemm_batched_f32_within_matrix(const BatchedGemm& g) {
    for (std::int64_t i = 0; i < g.batch; ++i) {
        matmul_f32_packed_parallel_strided(g.a + g.a_offsets[i], g.rsa, g.csa, g.b + g.b_offsets[i],
                                           g.rsb, g.csb, g.c + i * g.m * g.n, g.n, g.m, g.n, g.k);
        finish_matrix(g, i);
    }
}

void gemm_batched_f32_parallel(const BatchedGemm& g) {
    // A matrix below about 64 x 64 x 64 multiply-adds is too small for several threads to share,
    // and with at least one matrix per thread splitting the batch keeps every thread busy without
    // any synchronization inside a matrix.
    constexpr double small_matrix = 64.0 * 64.0 * 64.0;
    const double per_matrix =
        static_cast<double>(g.m) * static_cast<double>(g.n) * static_cast<double>(g.k);
    if (g.batch >= max_threads() || per_matrix < small_matrix) {
        gemm_batched_f32_over_batch(g);
    } else {
        gemm_batched_f32_within_matrix(g);
    }
}

} // namespace mensurflow::cpu
