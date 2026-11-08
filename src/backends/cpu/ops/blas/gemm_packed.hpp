// Single-threaded packed GEMM for float32 (the Goto/BLIS design): C is built from 6 x 16 register
// tiles, and A and B are first copied into contiguous panels so the tile kernel streams through
// memory in order. Uses AVX2 + FMA, chosen at run time; on a CPU without them (or a non-x86 build)
// it falls back to matmul_f32_sequential, so callers need no check of their own.
//
// matmul_f32_packed is the single-thread version; matmul_f32_packed_parallel splits the work over threads
// (several strategies, see PackedGemmBlocking::parallel).
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// True when this CPU and OS can run the AVX2 + FMA kernel.
bool packed_gemm_supported();

// Cache-block sizes of the packed kernels: an MC x KC block of A is meant to stay in L2, a KC x NC
// block of B in L3. mc and nc are rounded down to whole tiles of the chosen shape. The defaults are BLIS's
// Haswell values (168, 256, 4080, 4 tiles per thread), except with an L2 of 512 KB or more, where
// they are (240, 768, L3-derived up to 8160, 16). The setter exists for tuning benchmarks: it is not thread-safe
// and must not be called while a matmul is running.
struct PackedGemmBlocking {
    std::int64_t mc = 168;
    std::int64_t kc = 256;
    std::int64_t nc = 4080;
    std::int64_t tiles_per_thread = 4; // parallel version: tiles of C handed out per thread
    std::int64_t prefetch = 0;         // micro-kernel: 0 none, 1 prefetch the C tile, 2 also A and B
    std::int64_t shape = 0;            // micro-kernel tile: 0 6x16, 1 6x16 unrolled, 2 4x24, 3 12x8
    std::int64_t scalar_pack = 0;      // 1 = use the scalar packers (to measure the AVX2 ones against)
    // Parallel version only:
    std::int64_t parallel = 0;   // strategy: 0 two parallel loops per step (shared packed A and B),
                                 // 1 one parallel region per call (same split, fewer thread wake-ups),
                                 // 2 one region, B packed together, A packed privately by each thread
                                 // (BLIS-style rows x columns grid of threads),
                                 // 3 one region, the tiles of 0 with A packed privately per tile,
                                 // 4 one region, the next step packed while this one is multiplied
                                 // (two buffers); below 2^29 multiply-adds the team is split by L3
                                 // cache as in 5 (the default),
                                 // 5 as 4, always split by L3 cache: each chiplet computes its own
                                 // columns of C from its own copy of A and its own B panels
    std::int64_t loop_order = 0; // inside a block: 0 B panel outer / A panel inner, 1 the other way round
    std::int64_t pin = 0;        // 0 threads not pinned, 1 each thread pinned to its own physical core
                                 // (either SMT half), 2 to one hardware thread of it, 3 only a preference
                                 // for that core (the OS may still move the thread). See affinity.hpp
    std::int64_t profile = 0;    // 1 = record where the threads spend their time (packed_gemm_profile)
};
// Sizes in bytes of this CPU's level-1 data, level-2 (per core) and level-3 caches as CPUID reports
// them; 0 when unknown. The L3 is usually shared by several cores.
struct CacheSizes {
    std::int64_t l1d = 0;
    std::int64_t l2 = 0;
    std::int64_t l3 = 0;
};
CacheSizes cpu_cache_sizes();

PackedGemmBlocking packed_gemm_blocking();
void set_packed_gemm_blocking(PackedGemmBlocking blocking); // throws std::invalid_argument

// Where the threads of the last matmul_f32_packed_parallel call spent their time, recorded when
// PackedGemmBlocking::profile is 1. Times are summed over the K steps of the call. "Mean" and "max"
// are over the threads; the wall time minus the busy time is what threads spent waiting (barriers,
// uneven work, thread start-up).
struct PackedGemmProfile {
    double wall_s = 0.0;         // whole call
    double pack_mean_s = 0.0;    // busy packing A and B
    double pack_max_s = 0.0;
    double compute_mean_s = 0.0; // busy in the micro-kernel
    double compute_max_s = 0.0;
    std::int64_t threads = 0;    // threads that took part
    std::int64_t tiles = 0;      // work items handed out in the compute step(s)
};
PackedGemmProfile packed_gemm_profile();

// Row-major C[m,n] = A[m,k] * B[k,n]. C is overwritten and must not overlap A or B. k == 0 fills C
// with zeros. Products are accumulated with fused multiply-adds in increasing p, so results can
// differ from matmul_f32_sequential in the last bits.
void matmul_f32_packed(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                       std::int64_t k);

// The same product split across max_threads() threads (packing and the tiles of C are shared out),
// with the same results as matmul_f32_packed. Small products and single-thread builds call
// matmul_f32_packed directly.
void matmul_f32_packed_parallel(const float* a, const float* b, float* c, std::int64_t m,
                                std::int64_t n, std::int64_t k);

// The same with strided operands, so a transposed, sliced or padded matrix needs no copy first.
// Element (i, p) of A is a[i * rsa + p * csa], element (p, j) of B is b[p * rsb + j * csb], and element
// (i, j) of C is c[i * ldc + j]. A row-major m x k matrix has (rsa, csa) = (k, 1), its transpose
// (1, m); for a view with a leading dimension ld, (ld, 1). Packing reads the operands in place:
// unit strides along the depth or along the other dimension take vectorized paths, any other stride
// (including 0) a scalar one. On a CPU without AVX2 the operands are copied into contiguous
// temporaries and the plain loops are used. C must not overlap A or B.
void matmul_f32_packed_strided(const float* a, std::int64_t rsa, std::int64_t csa, const float* b,
                               std::int64_t rsb, std::int64_t csb, float* c, std::int64_t ldc,
                               std::int64_t m, std::int64_t n, std::int64_t k);
void matmul_f32_packed_parallel_strided(const float* a, std::int64_t rsa, std::int64_t csa,
                                        const float* b, std::int64_t rsb, std::int64_t csb, float* c,
                                        std::int64_t ldc, std::int64_t m, std::int64_t n,
                                        std::int64_t k);

} // namespace mensurflow::cpu
