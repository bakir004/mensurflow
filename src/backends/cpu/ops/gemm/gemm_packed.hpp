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
    std::int64_t prefetch = 0;         // micro-kernel (6x16 shape 0): 0 none, 1 the C tile, 2 C and
                                       // A/B 8 steps ahead, 3 B 32 steps ahead (L1), 4 B 128 steps
                                       // ahead (L2), 5 the C tile with write intent, 6 = 5 + 3
    std::int64_t prefetch_block = 0;   // between tiles: 0 none, 1 the next C tile, 2 the same with
                                       // write intent, 3 the next A panel spread over the row of
                                       // tiles (loop order 1), 4 = 2 + 3
    std::int64_t shape = 0;            // micro-kernel tile: 0 6x16, 1 6x16 unrolled, 2 4x24, 3 12x8;
                                       // 6x16 loop variants: 4/5 unrolled x2/x4, 6 broadcasts
                                       // pipelined a row ahead, 7 pipelined x2, 8/9 unrolled x2/x4
                                       // with a compiler fence, 10/11 pipelined x2/x4 with the fence
                                       // (docs/MATMUL_OPTIMIZATION.md section 20)
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
    std::int64_t pool = 0;       // strategies 4 and 5: 0 run the team in an OpenMP region, 1 on the
                                 // library's own thread pool (thread_pool.hpp)
    std::int64_t private_b = 0;  // strategies 4 and 5: 0 B is packed into the shared KC x NC buffer,
                                 // 1 each thread packs the few B panels of its tile into its own
                                 // L2-sized buffer right before multiplying them (only A is shared),
                                 // 2 choose by the shape: 1 for products with few rows, where the
                                 // shared buffer makes packing a memory-bandwidth problem
    std::int64_t b_chunk = 0;    // private_b: B panels per tile (0 = as many as fit in half the L2)
    std::int64_t static_items = 0; // strategies 4 and 5: 0 work items are handed out first come first
                                 // served, 1 each thread owns a fixed run of them and takes from the
                                 // others only when it is done (the same thread then reads the same
                                 // columns of B and writes the same part of C on every call)
    std::int64_t private_a = 0;  // private B, one depth step and a small A: 1 = each thread also packs
                                 // its own copy of A, and the team runs without any barrier
    std::int64_t fine_rows = 0;  // shared B: 0 a tile is MC rows (the last block what is left), 1 the
                                 // rows are split into equal blocks, and into more of them when
                                 // there are too few B panels for tiles_per_thread tiles per thread
    std::int64_t split_rows = 0; // chiplet split with few B panels: 1 = the groups divide the rows of C
                                 // (each packs its own rows of A once) instead of the columns
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

// For benchmarks of the micro-kernel alone: the tile (mr x nr) of micro-kernel `shape` (see
// PackedGemmBlocking::shape), and one call of it on packed panels pa (kc steps of mr floats) and pb
// (kc steps of nr floats, 32-byte aligned), writing (or adding to) C[mr, nr] with row stride ldc.
// Both return false, doing nothing, for an unknown shape or a CPU without AVX2.
bool packed_microkernel_tile(int shape, std::int64_t& mr, std::int64_t& nr);
bool run_packed_microkernel(int shape, std::int64_t kc, const float* pa, const float* pb, float* c,
                            std::int64_t ldc, bool accumulate);

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
