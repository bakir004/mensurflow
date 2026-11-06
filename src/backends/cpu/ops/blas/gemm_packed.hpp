// Single-threaded packed GEMM for float32 (the Goto/BLIS design): C is built from 6 x 16 register
// tiles, and A and B are first copied into contiguous panels so the tile kernel streams through
// memory in order. Uses AVX2 + FMA, chosen at run time; on a CPU without them (or a non-x86 build)
// it falls back to matmul_f32_sequential, so callers need no check of their own.
//
// This is the sequential building block. The threaded version and tuned block sizes come later.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// True when this CPU and OS can run the AVX2 + FMA kernel.
bool packed_gemm_supported();

// Cache-block sizes of the packed kernels: an MC x KC block of A is meant to stay in L2, a KC x NC
// block of B in L3. mc and nc are rounded down to whole tiles of the chosen shape. The defaults are BLIS's
// Haswell values (168, 256, 4080, 4 tiles per thread), except with an L2 of 512 KB or more, where
// they are (240, 1024, L3-derived up to 8160, 16). The setter exists for tuning benchmarks: it is not thread-safe
// and must not be called while a matmul is running.
struct PackedGemmBlocking {
    std::int64_t mc = 168;
    std::int64_t kc = 256;
    std::int64_t nc = 4080;
    std::int64_t tiles_per_thread = 4; // parallel version: tiles of C handed out per thread
    std::int64_t prefetch = 0;         // micro-kernel: 0 none, 1 prefetch the C tile, 2 also A and B
    std::int64_t shape = 0;            // micro-kernel tile: 0 6x16, 1 6x16 unrolled, 2 4x24, 3 12x8
    std::int64_t scalar_pack = 0;      // 1 = use the scalar packers (to measure the AVX2 ones against)
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

} // namespace mensurflow::cpu
