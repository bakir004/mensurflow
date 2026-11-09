// Packed GEMM, five loops around a 6 x 16 AVX2 micro-kernel (BLIS's Haswell design):
//
//   jc: NC columns of B/C        pack B[kc, nc] into 16-column panels   (sized for L3)
//   pc: KC steps of the depth      (C is stored at the first, added to at the rest)
//   ic: MC rows of A/C           pack A[mc, kc] into 6-row panels       (sized for L2)
//   jr: one 16-column panel of B                                        (panel stays in L1)
//   ir: one 6-row panel of A     micro-kernel: C[6,16] += Apanel * Bpanel
//
// Packed layouts, both zero-padded to full panels so the kernel never branches on edges:
//   A panel: for each p, 6 floats  (A[i0..i0+5][p])
//   B panel: for each p, 16 floats (B[p][j0..j0+15])
#include "backends/cpu/ops/gemm/gemm_packed.hpp"

#include "backends/cpu/affinity.hpp"
#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/ops/gemm/gemm_pack.hpp"
#include "backends/cpu/thread_pool.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define MENSURFLOW_PACKED_GEMM 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace mensurflow::cpu {

// Packing: the AVX2 packers, or the scalar reference when blocking.scalar_pack is set.
void pack_a(bool scalar, std::int64_t mr, const float* a, std::int64_t rsa, std::int64_t csa,
            std::int64_t mc, std::int64_t kc, float* dst) {
    if (scalar) {
        detail::pack_a_scalar(mr, a, rsa, csa, mc, kc, dst);
    } else {
        detail::pack_a(mr, a, rsa, csa, mc, kc, dst);
    }
}

void pack_b(bool scalar, std::int64_t nr, const float* b, std::int64_t rsb, std::int64_t csb,
            std::int64_t kc, std::int64_t nc, float* dst, std::int64_t run = detail::pack_b_run,
            std::int64_t ahead = 0) {
    if (scalar) {
        detail::pack_b_scalar(nr, b, rsb, csb, kc, nc, dst);
    } else {
        detail::pack_b(nr, b, rsb, csb, kc, nc, dst, run, ahead);
    }
}

// The plain loops (matmul_f32_sequential / parallel) read contiguous row-major data. For a product the
// packed kernel cannot run (no AVX2, or a non-x86 build) with strided operands, the operands are
// copied into contiguous temporaries first and the result copied back.
void matmul_strided_fallback(const float* a, std::int64_t rsa, std::int64_t csa, const float* b,
                             std::int64_t rsb, std::int64_t csb, float* c, std::int64_t ldc,
                             std::int64_t m, std::int64_t n, std::int64_t k, bool parallel) {
    const auto run = [&](const float* x, const float* y, float* z) {
        if (parallel) {
            matmul_f32_parallel(x, y, z, m, n, k);
        } else {
            matmul_f32_sequential(x, y, z, m, n, k);
        }
    };
    if (rsa == k && csa == 1 && rsb == n && csb == 1 && ldc == n) {
        run(a, b, c);
        return;
    }
    std::vector<float> a_copy(static_cast<std::size_t>(m * k));
    std::vector<float> b_copy(static_cast<std::size_t>(k * n));
    std::vector<float> c_copy(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t p = 0; p < k; ++p) {
            a_copy[static_cast<std::size_t>(i * k + p)] = a[i * rsa + p * csa];
        }
    }
    for (std::int64_t p = 0; p < k; ++p) {
        for (std::int64_t j = 0; j < n; ++j) {
            b_copy[static_cast<std::size_t>(p * n + j)] = b[p * rsb + j * csb];
        }
    }
    run(a_copy.data(), b_copy.data(), c_copy.data());
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            c[i * ldc + j] = c_copy[static_cast<std::size_t>(i * n + j)];
        }
    }
}

#ifdef MENSURFLOW_PACKED_GEMM

namespace {

constexpr std::int64_t MR6 = 6;
constexpr std::int64_t NR6 = 16;

// Multiply-adds a thread needs before it is worth waking (see matmul_f32_packed_parallel).
constexpr double min_macs_per_thread = 512.0 * 1024.0;

// Pinning costs a few system calls per thread, so it is only done for products with at least this many
// multiply-adds (about n = 200).
constexpr double min_macs_to_pin = 4.0 * 1024.0 * 1024.0;

// Products with fewer multiply-adds than this use at most one thread per physical core (about n = 1024).
constexpr double min_macs_for_smt = 1024.0 * 1024.0 * 1024.0;
// PackedGemmBlocking::private_b == 2: products with at most this many rows pack B privately.
constexpr std::int64_t private_b_max_rows = 512;
// Private B packs whole runs of panels row by row; the rows are n floats apart, which the hardware
// prefetcher does not follow, so the packer prefetches this many rows ahead (measured 0 to 12:
// m = 32, n = k = 4096 on 12 threads 368, 396, 418, 419, 435, 437 GFLOP/s for 0, 2, 4, 6, 8, 12).
constexpr std::int64_t private_b_pack_ahead = 8;

// Strategy 4 splits the team by L3 cache (as strategy 5 always does) for products below this many
// multiply-adds (about n = 810). Measured on a Ryzen 9 5900X (two chiplets), 12 threads: +11% at
// n = 256, +18-23% at n = 384-768; at n = 1024 the split was erratic, so larger products share.
constexpr double max_macs_for_groups = 512.0 * 1024.0 * 1024.0;

// How many depth steps ahead the optional in-loop prefetch looks (one B line per step).
constexpr std::int64_t prefetch_steps = 8;

// Size in bytes of the level-1 data, level-2 or level-3 cache from CPUID (Intel leaf 4, AMD leaf
// 0x8000001D), or 0 when
// it cannot be read.
std::int64_t cache_bytes(int wanted_level) {
    int regs[4] = {0, 0, 0, 0};
    const auto cpuid = [&regs](int leaf, int sub) {
#if defined(_MSC_VER)
        __cpuidex(regs, leaf, sub);
#else
        unsigned a = 0, b = 0, c = 0, d = 0;
        __cpuid_count(leaf, sub, a, b, c, d);
        regs[0] = static_cast<int>(a);
        regs[1] = static_cast<int>(b);
        regs[2] = static_cast<int>(c);
        regs[3] = static_cast<int>(d);
#endif
    };
    cpuid(0, 0);
    char vendor[13];
    std::memcpy(vendor + 0, &regs[1], 4);
    std::memcpy(vendor + 4, &regs[3], 4);
    std::memcpy(vendor + 8, &regs[2], 4);
    vendor[12] = '\0';
    int leaf = 0;
    if (std::strcmp(vendor, "GenuineIntel") == 0) {
        leaf = 4;
    } else if (std::strcmp(vendor, "AuthenticAMD") == 0) {
        cpuid(static_cast<int>(0x80000000u), 0);
        if (static_cast<unsigned>(regs[0]) >= 0x8000001Du) {
            leaf = 0x8000001D;
        }
    }
    if (leaf == 0) {
        return 0;
    }
    for (int sub = 0; sub < 16; ++sub) {
        cpuid(leaf, sub);
        const int type = regs[0] & 0x1f; // 0 = no more caches, 1 = data, 2 = instruction, 3 = unified
        const int level = (regs[0] >> 5) & 0x7;
        if (type == 0) {
            break;
        }
        const bool kind_matches = wanted_level == 1 ? type == 1 : (type == 1 || type == 3);
        if (level == wanted_level && kind_matches) {
            const std::int64_t ways = (static_cast<unsigned>(regs[1]) >> 22) + 1;
            const std::int64_t partitions = ((static_cast<unsigned>(regs[1]) >> 12) & 0x3ff) + 1;
            const std::int64_t line = (static_cast<unsigned>(regs[1]) & 0xfff) + 1;
            const std::int64_t sets = static_cast<std::int64_t>(static_cast<unsigned>(regs[2])) + 1;
            return ways * partitions * line * sets;
        }
    }
    return 0;
}

// Default block sizes, chosen from the cache sizes.
//   L2 of 512 KB or more (measured on a Ryzen 9 5900X, 512 KB L2, 32 MB L3): KC = 1024, MC = 240,
//   NC = L3 / (KC * 4 bytes) rounded down to a multiple of 16 and clamped to 4080..8160, 16 tiles per
//   thread. A sweep of MC 120-720, KC 256-2048, NC 1024-8160 and 1-64 tiles per thread found this
//   +3% to +30% faster than BLIS's Haswell values at 8 and 24 threads (n = 2048, 4096). KC from 1024 to
//   2048 and 16-64 tiles per thread were equal within noise; single-threaded, every setting was within
//   2%. A deep KC means fewer pack-and-barrier steps and fewer passes over C; it works although the
//   240 x 1024 block of A (960 KB) is larger than the L2, because the panels are read in order and
//   the hardware prefetcher keeps up.
//   Smaller or unknown L2: BLIS's Haswell values (168, 256, 4080, 4), which fit a 256 KB L2 and have
//   not been re-measured on such a CPU.
//
// Loop order inside a block: the operand that stays in the L1 cache while the other streams past it
// should be the one that fits. A packed B panel is KC x 16 floats; if that is larger than the L1
// (KC = 1024 is 64 KB against 32 KB) the A panel (KC x 6 floats = 24 KB) is held instead: loop order 1.
// Measured on the Ryzen: +4-7% at n = 2048 and 25% less loss on calls that follow other work.
//
// Parallel strategy 1 (see PackedGemmBlocking::parallel): same speed as strategy 0, one team wake-up
// per call instead of two per K step, and the threads pin themselves when the team starts.
//
// Pinning: one thread per physical core removes a variable 20-40% slowdown (the OS sometimes puts
// two of our threads on the two halves of one core). It is on when the topology can be read.
PackedGemmBlocking initial_blocking() {
    PackedGemmBlocking blocking; // 168, 256, 4080, 4
    if (cache_bytes(2) >= 512 * 1024) {
        blocking.mc = 240;
        blocking.kc = 1024;
        blocking.tiles_per_thread = 16;
        const std::int64_t l3 = cache_bytes(3);
        if (l3 > 0) {
            const std::int64_t nc = l3 / (blocking.kc * 4) / 16 * 16;
            blocking.nc = std::min<std::int64_t>(std::max<std::int64_t>(nc, 4080), 8160);
        }
    }
    const std::int64_t l1d = cache_bytes(1);
    blocking.loop_order = (l1d > 0 && blocking.kc * 16 * 4 > l1d) ? 1 : 0;
    blocking.pin = physical_core_count() > 0 ? 1 : 0;
    // Strategy 4: one parallel region, the next step packed while this one is multiplied (double
    // buffered). With it, KC = 768 instead of 1024: same speed when calls come back to back, and it
    // avoids most of a slowdown of calls after idle time that KC = 1024 shows at n = 2048 (see
    // docs/MATMUL_OPTIMIZATION.md, section 16). Measured +3-12% over strategy 1 at n = 1536-4096.
    blocking.parallel = 4;
    if (blocking.kc == 1024) {
        blocking.kc = 768;
    }
    // Micro-kernel prefetch 5: the C tile with write intent (PREFETCHW) before the depth loop, so the
    // C update after it does not wait for the lines (which another core may own). Measured 0 to +3.5%
    // (best at n = 1024 on 12 threads), never more than 0.7% behind (section 21).
    blocking.prefetch = 5;
    // The team runs on the library's own thread pool, whose idle workers spin for a moment instead
    // of sleeping at once: +17% at n = 512, +3% at n = 1024, and up to 2x on small rectangular
    // products (section 22).
    blocking.pool = 1;
    // Products with few rows (a batch times a wide weight matrix) pack B privately per thread: up to
    // 2.3x where the shared packed B made the call wait for memory (section 23).
    blocking.private_b = 2;
    return blocking;
}

PackedGemmBlocking& blocking_state() {
    static PackedGemmBlocking state = initial_blocking();
    return state;
}


// Prefetch with intent to write (PREFETCHW): the line arrives in a state that can be written without a
// second request. Every AVX2 CPU has the instruction.
#if defined(_MSC_VER) && !defined(__clang__)
#define MENSURFLOW_PREFETCH_WRITE(p) _m_prefetchw(p)
#else
#define MENSURFLOW_PREFETCH_WRITE(p) __builtin_prefetch((p), 1, 3)
#endif

// Prefetches the 6 rows of a 6 x 16 tile of C (each row is 64 bytes, so it can touch 2 lines).
template <bool Write>
inline void prefetch_c_tile(const float* c, std::int64_t ldc) {
    for (int r = 0; r < 6; ++r) {
        if constexpr (Write) {
            MENSURFLOW_PREFETCH_WRITE(c + r * ldc);
            MENSURFLOW_PREFETCH_WRITE(c + r * ldc + 15);
        } else {
            _mm_prefetch(reinterpret_cast<const char*>(c + r * ldc), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(c + r * ldc + 15), _MM_HINT_T0);
        }
    }
}

// C[6,16] (row stride ldc) = or += sum over p < kc of pa[p*6 + r] * pb[p*16 + j].
// Prefetch (one prefetch per depth step covers one 64-byte B row):
//   0 none
//   1 the C tile before the loop (it is read and written after it)
//   2 1, and the packed A and B panels 8 steps ahead (T0)
//   3 B 32 steps (2 KB) ahead into L1 (T0)
//   4 B 128 steps (8 KB) ahead into L2 (T1): for B panels streaming from L3
//   5 the C tile with write intent (PREFETCHW) before the loop
//   6 5 and 3 together
template <int Prefetch>
MENSURFLOW_TARGET_AVX2 void kernel_6x16(std::int64_t kc, const float* pa, const float* pb,
                                        float* c, std::int64_t ldc, bool accumulate) {
    if constexpr (Prefetch == 1 || Prefetch == 2) {
        prefetch_c_tile<false>(c, ldc);
    }
    if constexpr (Prefetch == 5 || Prefetch == 6) {
        prefetch_c_tile<true>(c, ldc);
    }
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (std::int64_t p = 0; p < kc; ++p) {
        if constexpr (Prefetch == 2) {
            _mm_prefetch(reinterpret_cast<const char*>(pb + NR6 * prefetch_steps), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pa + MR6 * prefetch_steps), _MM_HINT_T0);
        }
        if constexpr (Prefetch == 3 || Prefetch == 6) {
            _mm_prefetch(reinterpret_cast<const char*>(pb + NR6 * 32), _MM_HINT_T0);
        }
        if constexpr (Prefetch == 4) {
            _mm_prefetch(reinterpret_cast<const char*>(pb + NR6 * 128), _MM_HINT_T1);
        }
        const __m256 b0 = _mm256_load_ps(pb);
        const __m256 b1 = _mm256_load_ps(pb + 8);
        __m256 a = _mm256_broadcast_ss(pa + 0);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(pa + 1);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(pa + 2);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(pa + 3);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(pa + 4);
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(pa + 5);
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);
        pa += MR6;
        pb += NR6;
    }

    auto store_row = [&](float* row, __m256 lo, __m256 hi) {
        if (accumulate) {
            lo = _mm256_add_ps(lo, _mm256_loadu_ps(row));
            hi = _mm256_add_ps(hi, _mm256_loadu_ps(row + 8));
        }
        _mm256_storeu_ps(row, lo);
        _mm256_storeu_ps(row + 8, hi);
    };
    store_row(c + 0 * ldc, c00, c01);
    store_row(c + 1 * ldc, c10, c11);
    store_row(c + 2 * ldc, c20, c21);
    store_row(c + 3 * ldc, c30, c31);
    store_row(c + 4 * ldc, c40, c41);
    store_row(c + 5 * ldc, c50, c51);
}

// ---- Other micro-kernel shapes (same contract as kernel_6x16) -----------------------------------
// An AVX2 core has 16 vector registers. A tile of MR x NR floats needs MR * NR / 8 accumulators plus
// NR / 8 registers for the B row and one for the broadcast A value. Per depth step the kernel does
// MR * NR / 8 FMAs and MR + NR / 8 loads, so a bigger and squarer tile means fewer loads per FMA:
//   6 x 16: 12 accumulators + 2 + 1 = 15 registers, 12 FMAs, 8 loads
//   4 x 24: 12 accumulators + 3 + 1 = 16 registers, 12 FMAs, 7 loads
//  12 x  8: 12 accumulators + 1 + 1 = 14 registers, 12 FMAs, 13 loads

#define MENSURFLOW_STEP_6X16(i)                                                                  \
    {                                                                                            \
        const float* a_ = pa + 6 * (i);                                                          \
        const float* b_ = pb + 16 * (i);                                                         \
        const __m256 b0 = _mm256_load_ps(b_);                                                    \
        const __m256 b1 = _mm256_load_ps(b_ + 8);                                                \
        __m256 a = _mm256_broadcast_ss(a_ + 0);                                                  \
        c00 = _mm256_fmadd_ps(a, b0, c00);                                                       \
        c01 = _mm256_fmadd_ps(a, b1, c01);                                                       \
        a = _mm256_broadcast_ss(a_ + 1);                                                         \
        c10 = _mm256_fmadd_ps(a, b0, c10);                                                       \
        c11 = _mm256_fmadd_ps(a, b1, c11);                                                       \
        a = _mm256_broadcast_ss(a_ + 2);                                                         \
        c20 = _mm256_fmadd_ps(a, b0, c20);                                                       \
        c21 = _mm256_fmadd_ps(a, b1, c21);                                                       \
        a = _mm256_broadcast_ss(a_ + 3);                                                         \
        c30 = _mm256_fmadd_ps(a, b0, c30);                                                       \
        c31 = _mm256_fmadd_ps(a, b1, c31);                                                       \
        a = _mm256_broadcast_ss(a_ + 4);                                                         \
        c40 = _mm256_fmadd_ps(a, b0, c40);                                                       \
        c41 = _mm256_fmadd_ps(a, b1, c41);                                                       \
        a = _mm256_broadcast_ss(a_ + 5);                                                         \
        c50 = _mm256_fmadd_ps(a, b0, c50);                                                       \
        c51 = _mm256_fmadd_ps(a, b1, c51);                                                       \
    }

// 6 x 16 with the depth loop unrolled four times (fewer loop-control instructions).
MENSURFLOW_TARGET_AVX2 void kernel_6x16_unrolled(std::int64_t kc, const float* pa, const float* pb,
                                                 float* c, std::int64_t ldc, bool accumulate) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    std::int64_t p = 0;
    for (; p + 4 <= kc; p += 4) {
        MENSURFLOW_STEP_6X16(0)
        MENSURFLOW_STEP_6X16(1)
        MENSURFLOW_STEP_6X16(2)
        MENSURFLOW_STEP_6X16(3)
        pa += 24;
        pb += 64;
    }
    for (; p < kc; ++p) {
        MENSURFLOW_STEP_6X16(0)
        pa += 6;
        pb += 16;
    }
    auto store_row = [&](float* row, __m256 lo, __m256 hi) {
        if (accumulate) {
            lo = _mm256_add_ps(lo, _mm256_loadu_ps(row));
            hi = _mm256_add_ps(hi, _mm256_loadu_ps(row + 8));
        }
        _mm256_storeu_ps(row, lo);
        _mm256_storeu_ps(row + 8, hi);
    };
    store_row(c + 0 * ldc, c00, c01);
    store_row(c + 1 * ldc, c10, c11);
    store_row(c + 2 * ldc, c20, c21);
    store_row(c + 3 * ldc, c30, c31);
    store_row(c + 4 * ldc, c40, c41);
    store_row(c + 5 * ldc, c50, c51);
}

// 4 x 24: C[4,24] = or += sum over p < kc of pa[p*4 + r] * pb[p*24 + j].
MENSURFLOW_TARGET_AVX2 void kernel_4x24(std::int64_t kc, const float* pa, const float* pb, float* c,
                                        std::int64_t ldc, bool accumulate) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();
    for (std::int64_t p = 0; p < kc; ++p) {
        const __m256 b0 = _mm256_load_ps(pb);
        const __m256 b1 = _mm256_load_ps(pb + 8);
        const __m256 b2 = _mm256_load_ps(pb + 16);
        __m256 a = _mm256_broadcast_ss(pa + 0);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);
        c02 = _mm256_fmadd_ps(a, b2, c02);
        a = _mm256_broadcast_ss(pa + 1);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);
        c12 = _mm256_fmadd_ps(a, b2, c12);
        a = _mm256_broadcast_ss(pa + 2);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);
        c22 = _mm256_fmadd_ps(a, b2, c22);
        a = _mm256_broadcast_ss(pa + 3);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);
        c32 = _mm256_fmadd_ps(a, b2, c32);
        pa += 4;
        pb += 24;
    }
    auto store_row = [&](float* row, __m256 v0, __m256 v1, __m256 v2) {
        if (accumulate) {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(row));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(row + 8));
            v2 = _mm256_add_ps(v2, _mm256_loadu_ps(row + 16));
        }
        _mm256_storeu_ps(row, v0);
        _mm256_storeu_ps(row + 8, v1);
        _mm256_storeu_ps(row + 16, v2);
    };
    store_row(c + 0 * ldc, c00, c01, c02);
    store_row(c + 1 * ldc, c10, c11, c12);
    store_row(c + 2 * ldc, c20, c21, c22);
    store_row(c + 3 * ldc, c30, c31, c32);
}

#define MENSURFLOW_FMA_ROW(r)                                                                    \
    a = _mm256_broadcast_ss(pa + (r));                                                           \
    c##r = _mm256_fmadd_ps(a, b, c##r);

// 12 x 8: C[12,8] = or += sum over p < kc of pa[p*12 + r] * pb[p*8 + j].
MENSURFLOW_TARGET_AVX2 void kernel_12x8(std::int64_t kc, const float* pa, const float* pb, float* c,
                                        std::int64_t ldc, bool accumulate) {
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps(), c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps(), c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps(), c8 = _mm256_setzero_ps();
    __m256 c9 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    for (std::int64_t p = 0; p < kc; ++p) {
        const __m256 b = _mm256_load_ps(pb);
        __m256 a;
        MENSURFLOW_FMA_ROW(0)
        MENSURFLOW_FMA_ROW(1)
        MENSURFLOW_FMA_ROW(2)
        MENSURFLOW_FMA_ROW(3)
        MENSURFLOW_FMA_ROW(4)
        MENSURFLOW_FMA_ROW(5)
        MENSURFLOW_FMA_ROW(6)
        MENSURFLOW_FMA_ROW(7)
        MENSURFLOW_FMA_ROW(8)
        MENSURFLOW_FMA_ROW(9)
        MENSURFLOW_FMA_ROW(10)
        MENSURFLOW_FMA_ROW(11)
        pa += 12;
        pb += 8;
    }
    auto store_row = [&](float* row, __m256 v) {
        if (accumulate) {
            v = _mm256_add_ps(v, _mm256_loadu_ps(row));
        }
        _mm256_storeu_ps(row, v);
    };
    store_row(c + 0 * ldc, c0);
    store_row(c + 1 * ldc, c1);
    store_row(c + 2 * ldc, c2);
    store_row(c + 3 * ldc, c3);
    store_row(c + 4 * ldc, c4);
    store_row(c + 5 * ldc, c5);
    store_row(c + 6 * ldc, c6);
    store_row(c + 7 * ldc, c7);
    store_row(c + 8 * ldc, c8);
    store_row(c + 9 * ldc, c9);
    store_row(c + 10 * ldc, c10);
    store_row(c + 11 * ldc, c11);
}

// ---- 6 x 16 variants: unrolling and where the broadcasts go (shapes 4-7) ------------------------
// All keep the 12 accumulators of kernel_6x16 in registers. They differ in how the depth loop is
// written; docs/MATMUL_OPTIMIZATION.md section 20 has the assembly and the measurements.

// A compiler-only fence (no instruction): memory accesses are not moved across it. Without it MSVC
// hoists the loads and broadcasts of every unrolled step to the top of the loop body, which needs
// more than the 16 ymm registers and spills accumulators to the stack inside the loop (seen in the
// disassembly; it is why the old unrolled kernel, shape 1, is 25% slower).
#if defined(_MSC_VER) && !defined(__clang__)
#define MENSURFLOW_STEP_FENCE() _ReadWriteBarrier()
#else
#define MENSURFLOW_STEP_FENCE() __asm__ __volatile__("" ::: "memory")
#endif

// Stores the 6 x 16 accumulators (row stride ldc), adding the old C when accumulating.
#define MENSURFLOW_STORE_6X16()                                                                  \
    {                                                                                            \
        __m256* acc[6][2] = {{&c00, &c01}, {&c10, &c11}, {&c20, &c21},                           \
                             {&c30, &c31}, {&c40, &c41}, {&c50, &c51}};                          \
        for (int r = 0; r < 6; ++r) {                                                            \
            float* row = c + r * ldc;                                                            \
            __m256 lo = *acc[r][0];                                                              \
            __m256 hi = *acc[r][1];                                                              \
            if (accumulate) {                                                                    \
                lo = _mm256_add_ps(lo, _mm256_loadu_ps(row));                                    \
                hi = _mm256_add_ps(hi, _mm256_loadu_ps(row + 8));                                \
            }                                                                                    \
            _mm256_storeu_ps(row, lo);                                                           \
            _mm256_storeu_ps(row + 8, hi);                                                       \
        }                                                                                        \
    }

#define MENSURFLOW_ZERO_6X16()                                                                   \
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();                                 \
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();                                 \
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();                                 \
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();                                 \
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();                                 \
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

// One depth step at offset `s` steps from pa / pb, reusing the three temporaries b0, b1, a (declared
// once outside the loop, so the compiler sees one live range per register, not one per step).
#define MENSURFLOW_STEP_REUSE(s)                                                                 \
    b0 = _mm256_load_ps(pb + 16 * (s));                                                          \
    b1 = _mm256_load_ps(pb + 16 * (s) + 8);                                                      \
    a = _mm256_broadcast_ss(pa + 6 * (s) + 0);                                                   \
    c00 = _mm256_fmadd_ps(a, b0, c00);                                                           \
    c01 = _mm256_fmadd_ps(a, b1, c01);                                                           \
    a = _mm256_broadcast_ss(pa + 6 * (s) + 1);                                                   \
    c10 = _mm256_fmadd_ps(a, b0, c10);                                                           \
    c11 = _mm256_fmadd_ps(a, b1, c11);                                                           \
    a = _mm256_broadcast_ss(pa + 6 * (s) + 2);                                                   \
    c20 = _mm256_fmadd_ps(a, b0, c20);                                                           \
    c21 = _mm256_fmadd_ps(a, b1, c21);                                                           \
    a = _mm256_broadcast_ss(pa + 6 * (s) + 3);                                                   \
    c30 = _mm256_fmadd_ps(a, b0, c30);                                                           \
    c31 = _mm256_fmadd_ps(a, b1, c31);                                                           \
    a = _mm256_broadcast_ss(pa + 6 * (s) + 4);                                                   \
    c40 = _mm256_fmadd_ps(a, b0, c40);                                                           \
    c41 = _mm256_fmadd_ps(a, b1, c41);                                                           \
    a = _mm256_broadcast_ss(pa + 6 * (s) + 5);                                                   \
    c50 = _mm256_fmadd_ps(a, b0, c50);                                                           \
    c51 = _mm256_fmadd_ps(a, b1, c51);

// The depth loop unrolled U times, one pointer update per U steps. Fence: a compiler fence between
// the steps. Shapes 4 (U 2) and 5 (U 4) without the fence, 8 and 9 with it.
template <int U, bool Fence>
MENSURFLOW_TARGET_AVX2 void kernel_6x16_reuse(std::int64_t kc, const float* pa, const float* pb,
                                              float* c, std::int64_t ldc, bool accumulate) {
    MENSURFLOW_ZERO_6X16()
    __m256 b0, b1, a;
    std::int64_t p = 0;
    for (; p + U <= kc; p += U) {
        MENSURFLOW_STEP_REUSE(0)
        if constexpr (Fence) {
            MENSURFLOW_STEP_FENCE();
        }
        MENSURFLOW_STEP_REUSE(1)
        if constexpr (U == 4) {
            if constexpr (Fence) {
                MENSURFLOW_STEP_FENCE();
            }
            MENSURFLOW_STEP_REUSE(2)
            if constexpr (Fence) {
                MENSURFLOW_STEP_FENCE();
            }
            MENSURFLOW_STEP_REUSE(3)
        }
        pa += 6 * U;
        pb += 16 * U;
    }
    for (; p < kc; ++p) {
        MENSURFLOW_STEP_REUSE(0)
        pa += 6;
        pb += 16;
    }
    MENSURFLOW_STORE_6X16()
}

// One depth step with the broadcasts pipelined one row ahead: the 16th register holds the A value of
// the next row while the current row's two FMAs run, so a broadcast never waits right before its
// use. On entry x0 holds A[row 0] of this step; on exit it holds A[row 0] of step s + 1, which is why
// the last step of a panel is done by MENSURFLOW_STEP_PIPE_LAST (it must not read past the panel).
#define MENSURFLOW_STEP_PIPE(s)                                                                  \
    b0 = _mm256_load_ps(pb + 16 * (s));                                                          \
    b1 = _mm256_load_ps(pb + 16 * (s) + 8);                                                      \
    x1 = _mm256_broadcast_ss(pa + 6 * (s) + 1);                                                  \
    c00 = _mm256_fmadd_ps(x0, b0, c00);                                                          \
    c01 = _mm256_fmadd_ps(x0, b1, c01);                                                          \
    x0 = _mm256_broadcast_ss(pa + 6 * (s) + 2);                                                  \
    c10 = _mm256_fmadd_ps(x1, b0, c10);                                                          \
    c11 = _mm256_fmadd_ps(x1, b1, c11);                                                          \
    x1 = _mm256_broadcast_ss(pa + 6 * (s) + 3);                                                  \
    c20 = _mm256_fmadd_ps(x0, b0, c20);                                                          \
    c21 = _mm256_fmadd_ps(x0, b1, c21);                                                          \
    x0 = _mm256_broadcast_ss(pa + 6 * (s) + 4);                                                  \
    c30 = _mm256_fmadd_ps(x1, b0, c30);                                                          \
    c31 = _mm256_fmadd_ps(x1, b1, c31);                                                          \
    x1 = _mm256_broadcast_ss(pa + 6 * (s) + 5);                                                  \
    c40 = _mm256_fmadd_ps(x0, b0, c40);                                                          \
    c41 = _mm256_fmadd_ps(x0, b1, c41);                                                          \
    x0 = _mm256_broadcast_ss(pa + 6 * (s) + 6);                                                  \
    c50 = _mm256_fmadd_ps(x1, b0, c50);                                                          \
    c51 = _mm256_fmadd_ps(x1, b1, c51);

#define MENSURFLOW_STEP_PIPE_LAST()                                                              \
    b0 = _mm256_load_ps(pb);                                                                     \
    b1 = _mm256_load_ps(pb + 8);                                                                 \
    x1 = _mm256_broadcast_ss(pa + 1);                                                            \
    c00 = _mm256_fmadd_ps(x0, b0, c00);                                                          \
    c01 = _mm256_fmadd_ps(x0, b1, c01);                                                          \
    x0 = _mm256_broadcast_ss(pa + 2);                                                            \
    c10 = _mm256_fmadd_ps(x1, b0, c10);                                                          \
    c11 = _mm256_fmadd_ps(x1, b1, c11);                                                          \
    x1 = _mm256_broadcast_ss(pa + 3);                                                            \
    c20 = _mm256_fmadd_ps(x0, b0, c20);                                                          \
    c21 = _mm256_fmadd_ps(x0, b1, c21);                                                          \
    x0 = _mm256_broadcast_ss(pa + 4);                                                            \
    c30 = _mm256_fmadd_ps(x1, b0, c30);                                                          \
    c31 = _mm256_fmadd_ps(x1, b1, c31);                                                          \
    x1 = _mm256_broadcast_ss(pa + 5);                                                            \
    c40 = _mm256_fmadd_ps(x0, b0, c40);                                                          \
    c41 = _mm256_fmadd_ps(x0, b1, c41);                                                          \
    c50 = _mm256_fmadd_ps(x1, b0, c50);                                                          \
    c51 = _mm256_fmadd_ps(x1, b1, c51);

// Broadcasts pipelined one row ahead, all 16 registers in use, the depth loop unrolled U times.
// Shapes 6 (U 1) and 7 (U 2) without fences, 10 (U 2) and 11 (U 4) with a fence between the steps.
template <int U, bool Fence>
MENSURFLOW_TARGET_AVX2 void kernel_6x16_pipe(std::int64_t kc, const float* pa, const float* pb,
                                             float* c, std::int64_t ldc, bool accumulate) {
    MENSURFLOW_ZERO_6X16()
    __m256 b0, b1, x1;
    __m256 x0 = _mm256_broadcast_ss(pa);
    std::int64_t p = 0;
    for (; p + U < kc; p += U) { // every step but the last reads the next step's first A value
        MENSURFLOW_STEP_PIPE(0)
        if constexpr (U >= 2) {
            if constexpr (Fence) {
                MENSURFLOW_STEP_FENCE();
            }
            MENSURFLOW_STEP_PIPE(1)
        }
        if constexpr (U == 4) {
            if constexpr (Fence) {
                MENSURFLOW_STEP_FENCE();
            }
            MENSURFLOW_STEP_PIPE(2)
            if constexpr (Fence) {
                MENSURFLOW_STEP_FENCE();
            }
            MENSURFLOW_STEP_PIPE(3)
        }
        pa += 6 * U;
        pb += 16 * U;
    }
    for (; p + 1 < kc; ++p) {
        MENSURFLOW_STEP_PIPE(0)
        pa += 6;
        pb += 16;
    }
    MENSURFLOW_STEP_PIPE_LAST()
    MENSURFLOW_STORE_6X16()
}

using KernelFn = void (*)(std::int64_t, const float*, const float*, float*, std::int64_t, bool);

// A micro-kernel and the tile it computes. Every shape has mr * nr == 96, which sizes the scratch
// tile used for partial tiles at the edges.
struct Shape {
    std::int64_t mr;
    std::int64_t nr;
    KernelFn kernel;
};
constexpr std::int64_t max_tile_floats = 96;

Shape select_shape(const PackedGemmBlocking& blocking) {
    switch (blocking.shape) {
    case 1:
        return {6, 16, &kernel_6x16_unrolled};
    case 2:
        return {4, 24, &kernel_4x24};
    case 3:
        return {12, 8, &kernel_12x8};
    case 4:
        return {6, 16, &kernel_6x16_reuse<2, false>};
    case 5:
        return {6, 16, &kernel_6x16_reuse<4, false>};
    case 6:
        return {6, 16, &kernel_6x16_pipe<1, false>};
    case 7:
        return {6, 16, &kernel_6x16_pipe<2, false>};
    case 8:
        return {6, 16, &kernel_6x16_reuse<2, true>};
    case 9:
        return {6, 16, &kernel_6x16_reuse<4, true>};
    case 10:
        return {6, 16, &kernel_6x16_pipe<2, true>};
    case 11:
        return {6, 16, &kernel_6x16_pipe<4, true>};
    default:
        switch (blocking.prefetch) {
        case 1:
            return {6, 16, &kernel_6x16<1>};
        case 2:
            return {6, 16, &kernel_6x16<2>};
        case 3:
            return {6, 16, &kernel_6x16<3>};
        case 4:
            return {6, 16, &kernel_6x16<4>};
        case 5:
            return {6, 16, &kernel_6x16<5>};
        case 6:
            return {6, 16, &kernel_6x16<6>};
        default:
            return {6, 16, &kernel_6x16<0>};
        }
    }
}


// 64-byte aligned scratch memory that grows on demand and is kept between calls, so a matmul does
// not pay for allocating and page-faulting its packing buffers every time (that dominated small
// products). One pair per calling thread, freed when that thread exits. The contents are not kept:
// reserve() only guarantees room for `floats` floats.
struct ScratchBuffer {
    ScratchBuffer() = default;
    ~ScratchBuffer() { release(); }
    ScratchBuffer(const ScratchBuffer&) = delete;
    ScratchBuffer& operator=(const ScratchBuffer&) = delete;

    void reserve(std::size_t floats) {
        if (floats <= capacity) {
            return;
        }
        release();
#if defined(_MSC_VER)
        data = static_cast<float*>(_aligned_malloc(floats * sizeof(float), 64));
#else
        data = static_cast<float*>(std::aligned_alloc(64, (floats * sizeof(float) + 63) / 64 * 64));
#endif
        if (data == nullptr) {
            throw std::bad_alloc();
        }
        capacity = floats;
    }

    void release() {
        if (data != nullptr) {
#if defined(_MSC_VER)
            _aligned_free(data);
#else
            std::free(data);
#endif
            data = nullptr;
            capacity = 0;
        }
    }

    float* data = nullptr;
    std::size_t capacity = 0;
};

ScratchBuffer& scratch_a() {
    thread_local ScratchBuffer buffer;
    return buffer;
}

ScratchBuffer& scratch_b() {
    thread_local ScratchBuffer buffer;
    return buffer;
}

bool cpu_supports_avx2_fma() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuid(info, 1);
    const bool fma = (info[2] & (1 << 12)) != 0;
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    if (!(fma && osxsave && avx)) {
        return false;
    }
    if ((_xgetbv(0) & 0x6) != 0x6) { // the OS saves XMM and YMM state
        return false;
    }
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

} // namespace


bool packed_gemm_supported() {
    static const bool supported = cpu_supports_avx2_fma();
    return supported;
}

CacheSizes cpu_cache_sizes() {
    return {cache_bytes(1), cache_bytes(2), cache_bytes(3)};
}

PackedGemmBlocking packed_gemm_blocking() {
    return blocking_state();
}

bool packed_microkernel_tile(int shape, std::int64_t& mr, std::int64_t& nr) {
    if (shape < 0 || shape > 11 || !packed_gemm_supported()) {
        return false;
    }
    PackedGemmBlocking blocking;
    blocking.shape = shape;
    const Shape s = select_shape(blocking);
    mr = s.mr;
    nr = s.nr;
    return true;
}

bool run_packed_microkernel(int shape, std::int64_t kc, const float* pa, const float* pb, float* c,
                            std::int64_t ldc, bool accumulate) {
    if (shape < 0 || shape > 11 || !packed_gemm_supported()) {
        return false;
    }
    PackedGemmBlocking blocking;
    blocking.shape = shape;
    select_shape(blocking).kernel(kc, pa, pb, c, ldc, accumulate);
    return true;
}

void set_packed_gemm_blocking(PackedGemmBlocking blocking) {
    if (blocking.mc <= 0 || blocking.kc <= 0 || blocking.nc <= 0 || blocking.tiles_per_thread <= 0 ||
        blocking.prefetch < 0 || blocking.prefetch > 6 || blocking.prefetch_block < 0 ||
        blocking.prefetch_block > 4 || blocking.shape < 0 || blocking.shape > 11 ||
        blocking.scalar_pack < 0 || blocking.scalar_pack > 1 || blocking.parallel < 0 ||
        blocking.parallel > 5 || blocking.loop_order < 0 || blocking.loop_order > 1 ||
        blocking.profile < 0 || blocking.profile > 1 || blocking.pool < 0 || blocking.pool > 1 ||
        blocking.private_b < 0 || blocking.private_b > 2 || blocking.b_chunk < 0 || blocking.pin < 0 || blocking.pin > 3) {
        throw std::invalid_argument("packed gemm blocking: mc, kc, nc and tiles_per_thread must be "
                                    "positive, prefetch 0-6, prefetch_block 0-4, shape 0-11, scalar_pack 0-1, parallel "
                                    "0-5, loop_order 0-1, profile 0-1, pin 0-3");
    }
    blocking_state() = blocking;
}

namespace {

PackedGemmProfile g_profile;

// This thread's number in the running team: the pool's index when it runs on the thread pool, else
// OpenMP's.
inline int team_index() {
    const int tid = pool_thread_index();
    return tid >= 0 ? tid : thread_index();
}
std::atomic<bool> g_workers_pinned{false}; // some worker threads were pinned by an earlier call
std::atomic<bool> g_pin_busy{false};       // a call is pinning the threads right now

// Busy time of one thread, for the profile; padded to a cache line so threads do not share one.
struct alignas(64) ThreadClock {
    double pack = 0.0;
    double compute = 0.0;
    double padding[6] = {};
};

// C[0..mc, 0..cols) (row stride ldc) = or += the product of the mc rows of packed A (first panel at
// pa) and the `cols` columns of packed B (first panel at pb), over a depth of kc. Partial tiles at the
// right and bottom edges are computed in full into a scratch tile and copied.
//
// loop_order 0: for each B panel, all A panels (the B panel is reused while it is in L1/L2).
// loop_order 1: for each A panel, all B panels (the A panel is reused).
//
// prefetch_block (prefetches issued between tiles, 6 x 16 tiles only):
//   0 none
//   1 the C tile of the next tile (T0) before computing this one
//   2 the same with write intent (PREFETCHW)
//   3 loop order 1: the next A panel, spread over the tiles of the current row (a share of its lines
//     into L2 before each tile, so it is there when the row ends; BLIS does this)
//   4 2 and 3 together
inline void compute_block(const Shape& shape, std::int64_t kc, const float* pa, std::int64_t mc,
                          const float* pb, std::int64_t cols, float* c, std::int64_t ldc,
                          bool accumulate, std::int64_t loop_order,
                          std::int64_t prefetch_block = 0) {
    const std::int64_t MR = shape.mr;
    const std::int64_t NR = shape.nr;
    const KernelFn kernel = shape.kernel;
    alignas(64) float edge[max_tile_floats];
    if (prefetch_block != 0 && MR == 6 && NR == 16) {
        const bool next_c = prefetch_block == 1 || prefetch_block == 2 || prefetch_block == 4;
        const bool write = prefetch_block != 1;
        const bool next_a = (prefetch_block == 3 || prefetch_block == 4) && loop_order == 1;
        const auto prefetch_c = [&](std::int64_t ir, std::int64_t jr) {
            if (next_c && ir + MR <= mc && jr + NR <= cols) {
                if (write) {
                    prefetch_c_tile<true>(c + ir * ldc + jr, ldc);
                } else {
                    prefetch_c_tile<false>(c + ir * ldc + jr, ldc);
                }
            }
        };
        const std::int64_t row_tiles = (cols + NR - 1) / NR;
        const std::int64_t a_lines = (kc * MR * 4 + 63) / 64;
        const std::int64_t share = (a_lines + row_tiles - 1) / row_tiles;
        const auto tile_at = [&](std::int64_t ir, std::int64_t jr) {
            const std::int64_t rows = std::min(MR, mc - ir);
            const std::int64_t cols_here = std::min(NR, cols - jr);
            const float* pa_tile = pa + (ir / MR) * kc * MR;
            const float* pb_tile = pb + (jr / NR) * kc * NR;
            float* ctile = c + ir * ldc + jr;
            if (rows == MR && cols_here == NR) {
                kernel(kc, pa_tile, pb_tile, ctile, ldc, accumulate);
            } else {
                kernel(kc, pa_tile, pb_tile, edge, NR, false);
                for (std::int64_t r = 0; r < rows; ++r) {
                    for (std::int64_t q = 0; q < cols_here; ++q) {
                        ctile[r * ldc + q] = accumulate ? ctile[r * ldc + q] + edge[r * NR + q]
                                                        : edge[r * NR + q];
                    }
                }
            }
        };
        if (loop_order == 0) {
            for (std::int64_t jr = 0; jr < cols; jr += NR) {
                for (std::int64_t ir = 0; ir < mc; ir += MR) {
                    if (ir + MR < mc) {
                        prefetch_c(ir + MR, jr);
                    } else {
                        prefetch_c(0, jr + NR);
                    }
                    tile_at(ir, jr);
                }
            }
        } else {
            for (std::int64_t ir = 0; ir < mc; ir += MR) {
                const char* a_next = ir + MR < mc
                                         ? reinterpret_cast<const char*>(pa + (ir / MR + 1) * kc * MR)
                                         : nullptr;
                for (std::int64_t jr = 0, t = 0; jr < cols; jr += NR, ++t) {
                    if (jr + NR < cols) {
                        prefetch_c(ir, jr + NR);
                    } else {
                        prefetch_c(ir + MR, 0);
                    }
                    if (next_a && a_next != nullptr) {
                        const std::int64_t end = std::min(a_lines, (t + 1) * share);
                        for (std::int64_t line = t * share; line < end; ++line) {
                            _mm_prefetch(a_next + line * 64, _MM_HINT_T1);
                        }
                    }
                    tile_at(ir, jr);
                }
            }
        }
        return;
    }
    const auto tile = [&](std::int64_t ir, std::int64_t jr) {
        const std::int64_t rows = std::min(MR, mc - ir);
        const std::int64_t cols_here = std::min(NR, cols - jr);
        const float* pa_tile = pa + (ir / MR) * kc * MR;
        const float* pb_tile = pb + (jr / NR) * kc * NR;
        float* ctile = c + ir * ldc + jr;
        if (rows == MR && cols_here == NR) {
            kernel(kc, pa_tile, pb_tile, ctile, ldc, accumulate);
        } else {
            kernel(kc, pa_tile, pb_tile, edge, NR, false);
            for (std::int64_t r = 0; r < rows; ++r) {
                for (std::int64_t q = 0; q < cols_here; ++q) {
                    if (accumulate) {
                        ctile[r * ldc + q] += edge[r * NR + q];
                    } else {
                        ctile[r * ldc + q] = edge[r * NR + q];
                    }
                }
            }
        }
    };
    if (loop_order == 0) {
        for (std::int64_t jr = 0; jr < cols; jr += NR) {
            for (std::int64_t ir = 0; ir < mc; ir += MR) {
                tile(ir, jr);
            }
        }
    } else {
        for (std::int64_t ir = 0; ir < mc; ir += MR) {
            for (std::int64_t jr = 0; jr < cols; jr += NR) {
                tile(ir, jr);
            }
        }
    }
}

// Strategy 2 gives each thread a block of rows and a range of B panels: team threads become
// ic_ways x jr_ways. Prefer all threads on rows (each packs its own A rows once); when there are too
// few rows for that (fewer than 48 per thread), split the columns as well, which makes the threads
// of one row group pack the same A rows redundantly. Only exact divisors of the team are used.
inline void make_grid(std::int64_t team, std::int64_t m, std::int64_t& ic_ways,
                      std::int64_t& jr_ways) {
    constexpr std::int64_t min_rows_per_thread = 48;
    ic_ways = 1;
    for (std::int64_t ways = team; ways >= 1; --ways) {
        if (team % ways == 0 && (ways == 1 || m / ways >= min_rows_per_thread)) {
            ic_ways = ways;
            break;
        }
    }
    jr_ways = team / ic_ways;
}

// One row times a matrix: c[0..n) = sum over p of a[p * csa] * B[p, :], with the rows of B contiguous
// (rsb floats apart). This is a network evaluated on a single sample. The packed kernel would fill
// one of its six rows and copy all of B first; here B is read once, in place, four rows per pass over
// a block of C that stays in L1. The products are added in increasing p, as the packed kernel does.
MENSURFLOW_TARGET_AVX2 void row_times_matrix(const float* a, std::int64_t csa, const float* b,
                                             std::int64_t rsb, float* c, std::int64_t n,
                                             std::int64_t k) {
    constexpr std::int64_t block = 2048; // 8 KB of C
    for (std::int64_t j0 = 0; j0 < n; j0 += block) {
        const std::int64_t cols = std::min(block, n - j0);
        float* const out = c + j0;
        std::fill(out, out + cols, 0.0f);
        std::int64_t p = 0;
        for (; p + 4 <= k; p += 4) {
            const __m256 a0 = _mm256_broadcast_ss(a + p * csa);
            const __m256 a1 = _mm256_broadcast_ss(a + (p + 1) * csa);
            const __m256 a2 = _mm256_broadcast_ss(a + (p + 2) * csa);
            const __m256 a3 = _mm256_broadcast_ss(a + (p + 3) * csa);
            const float* const b0 = b + p * rsb + j0;
            const float* const b1 = b0 + rsb;
            const float* const b2 = b1 + rsb;
            const float* const b3 = b2 + rsb;
            std::int64_t j = 0;
            for (; j + 16 <= cols; j += 16) {
                __m256 x = _mm256_loadu_ps(out + j);
                __m256 y = _mm256_loadu_ps(out + j + 8);
                x = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), x);
                y = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j + 8), y);
                x = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j), x);
                y = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j + 8), y);
                x = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j), x);
                y = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j + 8), y);
                x = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j), x);
                y = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j + 8), y);
                _mm256_storeu_ps(out + j, x);
                _mm256_storeu_ps(out + j + 8, y);
            }
            for (; j + 8 <= cols; j += 8) {
                __m256 x = _mm256_loadu_ps(out + j);
                x = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), x);
                x = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j), x);
                x = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j), x);
                x = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j), x);
                _mm256_storeu_ps(out + j, x);
            }
            for (; j < cols; ++j) {
                __m128 x = _mm_load_ss(out + j);
                x = _mm_fmadd_ss(_mm256_castps256_ps128(a0), _mm_load_ss(b0 + j), x);
                x = _mm_fmadd_ss(_mm256_castps256_ps128(a1), _mm_load_ss(b1 + j), x);
                x = _mm_fmadd_ss(_mm256_castps256_ps128(a2), _mm_load_ss(b2 + j), x);
                x = _mm_fmadd_ss(_mm256_castps256_ps128(a3), _mm_load_ss(b3 + j), x);
                _mm_store_ss(out + j, x);
            }
        }
        for (; p < k; ++p) {
            const __m256 a0 = _mm256_broadcast_ss(a + p * csa);
            const float* const b0 = b + p * rsb + j0;
            std::int64_t j = 0;
            for (; j + 8 <= cols; j += 8) {
                _mm256_storeu_ps(out + j, _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j),
                                                          _mm256_loadu_ps(out + j)));
            }
            for (; j < cols; ++j) {
                _mm_store_ss(out + j, _mm_fmadd_ss(_mm256_castps256_ps128(a0), _mm_load_ss(b0 + j),
                                                   _mm_load_ss(out + j)));
            }
        }
    }
}

} // namespace

void matmul_f32_packed_strided(const float* a, std::int64_t rsa, std::int64_t csa, const float* b,
                               std::int64_t rsb, std::int64_t csb, float* c, std::int64_t ldc,
                               std::int64_t m, std::int64_t n, std::int64_t k) {
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        for (std::int64_t i = 0; i < m; ++i) {
            std::fill(c + i * ldc, c + i * ldc + n, 0.0f);
        }
        return;
    }
    if (!packed_gemm_supported()) {
        matmul_strided_fallback(a, rsa, csa, b, rsb, csb, c, ldc, m, n, k, false);
        return;
    }
    if (m == 1 && csb == 1) {
        row_times_matrix(a, csa, b, rsb, c, n, k);
        return;
    }

    const Shape shape = select_shape(blocking_state());
    const std::int64_t MR = shape.mr;
    const std::int64_t NR = shape.nr;
    const bool scalar_pack = blocking_state().scalar_pack != 0;
    const std::int64_t loop_order = blocking_state().loop_order;
    const std::int64_t prefetch_block = blocking_state().prefetch_block;
    // Block sizes are rounded down to whole tiles (at least one).
    const std::int64_t MC = std::max(MR, blocking_state().mc / MR * MR);
    const std::int64_t KC = blocking_state().kc;
    const std::int64_t NC = std::max(NR, blocking_state().nc / NR * NR);
    const auto round_up = [](std::int64_t value, std::int64_t multiple) {
        return (value + multiple - 1) / multiple * multiple;
    };
    const std::int64_t kc_max = std::min(KC, k);
    const std::int64_t mc_max = std::min(MC, round_up(m, MR));
    const std::int64_t nc_max = std::min(NC, round_up(n, NR));
    ScratchBuffer& packed_a = scratch_a();
    ScratchBuffer& packed_b = scratch_b();

    // Few rows (see PackedGemmBlocking::private_b): all of A is packed once per depth step, and B a
    // few panels at a time into a buffer that stays in L2, instead of a KC x NC block of B that is
    // written to L3 or memory and read back once.
    const std::int64_t private_b = blocking_state().private_b;
    std::int64_t chunk = blocking_state().b_chunk;
    if (chunk == 0) {
        const std::int64_t l2 = cache_bytes(2) > 0 ? cache_bytes(2) : 256 * 1024;
        chunk = std::max<std::int64_t>(1, l2 / 2 / (kc_max * NR * 4));
    }
    if ((private_b == 1 || (private_b == 2 && m <= private_b_max_rows)) && n > chunk * NR) {
        const std::int64_t cols_max = chunk * NR;
        packed_a.reserve(static_cast<std::size_t>(round_up(m, MR) * kc_max));
        packed_b.reserve(static_cast<std::size_t>(kc_max * cols_max));
        for (std::int64_t pc = 0; pc < k; pc += KC) {
            const std::int64_t kc = std::min(KC, k - pc);
            pack_a(scalar_pack, MR, a + pc * csa, rsa, csa, m, kc, packed_a.data);
            for (std::int64_t jc = 0; jc < n; jc += cols_max) {
                const std::int64_t cols = std::min(cols_max, n - jc);
                pack_b(scalar_pack, NR, b + pc * rsb + jc * csb, rsb, csb, kc, cols, packed_b.data,
                       chunk, private_b_pack_ahead);
                for (std::int64_t ic = 0; ic < m; ic += MC) {
                    compute_block(shape, kc, packed_a.data + (ic / MR) * kc * MR,
                                  std::min(MC, m - ic), packed_b.data, cols, c + ic * ldc + jc, ldc,
                                  pc > 0, loop_order, prefetch_block);
                }
            }
        }
        return;
    }

    packed_a.reserve(static_cast<std::size_t>(mc_max * kc_max));
    packed_b.reserve(static_cast<std::size_t>(kc_max * nc_max));

    for (std::int64_t jc = 0; jc < n; jc += NC) {
        const std::int64_t nc = std::min(NC, n - jc);
        for (std::int64_t pc = 0; pc < k; pc += KC) {
            const std::int64_t kc = std::min(KC, k - pc);
            const bool accumulate = pc > 0;
            pack_b(scalar_pack, NR, b + pc * rsb + jc * csb, rsb, csb, kc, nc, packed_b.data);
            for (std::int64_t ic = 0; ic < m; ic += MC) {
                const std::int64_t mc = std::min(MC, m - ic);
                pack_a(scalar_pack, MR, a + ic * rsa + pc * csa, rsa, csa, mc, kc, packed_a.data);
                compute_block(shape, kc, packed_a.data, mc, packed_b.data, nc,
                              c + ic * ldc + jc, ldc, accumulate, loop_order, prefetch_block);
            }
        }
    }
}

// ---- Threaded version ---------------------------------------------------------------------------

PackedGemmProfile packed_gemm_profile() {
    return g_profile;
}

// The same five loops, with the work split across threads. Per (jc, pc) step the threads
//   1. pack B (and, in strategies 0 and 1, all of A) together, then
//   2. compute their parts of C. Every thread writes only its own part, so nothing is locked.
// Strategy 0 does each of those as its own parallel loop, so the team is woken twice per step.
// Strategy 1 is the same work inside one parallel region for the whole call (the team is woken once,
// with a barrier between the steps). Strategy 2 also runs in one region, but each thread packs the
// rows of A it will multiply into its own buffer, so A is written and read by the same core.
void matmul_f32_packed_parallel_strided(const float* a, std::int64_t rsa, std::int64_t csa,
                                        const float* b, std::int64_t rsb, std::int64_t csb, float* c,
                                        std::int64_t ldc, std::int64_t m, std::int64_t n,
                                        std::int64_t k) {
    // Small products get fewer threads: each thread should have at least min_macs_per_thread
    // multiply-adds, or waking and joining it costs more than it saves (measured: n = 128 is
    // fastest on 4 of 24 threads).
    const double macs = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    std::int64_t threads = std::min<std::int64_t>(
        max_threads(), std::max<std::int64_t>(1, static_cast<std::int64_t>(macs / min_macs_per_thread)));
    // The second hardware thread of a core adds no FMA throughput, only barrier and packing overhead
    // (n = 256: 266 GFLOP/s on 12 threads, 185 on 24), so below min_macs_for_smt stay at one thread per
    // physical core. For large products the extra threads help a little (+5% at n = 4096): they fill
    // the gaps while the other thread of the core waits for memory.
    const int physical_cores = physical_core_count();
    if (physical_cores > 0 && macs < min_macs_for_smt) {
        threads = std::min<std::int64_t>(threads, physical_cores);
    }
    const bool worth = threads > 1 && macs > static_cast<double>(parallel_work_threshold);
    if (!worth || !packed_gemm_supported()) {
        // Also handles m, n or k == 0, and a CPU without AVX2.
        matmul_f32_packed_strided(a, rsa, csa, b, rsb, csb, c, ldc, m, n, k);
        return;
    }
    if (m == 1 && csb == 1) {
        // One row (see row_times_matrix): each thread computes its own run of columns. At least 256
        // columns per thread; on the pool, or on this thread alone when the pool is taken.
        const std::int64_t team = std::min<std::int64_t>(threads, n / 256);
        if (team > 1 && blocking_state().pool != 0) {
            auto body = [&](int tid) {
                const std::int64_t j0 = n * tid / team / 16 * 16;
                const std::int64_t j1 = tid + 1 == team ? n : n * (tid + 1) / team / 16 * 16;
                row_times_matrix(a, csa, b + j0, rsb, c + j0, j1 - j0, k);
            };
            if (pool_run(static_cast<int>(team), body)) {
                return;
            }
        }
        row_times_matrix(a, csa, b, rsb, c, n, k);
        return;
    }

    const PackedGemmBlocking settings = blocking_state();
    const Shape shape = select_shape(settings);
    const std::int64_t MR = shape.mr;
    const std::int64_t NR = shape.nr;
    const bool scalar_pack = settings.scalar_pack != 0;
    const std::int64_t strategy = settings.parallel;
    const std::int64_t loop_order = settings.loop_order;
    const std::int64_t prefetch_block = settings.prefetch_block;
    const bool profile = settings.profile != 0;
    // Block sizes are rounded down to whole tiles (at least one).
    const std::int64_t MC = std::max(MR, settings.mc / MR * MR);
    const std::int64_t KC = settings.kc;
    const std::int64_t NC = std::max(NR, settings.nc / NR * NR);
    const auto round_up = [](std::int64_t value, std::int64_t multiple) {
        return (value + multiple - 1) / multiple * multiple;
    };
    const auto ceil_div = [](std::int64_t value, std::int64_t divisor) {
        return (value + divisor - 1) / divisor;
    };
    const std::int64_t m_padded = round_up(m, MR);
    const std::int64_t kc_max = std::min(KC, k);
    const std::int64_t nc_max = std::min(NC, round_up(n, NR));
    // Strategy 5 splits the team by level-3 cache: thread t runs on core t (when pinned), so its group
    // is that core's cache domain, numbered 0, 1, ... in order of appearance. Unpinned threads can be
    // anywhere, so then there is a single group.
    std::vector<int> group_of(static_cast<std::size_t>(threads), 0);
    std::int64_t groups = 1;
    const bool split = strategy == 5 || (strategy == 4 && macs < max_macs_for_groups);
    if (split && (settings.pin == 1 || settings.pin == 2) && macs >= min_macs_to_pin) {
        std::vector<int> seen;
        for (std::int64_t t = 0; t < threads; ++t) {
            const int domain = core_cache_domain(static_cast<int>(t));
            const auto found = std::find(seen.begin(), seen.end(), domain);
            group_of[static_cast<std::size_t>(t)] = static_cast<int>(found - seen.begin());
            if (found == seen.end()) {
                seen.push_back(domain);
            }
        }
        groups = static_cast<std::int64_t>(seen.size());
    }
    // Strategies 4 and 5 keep two copies of the packed A and B (one being multiplied, one being
    // packed), and strategy 5 one copy of A per group.
    const std::int64_t copies = strategy >= 4 ? 2 : 1;
    // Private B (strategies 4 and 5): no shared packed B. Each thread packs the B panels of its tile
    // into its own buffer, small enough to stay in its L2, and multiplies them by every row of A
    // straight away. The shared buffer costs a write and a read of all of B through L3 or memory per
    // call, which is what limits products with few rows (m = 32, n = k = 4096: 64 MB of B for 10 ms
    // of arithmetic on one core); the private one only reads B.
    const bool private_b =
        strategy >= 4 && (settings.private_b == 1 ||
                          (settings.private_b == 2 && m <= private_b_max_rows &&
                           // One MC block of rows: a tile is a run of B panels either way.
                           (m <= MC ||
                           ceil_div(std::min(n, NC), NR) >= 4 * threads)));
    std::int64_t chunk_panels = settings.b_chunk;
    if (private_b && chunk_panels == 0) {
        const std::int64_t l2 = cache_bytes(2) > 0 ? cache_bytes(2) : 256 * 1024;
        chunk_panels = std::max<std::int64_t>(1, l2 / 2 / (kc_max * NR * 4));
    }
    ScratchBuffer& packed_b = scratch_b();
    packed_b.reserve(static_cast<std::size_t>(
        private_b ? chunk_panels * NR * kc_max : copies * kc_max * nc_max));
    float* const pb_base = packed_b.data;
    float* pa_base = nullptr; // the shared packed A of strategies 0, 1, 4 and 5
    if (strategy <= 1 || strategy >= 4) {
        ScratchBuffer& packed_a = scratch_a();
        packed_a.reserve(static_cast<std::size_t>(copies * groups * m_padded * kc_max));
        pa_base = packed_a.data;
    }
    const std::int64_t a_panels = m_padded / MR;
    const std::int64_t ic_blocks = ceil_div(m, MC);

    std::vector<ThreadClock> clocks(profile ? static_cast<std::size_t>(max_threads()) : 0);
    std::int64_t tiles_total = 0;
    const double wall_start = profile ? wall_seconds() : 0.0;

    // Work items of strategies 0 and 1: a tile is one MC-row block times a group of B panels.
    // About tiles_per_thread of them per thread.
    const auto tile_layout = [&](std::int64_t b_panels, std::int64_t& panels_per_group,
                                 std::int64_t& tiles) {
        const std::int64_t wanted_groups = ceil_div(settings.tiles_per_thread * threads, ic_blocks);
        panels_per_group =
            ceil_div(b_panels, std::min(std::max<std::int64_t>(wanted_groups, 1), b_panels));
        tiles = ic_blocks * ceil_div(b_panels, panels_per_group);
    };

    // A pack work item of B is a run of b_run panels: the packer then reads each source row of B as
    // one run of b_run cache lines instead of a single line (see detail::pack_b_panels_avx2).
    const std::int64_t b_run = scalar_pack ? 1 : detail::pack_b_run;
    const auto b_items_of = [&](std::int64_t panels) { return ceil_div(panels, b_run); };

    // B panels [first, last) of the step (jc, pc).
    const auto pack_b_range = [&](std::int64_t first, std::int64_t last, std::int64_t jc,
                                  std::int64_t pc, std::int64_t kc, std::int64_t nc, float* pb) {
        const double t0 = profile ? wall_seconds() : 0.0;
        const std::int64_t j = first * NR;
        pack_b(scalar_pack, NR, b + pc * rsb + (jc + j) * csb, rsb, csb, kc,
               std::min((last - first) * NR, nc - j), pb + first * kc * NR);
        if (profile) {
            clocks[static_cast<std::size_t>(team_index())].pack += wall_seconds() - t0;
        }
    };
    // A panel `panel` (MR rows) of the depth step pc.
    const auto pack_a_panel = [&](std::int64_t panel, std::int64_t pc, std::int64_t kc, float* pa) {
        const double t0 = profile ? wall_seconds() : 0.0;
        const std::int64_t i = panel * MR;
        pack_a(scalar_pack, MR, a + i * rsa + pc * csa, rsa, csa, std::min(MR, m - i), kc,
               pa + panel * kc * MR);
        if (profile) {
            clocks[static_cast<std::size_t>(team_index())].pack += wall_seconds() - t0;
        }
    };

    // One pack work item (strategies 0 and 1): run t of B, or panel t - b_items_of(b_panels) of A.
    const auto pack_item = [&](std::int64_t t, std::int64_t jc, std::int64_t pc, std::int64_t kc,
                               std::int64_t nc, std::int64_t b_panels, float* pa, float* pb) {
        const std::int64_t b_items = b_items_of(b_panels);
        if (t < b_items) {
            pack_b_range(t * b_run, std::min((t + 1) * b_run, b_panels), jc, pc, kc, nc, pb);
        } else {
            pack_a_panel(t - b_items, pc, kc, pa);
        }
    };

    // One compute work item (strategies 0 and 1).
    const auto tile_item = [&](std::int64_t t, std::int64_t jc, std::int64_t kc, std::int64_t nc,
                               std::int64_t b_panels, std::int64_t panels_per_group,
                               bool accumulate, const float* pa, const float* pb) {
        const double t0 = profile ? wall_seconds() : 0.0;
        const std::int64_t ic = (t % ic_blocks) * MC;
        const std::int64_t mc = std::min(MC, m - ic);
        const std::int64_t first_panel = (t / ic_blocks) * panels_per_group;
        const std::int64_t last_panel = std::min(first_panel + panels_per_group, b_panels);
        const std::int64_t cols = std::min((last_panel - first_panel) * NR, nc - first_panel * NR);
        compute_block(shape, kc, pa + (ic / MR) * kc * MR, mc, pb + first_panel * kc * NR,
                      cols, c + ic * ldc + jc + first_panel * NR, ldc, accumulate, loop_order, prefetch_block);
        if (profile) {
            clocks[static_cast<std::size_t>(team_index())].compute += wall_seconds() - t0;
        }
    };

    // Thread placement (see affinity.hpp). The calling thread is thread 0 of the team: it is pinned for
    // the duration of the call and gets its old restriction back at the end. The worker threads stay
    // pinned until a call asks for no pinning, which releases them once.
    const int pin = static_cast<int>(settings.pin);
    // Only one call pins at a time (threads that call matmul concurrently would otherwise all pin
    // their calling thread to core 0), and never from inside another parallel region.
    const bool want_pin = pin != 0 && macs >= min_macs_to_pin && !in_parallel_region() &&
                          pool_thread_index() < 0;
    const bool do_pin = want_pin && !g_pin_busy.exchange(true);
    if (do_pin) {
        set_thread_pin(0, pin);
        g_workers_pinned = true;
        if (strategy == 0) { // the other strategies pin at the start of their own region
            MENSURFLOW_PARALLEL_REGION(num_threads(static_cast<int>(threads)))
            {
                set_thread_pin(thread_index(), pin);
            }
        }
    } else if (pin == 0 && g_workers_pinned) {
        MENSURFLOW_PARALLEL_REGION(num_threads(static_cast<int>(max_threads())))
        {
            set_thread_pin(thread_index(), 0);
        }
        g_workers_pinned = false;
    }

    if (strategy == 0) {
        for (std::int64_t jc = 0; jc < n; jc += NC) {
            const std::int64_t nc = std::min(NC, n - jc);
            const std::int64_t b_panels = ceil_div(nc, NR);
            std::int64_t panels_per_group = 0;
            std::int64_t tiles = 0;
            tile_layout(b_panels, panels_per_group, tiles);
            for (std::int64_t pc = 0; pc < k; pc += KC) {
                const std::int64_t kc = std::min(KC, k - pc);
                const bool accumulate = pc > 0;
                tiles_total += tiles;

                MENSURFLOW_PARALLEL_FOR(schedule(static) num_threads(static_cast<int>(threads)))
                for (std::int64_t t = 0; t < b_items_of(b_panels) + a_panels; ++t) {
                    pack_item(t, jc, pc, kc, nc, b_panels, pa_base, pb_base);
                }

                MENSURFLOW_PARALLEL_FOR(schedule(dynamic) num_threads(static_cast<int>(threads)))
                for (std::int64_t t = 0; t < tiles; ++t) {
                    tile_item(t, jc, kc, nc, b_panels, panels_per_group, accumulate, pa_base,
                              pb_base);
                }
            }
        }
    } else if (strategy == 1) {
        MENSURFLOW_PARALLEL_REGION(num_threads(static_cast<int>(threads)))
        {
            if (do_pin) {
                set_thread_pin(thread_index(), pin);
            }
            for (std::int64_t jc = 0; jc < n; jc += NC) {
                const std::int64_t nc = std::min(NC, n - jc);
                const std::int64_t b_panels = ceil_div(nc, NR);
                std::int64_t panels_per_group = 0;
                std::int64_t tiles = 0;
                tile_layout(b_panels, panels_per_group, tiles);
                for (std::int64_t pc = 0; pc < k; pc += KC) {
                    const std::int64_t kc = std::min(KC, k - pc);
                    const bool accumulate = pc > 0;

                    MENSURFLOW_FOR(schedule(static))
                    for (std::int64_t t = 0; t < b_items_of(b_panels) + a_panels; ++t) {
                        pack_item(t, jc, pc, kc, nc, b_panels, pa_base, pb_base);
                    }

                    MENSURFLOW_FOR(schedule(dynamic))
                    for (std::int64_t t = 0; t < tiles; ++t) {
                        tile_item(t, jc, kc, nc, b_panels, panels_per_group, accumulate, pa_base,
                                  pb_base);
                    }
                }
            }
        }
        for (std::int64_t jc = 0; jc < n; jc += NC) {
            std::int64_t panels_per_group = 0;
            std::int64_t tiles = 0;
            tile_layout(ceil_div(std::min(NC, n - jc), NR), panels_per_group, tiles);
            tiles_total += tiles * ceil_div(k, KC);
        }
    } else if (strategy >= 4) {
        // Strategy 4: strategy 1 with the packing of the next (jc, pc) step done while this step is
        // multiplied. Packing reads A and B from memory and is limited by memory bandwidth when every
        // thread packs at once; the micro-kernel is limited by the FMA units and hardly touches
        // memory, so mixing the two lets the copies run under the arithmetic. One work list per step:
        // the tiles of this step with the pack items of the next one spread evenly among them, handed
        // out dynamically, writing into the other of two buffers. The barrier at the end of a step is
        // the only one.
        //
        // Strategy 5: the same, with the team split by level-3 cache (one group per chiplet). Each
        // group computes its own share of the columns of C (in proportion to its threads), packs the B
        // panels of those columns and its own copy of A, and takes work only from its own lists, so a
        // packed panel is never read across chiplets. With one cache domain it is strategy 4.
        struct Step {
            std::int64_t jc, nc, pc, kc, b_panels;
        };
        std::vector<Step> steps;
        for (std::int64_t jc = 0; jc < n; jc += NC) {
            const std::int64_t nc = std::min(NC, n - jc);
            for (std::int64_t pc = 0; pc < k; pc += KC) {
                steps.push_back({jc, nc, pc, std::min(KC, k - pc), ceil_div(nc, NR)});
            }
        }
        const std::int64_t step_count = static_cast<std::int64_t>(steps.size());
        std::vector<std::int64_t> first_thread(static_cast<std::size_t>(groups) + 1, 0);
        for (std::int64_t t = 0; t < threads; ++t) {
            ++first_thread[static_cast<std::size_t>(group_of[static_cast<std::size_t>(t)]) + 1];
        }
        for (std::int64_t g = 0; g < groups; ++g) {
            first_thread[static_cast<std::size_t>(g) + 1] += first_thread[static_cast<std::size_t>(g)];
        }

        // Group g's part of step s: B panels [p0, p1), split into tiles like tile_layout does.
        struct Part {
            std::int64_t p0, p1, panels_per_tile, tiles, packs;
        };
        const auto part_of = [&](std::int64_t s, std::int64_t g) {
            const Step& st = steps[static_cast<std::size_t>(s)];
            Part part{};
            part.p0 = st.b_panels * first_thread[static_cast<std::size_t>(g)] / threads;
            part.p1 = st.b_panels * first_thread[static_cast<std::size_t>(g) + 1] / threads;
            const std::int64_t panels = part.p1 - part.p0;
            if (panels > 0) {
                const std::int64_t members = first_thread[static_cast<std::size_t>(g) + 1] -
                                             first_thread[static_cast<std::size_t>(g)];
                if (private_b) {
                    // A tile is a run of B panels times all the rows; at least 4 tiles per thread
                    // so the threads finish together.
                    part.panels_per_tile = std::max<std::int64_t>(
                        1, std::min(chunk_panels, panels / (4 * members)));
                    part.tiles = ceil_div(panels, part.panels_per_tile);
                    part.packs = a_panels; // only its own copy of A
                    return part;
                }
                const std::int64_t wanted = ceil_div(settings.tiles_per_thread * members, ic_blocks);
                part.panels_per_tile =
                    ceil_div(panels, std::min(std::max<std::int64_t>(wanted, 1), panels));
                part.tiles = ic_blocks * ceil_div(panels, part.panels_per_tile);
                part.packs = b_items_of(panels) + a_panels; // its B panels and its own copy of A
            }
            return part;
        };
        for (std::int64_t s = 0; s < step_count; ++s) {
            for (std::int64_t g = 0; g < groups; ++g) {
                tiles_total += part_of(s, g).tiles;
            }
        }
        const auto packed_a_of = [&](std::int64_t s, std::int64_t g) {
            return pa_base + ((s & 1) * groups + g) * m_padded * kc_max;
        };
        const auto packed_b_of = [&](std::int64_t s) { return pb_base + (s & 1) * kc_max * nc_max; };

        // Pack item t of group g for step s: a run of its B panels, then the panels of its copy of A.
        const auto pack_part = [&](std::int64_t s, std::int64_t g, const Part& part, std::int64_t t) {
            const Step& st = steps[static_cast<std::size_t>(s)];
            const std::int64_t b_items = private_b ? 0 : b_items_of(part.p1 - part.p0);
            if (t < b_items) {
                const std::int64_t first = part.p0 + t * b_run;
                pack_b_range(first, std::min(first + b_run, part.p1), st.jc, st.pc, st.kc, st.nc,
                             packed_b_of(s));
            } else {
                pack_a_panel(t - b_items, st.pc, st.kc, packed_a_of(s, g));
            }
        };
        // Tile t of group g for step s.
        const auto tile_part = [&](std::int64_t s, std::int64_t g, const Part& part, std::int64_t t) {
            const Step& st = steps[static_cast<std::size_t>(s)];
            if (private_b) {
                const double t0 = profile ? wall_seconds() : 0.0;
                const std::int64_t first_panel = part.p0 + t * part.panels_per_tile;
                const std::int64_t last_panel = std::min(first_panel + part.panels_per_tile, part.p1);
                const std::int64_t cols =
                    std::min((last_panel - first_panel) * NR, st.nc - first_panel * NR);
                ScratchBuffer& mine = scratch_b();
                mine.reserve(static_cast<std::size_t>(chunk_panels * NR * kc_max));
                pack_b(scalar_pack, NR, b + st.pc * rsb + (st.jc + first_panel * NR) * csb, rsb, csb,
                       st.kc, cols, mine.data, chunk_panels, private_b_pack_ahead);
                const double t1 = profile ? wall_seconds() : 0.0;
                for (std::int64_t ic = 0; ic < m; ic += MC) {
                    compute_block(shape, st.kc, packed_a_of(s, g) + (ic / MR) * st.kc * MR,
                                  std::min(MC, m - ic), mine.data, cols,
                                  c + ic * ldc + st.jc + first_panel * NR, ldc, st.pc > 0, loop_order,
                                  prefetch_block);
                }
                if (profile) {
                    ThreadClock& clock = clocks[static_cast<std::size_t>(team_index())];
                    clock.pack += t1 - t0;
                    clock.compute += wall_seconds() - t1;
                }
                return;
            }
            const double t0 = profile ? wall_seconds() : 0.0;
            const std::int64_t ic = (t % ic_blocks) * MC;
            const std::int64_t mc = std::min(MC, m - ic);
            const std::int64_t first_panel = part.p0 + (t / ic_blocks) * part.panels_per_tile;
            const std::int64_t last_panel = std::min(first_panel + part.panels_per_tile, part.p1);
            const std::int64_t cols =
                std::min((last_panel - first_panel) * NR, st.nc - first_panel * NR);
            compute_block(shape, st.kc, packed_a_of(s, g) + (ic / MR) * st.kc * MR, mc,
                          packed_b_of(s) + first_panel * st.kc * NR, cols,
                          c + ic * ldc + st.jc + first_panel * NR, ldc, st.pc > 0, loop_order,
                          prefetch_block);
            if (profile) {
                clocks[static_cast<std::size_t>(team_index())].compute += wall_seconds() - t0;
            }
        };

        // One work counter per list and group: list 0 packs step 0, list s + 1 multiplies step s
        // (and packs step s + 1). Each on its own cache line.
        struct alignas(64) Counter {
            std::atomic<std::int64_t> next{0};
        };
        std::vector<Counter> counters(static_cast<std::size_t>((step_count + 1) * groups));
        const auto take = [&](std::int64_t list, std::int64_t g) {
            return counters[static_cast<std::size_t>(list * groups + g)].next.fetch_add(
                1, std::memory_order_relaxed);
        };

        // The work of one thread of the team. `barrier` makes the team wait for each other: OpenMP's
        // barrier inside an OpenMP region, the pool's on the thread pool.
        const auto team_body = [&](std::int64_t tid, std::int64_t team, bool on_pool,
                                   const auto& barrier) {
            if (do_pin) {
                set_thread_pin(static_cast<int>(tid), pin);
            } else if (on_pool && tid != 0 && pin != 0) {
                // Pool workers are dedicated threads: pin them even for small products (once; it
                // costs nothing while the placement stays the same).
                set_thread_pin(static_cast<int>(tid), pin);
            }
            const std::int64_t mine = tid < threads ? group_of[static_cast<std::size_t>(tid)] : 0;
            // If the runtime gave us fewer threads than asked, a group may have nobody: then every
            // thread works through every group's list, so all the work is still done.
            const std::int64_t visit = team == threads ? 1 : groups;

            for (std::int64_t v = 0; v < visit; ++v) {
                const std::int64_t g = (mine + v) % groups;
                const Part part = part_of(0, g);
                for (std::int64_t t = take(0, g); t < part.packs; t = take(0, g)) {
                    pack_part(0, g, part, t);
                }
            }
            barrier();
            for (std::int64_t s = 0; s < step_count; ++s) {
                for (std::int64_t v = 0; v < visit; ++v) {
                    const std::int64_t g = (mine + v) % groups;
                    const Part part = part_of(s, g);
                    const Part next = s + 1 < step_count ? part_of(s + 1, g) : Part{};
                    const std::int64_t items = part.tiles + next.packs;
                    // Item t is a pack item when the count of packs among items 0..t steps up at t.
                    for (std::int64_t t = take(s + 1, g); t < items; t = take(s + 1, g)) {
                        const std::int64_t before = t * next.packs / items;
                        const std::int64_t through = (t + 1) * next.packs / items;
                        if (through != before) {
                            pack_part(s + 1, g, next, before);
                        } else {
                            tile_part(s, g, part, t - before);
                        }
                    }
                }
                barrier();
            }
        };

        // On the thread pool (PackedGemmBlocking::pool), or in an OpenMP region when the pool is off
        // or already running another team.
        bool ran = false;
        if (settings.pool != 0) {
            const auto pool_barrier_fn = [] { pool_barrier(); };
            auto on_pool = [&](int tid) { team_body(tid, threads, true, pool_barrier_fn); };
            ran = pool_run(static_cast<int>(threads), on_pool);
        }
        if (!ran) {
            const auto omp_barrier_fn = [] { MENSURFLOW_BARRIER() };
            MENSURFLOW_PARALLEL_REGION(num_threads(static_cast<int>(threads)))
            {
                team_body(thread_index(), team_size(), false, omp_barrier_fn);
            }
        }
    } else if (strategy == 2) {
        std::int64_t workers = 0;
        MENSURFLOW_PARALLEL_REGION(num_threads(static_cast<int>(threads)))
        {
            if (do_pin) {
                set_thread_pin(thread_index(), pin);
            }
            const std::int64_t team = team_size();
            const std::int64_t tid = thread_index();
            std::int64_t ic_ways = 1;
            std::int64_t jr_ways = 1;
            make_grid(team, m, ic_ways, jr_ways);
            const bool active = tid < ic_ways * jr_ways;
            const std::int64_t ti = active ? tid / jr_ways : 0;
            const std::int64_t tj = active ? tid % jr_ways : 0;
            // My rows (whole 6-row panels) and, per column block, my range of B panels.
            const std::int64_t row_units = ceil_div(m, MR);
            const std::int64_t r0 = ti * row_units / ic_ways * MR;
            const std::int64_t r1 = std::min(m, (ti + 1) * row_units / ic_ways * MR);
            float* my_a = nullptr;
            if (active && r1 > r0) {
                ScratchBuffer& buffer = scratch_a();
                // Whole panels: the last panel of a block is padded with zero rows up to MR.
                buffer.reserve(static_cast<std::size_t>(round_up(std::min(MC, r1 - r0), MR) * kc_max));
                my_a = buffer.data;
            }
            if (thread_index() == 0) {
                workers = ic_ways * jr_ways;
            }

            for (std::int64_t jc = 0; jc < n; jc += NC) {
                const std::int64_t nc = std::min(NC, n - jc);
                const std::int64_t b_panels = ceil_div(nc, NR);
                const std::int64_t p0 = tj * b_panels / jr_ways;
                const std::int64_t p1 = (tj + 1) * b_panels / jr_ways;
                for (std::int64_t pc = 0; pc < k; pc += KC) {
                    const std::int64_t kc = std::min(KC, k - pc);
                    const bool accumulate = pc > 0;

                    // B is packed by the whole team (the loop ends with a barrier) ...
                    MENSURFLOW_FOR(schedule(static))
                    for (std::int64_t t = 0; t < b_panels; ++t) {
                        const double t0 = profile ? wall_seconds() : 0.0;
                        const std::int64_t j = t * NR;
                        pack_b(scalar_pack, NR, b + pc * rsb + (jc + j) * csb, rsb, csb, kc, std::min(NR, nc - j),
                               pb_base + t * kc * NR);
                        if (profile) {
                            clocks[static_cast<std::size_t>(team_index())].pack +=
                                wall_seconds() - t0;
                        }
                    }

                    // ... then each thread packs its own rows of A and multiplies them.
                    if (my_a != nullptr && p1 > p0) {
                        const std::int64_t cols = std::min((p1 - p0) * NR, nc - p0 * NR);
                        for (std::int64_t ic = r0; ic < r1; ic += MC) {
                            const std::int64_t mc = std::min(MC, r1 - ic);
                            const double t0 = profile ? wall_seconds() : 0.0;
                            pack_a(scalar_pack, MR, a + ic * rsa + pc * csa, rsa, csa, mc, kc, my_a);
                            const double t1 = profile ? wall_seconds() : 0.0;
                            compute_block(shape, kc, my_a, mc, pb_base + p0 * kc * NR, cols,
                                          c + ic * ldc + jc + p0 * NR, ldc, accumulate, loop_order, prefetch_block);
                            if (profile) {
                                ThreadClock& clock = clocks[static_cast<std::size_t>(team_index())];
                                clock.pack += t1 - t0;
                                clock.compute += wall_seconds() - t1;
                            }
                        }
                    }
                    // Nobody may repack B while another thread is still reading it.
                    MENSURFLOW_BARRIER()
                }
            }
        }
        tiles_total = workers;
    } else {
        // Strategy 3: the tiles of strategies 0 and 1 (dynamically scheduled, so a slow core simply
        // takes fewer of them), but A is packed privately: a thread packs the MC rows of a tile into
        // its own buffer and keeps them for the next tile it takes from the same rows. Tiles are
        // numbered row block first, so neighbouring tiles share their rows.
        MENSURFLOW_PARALLEL_REGION(num_threads(static_cast<int>(threads)))
        {
            if (do_pin) {
                set_thread_pin(thread_index(), pin);
            }
            ScratchBuffer& buffer = scratch_a();
            buffer.reserve(static_cast<std::size_t>(MC * kc_max));
            float* const my_a = buffer.data;

            for (std::int64_t jc = 0; jc < n; jc += NC) {
                const std::int64_t nc = std::min(NC, n - jc);
                const std::int64_t b_panels = ceil_div(nc, NR);
                std::int64_t panels_per_group = 0;
                std::int64_t tiles = 0;
                tile_layout(b_panels, panels_per_group, tiles);
                const std::int64_t groups = ceil_div(b_panels, panels_per_group);
                for (std::int64_t pc = 0; pc < k; pc += KC) {
                    const std::int64_t kc = std::min(KC, k - pc);
                    const bool accumulate = pc > 0;

                    MENSURFLOW_FOR(schedule(static))
                    for (std::int64_t t = 0; t < b_panels; ++t) {
                        const double t0 = profile ? wall_seconds() : 0.0;
                        const std::int64_t j = t * NR;
                        pack_b(scalar_pack, NR, b + pc * rsb + (jc + j) * csb, rsb, csb, kc, std::min(NR, nc - j),
                               pb_base + t * kc * NR);
                        if (profile) {
                            clocks[static_cast<std::size_t>(team_index())].pack +=
                                wall_seconds() - t0;
                        }
                    }

                    std::int64_t packed_ic = -1; // the rows now in my_a (private to this thread)
                    MENSURFLOW_FOR(schedule(dynamic))
                    for (std::int64_t t = 0; t < tiles; ++t) {
                        const std::int64_t ic = (t / groups) * MC;
                        const std::int64_t mc = std::min(MC, m - ic);
                        const std::int64_t first_panel = (t % groups) * panels_per_group;
                        const std::int64_t last_panel =
                            std::min(first_panel + panels_per_group, b_panels);
                        const std::int64_t cols =
                            std::min((last_panel - first_panel) * NR, nc - first_panel * NR);
                        const double t0 = profile ? wall_seconds() : 0.0;
                        if (packed_ic != ic) {
                            pack_a(scalar_pack, MR, a + ic * rsa + pc * csa, rsa, csa, mc, kc, my_a);
                            packed_ic = ic;
                        }
                        const double t1 = profile ? wall_seconds() : 0.0;
                        compute_block(shape, kc, my_a, mc, pb_base + first_panel * kc * NR, cols,
                                      c + ic * ldc + jc + first_panel * NR, ldc, accumulate,
                                      loop_order, prefetch_block);
                        if (profile) {
                            ThreadClock& clock = clocks[static_cast<std::size_t>(team_index())];
                            clock.pack += t1 - t0;
                            clock.compute += wall_seconds() - t1;
                        }
                    }
                }
            }
        }
        for (std::int64_t jc = 0; jc < n; jc += NC) {
            std::int64_t panels_per_group = 0;
            std::int64_t tiles = 0;
            tile_layout(ceil_div(std::min(NC, n - jc), NR), panels_per_group, tiles);
            tiles_total += tiles * ceil_div(k, KC);
        }
    }

    if (do_pin) {
        set_thread_pin(0, 0); // give the calling thread its old placement back
        g_pin_busy.store(false);
    }

    if (profile) {
        PackedGemmProfile result;
        result.wall_s = wall_seconds() - wall_start;
        result.tiles = tiles_total;
        for (const ThreadClock& clock : clocks) {
            if (clock.pack > 0.0 || clock.compute > 0.0) {
                ++result.threads;
                result.pack_mean_s += clock.pack;
                result.compute_mean_s += clock.compute;
                result.pack_max_s = std::max(result.pack_max_s, clock.pack);
                result.compute_max_s = std::max(result.compute_max_s, clock.compute);
            }
        }
        if (result.threads > 0) {
            result.pack_mean_s /= static_cast<double>(result.threads);
            result.compute_mean_s /= static_cast<double>(result.threads);
        }
        g_profile = result;
    }
}

#else // not x86-64: no AVX2 kernel, use the plain loop

bool packed_gemm_supported() {
    return false;
}

CacheSizes cpu_cache_sizes() {
    return {};
}

PackedGemmProfile packed_gemm_profile() {
    return {};
}

PackedGemmBlocking packed_gemm_blocking() {
    return {};
}

bool packed_microkernel_tile(int, std::int64_t&, std::int64_t&) {
    return false;
}

bool run_packed_microkernel(int, std::int64_t, const float*, const float*, float*, std::int64_t,
                            bool) {
    return false;
}

void set_packed_gemm_blocking(PackedGemmBlocking) {}

void matmul_f32_packed_strided(const float* a, std::int64_t rsa, std::int64_t csa, const float* b,
                               std::int64_t rsb, std::int64_t csb, float* c, std::int64_t ldc,
                               std::int64_t m, std::int64_t n, std::int64_t k) {
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        for (std::int64_t i = 0; i < m; ++i) {
            std::fill(c + i * ldc, c + i * ldc + n, 0.0f);
        }
        return;
    }
    matmul_strided_fallback(a, rsa, csa, b, rsb, csb, c, ldc, m, n, k, false);
}

void matmul_f32_packed_parallel_strided(const float* a, std::int64_t rsa, std::int64_t csa,
                                        const float* b, std::int64_t rsb, std::int64_t csb, float* c,
                                        std::int64_t ldc, std::int64_t m, std::int64_t n,
                                        std::int64_t k) {
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        matmul_f32_packed_strided(a, rsa, csa, b, rsb, csb, c, ldc, m, n, k);
        return;
    }
    matmul_strided_fallback(a, rsa, csa, b, rsb, csb, c, ldc, m, n, k, true);
}

#endif

// Contiguous row-major operands: the layout most callers have.
void matmul_f32_packed(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                       std::int64_t k) {
    matmul_f32_packed_strided(a, k, 1, b, n, 1, c, n, m, n, k);
}

void matmul_f32_packed_parallel(const float* a, const float* b, float* c, std::int64_t m,
                                std::int64_t n, std::int64_t k) {
    matmul_f32_packed_parallel_strided(a, k, 1, b, n, 1, c, n, m, n, k);
}

} // namespace mensurflow::cpu
