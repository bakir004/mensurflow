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
#include "backends/cpu/ops/blas/gemm_packed.hpp"

#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/ops/blas/gemm_pack.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>

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
void pack_a(bool scalar, std::int64_t mr, const float* a, std::int64_t lda, std::int64_t mc,
            std::int64_t kc, float* dst) {
    if (scalar) {
        detail::pack_a_scalar(mr, a, lda, mc, kc, dst);
    } else {
        detail::pack_a(mr, a, lda, mc, kc, dst);
    }
}

void pack_b(bool scalar, std::int64_t nr, const float* b, std::int64_t ldb, std::int64_t kc,
            std::int64_t nc, float* dst) {
    if (scalar) {
        detail::pack_b_scalar(nr, b, ldb, kc, nc, dst);
    } else {
        detail::pack_b(nr, b, ldb, kc, nc, dst);
    }
}

#ifdef MENSURFLOW_PACKED_GEMM

namespace {

constexpr std::int64_t MR6 = 6;
constexpr std::int64_t NR6 = 16;

// Multiply-adds a thread needs before it is worth waking (see matmul_f32_packed_parallel).
constexpr double min_macs_per_thread = 512.0 * 1024.0;

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
    return blocking;
}

PackedGemmBlocking& blocking_state() {
    static PackedGemmBlocking state = initial_blocking();
    return state;
}


// C[6,16] (row stride ldc) = or += sum over p < kc of pa[p*6 + r] * pb[p*16 + j].
// Prefetch: 0 none; 1 also prefetch the C tile before the loop (it is read and written after it);
// 2 also prefetch the packed A and B panels PrefetchSteps depth steps ahead inside the loop.
template <int Prefetch>
MENSURFLOW_TARGET_AVX2 void kernel_6x16(std::int64_t kc, const float* pa, const float* pb,
                                        float* c, std::int64_t ldc, bool accumulate) {
    if constexpr (Prefetch >= 1) {
        for (int r = 0; r < 6; ++r) {
            _mm_prefetch(reinterpret_cast<const char*>(c + r * ldc), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(c + r * ldc + 15), _MM_HINT_T0);
        }
    }
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (std::int64_t p = 0; p < kc; ++p) {
        if constexpr (Prefetch >= 2) {
            _mm_prefetch(reinterpret_cast<const char*>(pb + NR6 * prefetch_steps), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pa + MR6 * prefetch_steps), _MM_HINT_T0);
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
    default:
        switch (blocking.prefetch) {
        case 1:
            return {6, 16, &kernel_6x16<1>};
        case 2:
            return {6, 16, &kernel_6x16<2>};
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

void set_packed_gemm_blocking(PackedGemmBlocking blocking) {
    if (blocking.mc <= 0 || blocking.kc <= 0 || blocking.nc <= 0 || blocking.tiles_per_thread <= 0 ||
        blocking.prefetch < 0 || blocking.prefetch > 2 || blocking.shape < 0 || blocking.shape > 3 ||
        blocking.scalar_pack < 0 || blocking.scalar_pack > 1) {
        throw std::invalid_argument("packed gemm blocking: mc, kc, nc and tiles_per_thread must be "
                                    "positive, prefetch 0-2, shape 0-3");
    }
    blocking_state() = blocking;
}

void matmul_f32_packed(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                       std::int64_t k) {
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        std::fill(c, c + m * n, 0.0f);
        return;
    }
    if (!packed_gemm_supported()) {
        matmul_f32_sequential(a, b, c, m, n, k);
        return;
    }

    const Shape shape = select_shape(blocking_state());
    const std::int64_t MR = shape.mr;
    const std::int64_t NR = shape.nr;
    const KernelFn kernel = shape.kernel;
    const bool scalar_pack = blocking_state().scalar_pack != 0;
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
    packed_a.reserve(static_cast<std::size_t>(mc_max * kc_max));
    ScratchBuffer& packed_b = scratch_b();
    packed_b.reserve(static_cast<std::size_t>(kc_max * nc_max));
    alignas(64) float edge[max_tile_floats];

    for (std::int64_t jc = 0; jc < n; jc += NC) {
        const std::int64_t nc = std::min(NC, n - jc);
        for (std::int64_t pc = 0; pc < k; pc += KC) {
            const std::int64_t kc = std::min(KC, k - pc);
            const bool accumulate = pc > 0;
            pack_b(scalar_pack, NR, b + pc * n + jc, n, kc, nc, packed_b.data);
            for (std::int64_t ic = 0; ic < m; ic += MC) {
                const std::int64_t mc = std::min(MC, m - ic);
                pack_a(scalar_pack, MR, a + ic * k + pc, k, mc, kc, packed_a.data);
                for (std::int64_t jr = 0; jr < nc; jr += NR) {
                    const std::int64_t cols = std::min(NR, nc - jr);
                    const float* pb = packed_b.data + (jr / NR) * kc * NR;
                    for (std::int64_t ir = 0; ir < mc; ir += MR) {
                        const std::int64_t rows = std::min(MR, mc - ir);
                        const float* pa = packed_a.data + (ir / MR) * kc * MR;
                        float* ctile = c + (ic + ir) * n + jc + jr;
                        if (rows == MR && cols == NR) {
                            kernel(kc, pa, pb, ctile, n, accumulate);
                        } else {
                            // Edge tile: compute a full tile into scratch, copy the valid part.
                            kernel(kc, pa, pb, edge, NR, false);
                            for (std::int64_t r = 0; r < rows; ++r) {
                                for (std::int64_t q = 0; q < cols; ++q) {
                                    if (accumulate) {
                                        ctile[r * n + q] += edge[r * NR + q];
                                    } else {
                                        ctile[r * n + q] = edge[r * NR + q];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// The same five loops, with the work split across threads. Per (jc, pc) step:
//   1. one parallel loop packs every 16-column panel of B and every 6-row panel of A (all m rows),
//   2. one parallel loop over tiles of C: a tile is one MC-row block times a group of B panels. A
//      thread reads the shared packed A and B and writes only its own tile of C, so tiles need no
//      locking. Dynamic scheduling evens out uneven tiles (edges, busy cores).
// Tile groups are sized so there are about 4 tiles per thread, which is also why small matrices
// (a single MC block of rows) split along the columns instead.
void matmul_f32_packed_parallel(const float* a, const float* b, float* c, std::int64_t m,
                                std::int64_t n, std::int64_t k) {
    // Small products get fewer threads: each thread should have at least min_macs_per_thread
    // multiply-adds, or waking and joining it costs more than it saves (measured: n = 128 is
    // fastest on 4 of 24 threads).
    const double macs = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const std::int64_t threads = std::min<std::int64_t>(
        max_threads(), std::max<std::int64_t>(1, static_cast<std::int64_t>(macs / min_macs_per_thread)));
    const bool worth = threads > 1 && macs > static_cast<double>(parallel_work_threshold);
    if (!worth || !packed_gemm_supported()) {
        matmul_f32_packed(a, b, c, m, n, k); // also handles m, n or k == 0
        return;
    }

    const Shape shape = select_shape(blocking_state());
    const std::int64_t MR = shape.mr;
    const std::int64_t NR = shape.nr;
    const KernelFn kernel = shape.kernel;
    const bool scalar_pack = blocking_state().scalar_pack != 0;
    // Block sizes are rounded down to whole tiles (at least one).
    const std::int64_t MC = std::max(MR, blocking_state().mc / MR * MR);
    const std::int64_t KC = blocking_state().kc;
    const std::int64_t NC = std::max(NR, blocking_state().nc / NR * NR);
    const auto round_up = [](std::int64_t value, std::int64_t multiple) {
        return (value + multiple - 1) / multiple * multiple;
    };
    const auto ceil_div = [](std::int64_t value, std::int64_t divisor) {
        return (value + divisor - 1) / divisor;
    };
    const std::int64_t m_padded = round_up(m, MR);
    const std::int64_t kc_max = std::min(KC, k);
    const std::int64_t nc_max = std::min(NC, round_up(n, NR));
    ScratchBuffer& packed_a = scratch_a();
    packed_a.reserve(static_cast<std::size_t>(m_padded * kc_max));
    ScratchBuffer& packed_b = scratch_b();
    packed_b.reserve(static_cast<std::size_t>(kc_max * nc_max));
    float* const pa_base = packed_a.data;
    float* const pb_base = packed_b.data;
    const std::int64_t a_panels = m_padded / MR;
    const std::int64_t ic_blocks = ceil_div(m, MC);

    for (std::int64_t jc = 0; jc < n; jc += NC) {
        const std::int64_t nc = std::min(NC, n - jc);
        const std::int64_t b_panels = ceil_div(nc, NR);
        const std::int64_t wanted_groups = ceil_div(blocking_state().tiles_per_thread * threads, ic_blocks);
        const std::int64_t panels_per_group =
            ceil_div(b_panels, std::min(std::max<std::int64_t>(wanted_groups, 1), b_panels));
        const std::int64_t groups = ceil_div(b_panels, panels_per_group);
        const std::int64_t tiles = ic_blocks * groups;

        for (std::int64_t pc = 0; pc < k; pc += KC) {
            const std::int64_t kc = std::min(KC, k - pc);
            const bool accumulate = pc > 0;

            MENSURFLOW_PARALLEL_FOR(schedule(static) num_threads(static_cast<int>(threads)))
            for (std::int64_t t = 0; t < b_panels + a_panels; ++t) {
                if (t < b_panels) {
                    const std::int64_t j = t * NR;
                    pack_b(scalar_pack, NR, b + pc * n + jc + j, n, kc, std::min(NR, nc - j), pb_base + t * kc * NR);
                } else {
                    const std::int64_t i = (t - b_panels) * MR;
                    pack_a(scalar_pack, MR, a + i * k + pc, k, std::min(MR, m - i), kc, pa_base + (t - b_panels) * kc * MR);
                }
            }

            MENSURFLOW_PARALLEL_FOR(schedule(dynamic) num_threads(static_cast<int>(threads)))
            for (std::int64_t t = 0; t < tiles; ++t) {
                alignas(64) float edge[max_tile_floats];
                const std::int64_t ic = (t % ic_blocks) * MC;
                const std::int64_t mc = std::min(MC, m - ic);
                const std::int64_t first_panel = (t / ic_blocks) * panels_per_group;
                const std::int64_t last_panel = std::min(first_panel + panels_per_group, b_panels);
                for (std::int64_t jp = first_panel; jp < last_panel; ++jp) {
                    const std::int64_t jr = jp * NR;
                    const std::int64_t cols = std::min(NR, nc - jr);
                    const float* pb = pb_base + jp * kc * NR;
                    for (std::int64_t ir = 0; ir < mc; ir += MR) {
                        const std::int64_t rows = std::min(MR, mc - ir);
                        const float* pa = pa_base + ((ic + ir) / MR) * kc * MR;
                        float* ctile = c + (ic + ir) * n + jc + jr;
                        if (rows == MR && cols == NR) {
                            kernel(kc, pa, pb, ctile, n, accumulate);
                        } else {
                            kernel(kc, pa, pb, edge, NR, false);
                            for (std::int64_t r = 0; r < rows; ++r) {
                                for (std::int64_t q = 0; q < cols; ++q) {
                                    if (accumulate) {
                                        ctile[r * n + q] += edge[r * NR + q];
                                    } else {
                                        ctile[r * n + q] = edge[r * NR + q];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#else // not x86-64: no AVX2 kernel, use the plain loop
CacheSizes cpu_cache_sizes() {    return {};}

bool packed_gemm_supported() {
    return false;
}

CacheSizes cpu_cache_sizes() {
    return {cache_bytes(1), cache_bytes(2), cache_bytes(3)};
}

PackedGemmBlocking packed_gemm_blocking() {
    return {};
}

void set_packed_gemm_blocking(PackedGemmBlocking) {}

void matmul_f32_packed(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                       std::int64_t k) {
    matmul_f32_sequential(a, b, c, m, n, k);
}

void matmul_f32_packed_parallel(const float* a, const float* b, float* c, std::int64_t m,
                                std::int64_t n, std::int64_t k) {
    matmul_f32_parallel(a, b, c, m, n, k);
}

#endif

} // namespace mensurflow::cpu
