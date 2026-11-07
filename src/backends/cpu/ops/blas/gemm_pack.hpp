// Packing for the packed GEMM (gemm_packed.cpp): copies a block of A or B into the contiguous panels
// the micro-kernel reads in order.
//
//   A panel (MR rows): for each depth step p, MR floats  A[i0..i0+MR-1][p]   (a transpose)
//   B panel (NR cols): for each depth step p, NR floats  B[p][j0..j0+NR-1]   (a row copy)
//
// A panel with fewer than MR rows (or NR columns) is padded with zeros so the kernel never branches.
// The scalar versions are the reference and the fallback; the AVX2 versions are written by hand and
// are chosen automatically for the default shape (MR = 6) and for NR = 8, 16, 24.
#pragma once

#include <algorithm>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#define MENSURFLOW_PACK_AVX2 1
#include <immintrin.h>
#endif

#ifndef MENSURFLOW_TARGET_AVX2
#if defined(_MSC_VER)
#define MENSURFLOW_TARGET_AVX2
#else
#define MENSURFLOW_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif
#endif

namespace mensurflow::cpu::detail {

// ---- Scalar reference ---------------------------------------------------------------------------

// Copies A[0..mc, 0..kc) (row stride lda) into mr-row panels, padding the last panel with zeros.
inline void pack_a_scalar(std::int64_t mr, const float* a, std::int64_t lda, std::int64_t mc,
                          std::int64_t kc, float* dst) {
    for (std::int64_t i = 0; i < mc; i += mr) {
        const std::int64_t rows = std::min(mr, mc - i);
        for (std::int64_t r = 0; r < rows; ++r) {
            const float* src = a + (i + r) * lda;
            for (std::int64_t p = 0; p < kc; ++p) {
                dst[p * mr + r] = src[p];
            }
        }
        for (std::int64_t r = rows; r < mr; ++r) {
            for (std::int64_t p = 0; p < kc; ++p) {
                dst[p * mr + r] = 0.0f;
            }
        }
        dst += kc * mr;
    }
}

// Copies B[0..kc, 0..nc) (row stride ldb) into nr-column panels, padding the last with zeros.
inline void pack_b_scalar(std::int64_t nr, const float* b, std::int64_t ldb, std::int64_t kc,
                          std::int64_t nc, float* dst) {
    for (std::int64_t j = 0; j < nc; j += nr) {
        const std::int64_t cols = std::min(nr, nc - j);
        for (std::int64_t p = 0; p < kc; ++p) {
            const float* src = b + p * ldb + j;
            float* out = dst + p * nr;
            std::int64_t q = 0;
            for (; q < cols; ++q) {
                out[q] = src[q];
            }
            for (; q < nr; ++q) {
                out[q] = 0.0f;
            }
        }
        dst += kc * nr;
    }
}

// The same with any strides: element (i, p) of A is a[i * rsa + p * csa]. This is what a transposed,
// sliced or padded view looks like; it is the fallback when neither stride is 1.
inline void pack_a_scalar(std::int64_t mr, const float* a, std::int64_t rsa, std::int64_t csa,
                          std::int64_t mc, std::int64_t kc, float* dst) {
    for (std::int64_t i = 0; i < mc; i += mr) {
        const std::int64_t rows = std::min(mr, mc - i);
        for (std::int64_t p = 0; p < kc; ++p) {
            std::int64_t r = 0;
            for (; r < rows; ++r) {
                dst[p * mr + r] = a[(i + r) * rsa + p * csa];
            }
            for (; r < mr; ++r) {
                dst[p * mr + r] = 0.0f;
            }
        }
        dst += kc * mr;
    }
}

// Element (p, j) of B is b[p * rsb + j * csb].
inline void pack_b_scalar(std::int64_t nr, const float* b, std::int64_t rsb, std::int64_t csb,
                          std::int64_t kc, std::int64_t nc, float* dst) {
    for (std::int64_t j = 0; j < nc; j += nr) {
        const std::int64_t cols = std::min(nr, nc - j);
        for (std::int64_t p = 0; p < kc; ++p) {
            float* out = dst + p * nr;
            std::int64_t q = 0;
            for (; q < cols; ++q) {
                out[q] = b[p * rsb + (j + q) * csb];
            }
            for (; q < nr; ++q) {
                out[q] = 0.0f;
            }
        }
        dst += kc * nr;
    }
}

#ifdef MENSURFLOW_PACK_AVX2

// ---- AVX2 ---------------------------------------------------------------------------------------

// One panel of up to 6 rows of A: 8 depth steps at a time, an 8 x 8 transpose in registers.
//
// Register use: the 6 loaded rows (rows 6 and 7 are zero and fold away), then the unpack step
// produces 8 values t0..t7, the shuffle step 8 values s0..s7, the permute step the 8 columns. With
// each step consuming its inputs the peak is the full 16 ymm registers, no spills.
//
// Column q of the block (rows 0..5 of depth step p + q) goes to dst[(p + q) * 6 .. +5]. A 256-bit
// store writes 8 floats, so the two extra floats land in the next column's slot and are overwritten
// by the next store, except for the last column of the block, which is written as 4 + 2 floats so
// nothing beyond the panel is ever touched (panels are packed by different threads).
MENSURFLOW_TARGET_AVX2 inline void pack_a_panel6_avx2(const float* a, std::int64_t lda,
                                                      std::int64_t rows, std::int64_t kc,
                                                      float* dst) {
    std::int64_t p = 0;
    if (rows == 6) {
        const float* r0 = a;
        const float* r1 = a + lda;
        const float* r2 = a + 2 * lda;
        const float* r3 = a + 3 * lda;
        const float* r4 = a + 4 * lda;
        const float* r5 = a + 5 * lda;
        const __m256 zero = _mm256_setzero_ps();
        for (; p + 8 <= kc; p += 8) {
            const __m256 v0 = _mm256_loadu_ps(r0 + p);
            const __m256 v1 = _mm256_loadu_ps(r1 + p);
            const __m256 v2 = _mm256_loadu_ps(r2 + p);
            const __m256 v3 = _mm256_loadu_ps(r3 + p);
            const __m256 v4 = _mm256_loadu_ps(r4 + p);
            const __m256 v5 = _mm256_loadu_ps(r5 + p);

            const __m256 t0 = _mm256_unpacklo_ps(v0, v1);
            const __m256 t1 = _mm256_unpackhi_ps(v0, v1);
            const __m256 t2 = _mm256_unpacklo_ps(v2, v3);
            const __m256 t3 = _mm256_unpackhi_ps(v2, v3);
            const __m256 t4 = _mm256_unpacklo_ps(v4, v5);
            const __m256 t5 = _mm256_unpackhi_ps(v4, v5);

            const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
            const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
            const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            const __m256 s4 = _mm256_shuffle_ps(t4, zero, 0x44);
            const __m256 s5 = _mm256_shuffle_ps(t4, zero, 0xEE);
            const __m256 s6 = _mm256_shuffle_ps(t5, zero, 0x44);
            const __m256 s7 = _mm256_shuffle_ps(t5, zero, 0xEE);

            float* out = dst + p * 6;
            _mm256_storeu_ps(out + 0 * 6, _mm256_permute2f128_ps(s0, s4, 0x20)); // column 0
            _mm256_storeu_ps(out + 1 * 6, _mm256_permute2f128_ps(s1, s5, 0x20)); // column 1
            _mm256_storeu_ps(out + 2 * 6, _mm256_permute2f128_ps(s2, s6, 0x20)); // column 2
            _mm256_storeu_ps(out + 3 * 6, _mm256_permute2f128_ps(s3, s7, 0x20)); // column 3
            _mm256_storeu_ps(out + 4 * 6, _mm256_permute2f128_ps(s0, s4, 0x31)); // column 4
            _mm256_storeu_ps(out + 5 * 6, _mm256_permute2f128_ps(s1, s5, 0x31)); // column 5
            _mm256_storeu_ps(out + 6 * 6, _mm256_permute2f128_ps(s2, s6, 0x31)); // column 6
            const __m256 c7 = _mm256_permute2f128_ps(s3, s7, 0x31);              // column 7
            _mm_storeu_ps(out + 7 * 6, _mm256_castps256_ps128(c7));              // rows 0..3
            _mm_storel_pi(reinterpret_cast<__m64*>(out + 7 * 6 + 4),
                          _mm256_extractf128_ps(c7, 1));                         // rows 4, 5
        }
    }
    // Depth tail (kc % 8) and partial panels (fewer than 6 rows): scalar, with zero padding.
    for (; p < kc; ++p) {
        for (std::int64_t r = 0; r < 6; ++r) {
            dst[p * 6 + r] = r < rows ? a[r * lda + p] : 0.0f;
        }
    }
}


// One full panel of NV * 8 columns of B: 4 depth steps at a time, NV vectors per step, loads first
// and stores after (4 * NV <= 12 registers in flight).
template <int NV>
MENSURFLOW_TARGET_AVX2 inline void pack_b_panel_avx2(const float* b, std::int64_t ldb,
                                                     std::int64_t kc, float* dst) {
    constexpr int nr = NV * 8;
    std::int64_t p = 0;
    for (; p + 4 <= kc; p += 4) {
        __m256 x[4][NV];
        for (int u = 0; u < 4; ++u) {
            for (int v = 0; v < NV; ++v) {
                x[u][v] = _mm256_loadu_ps(b + (p + u) * ldb + v * 8);
            }
        }
        for (int u = 0; u < 4; ++u) {
            for (int v = 0; v < NV; ++v) {
                _mm256_storeu_ps(dst + (p + u) * nr + v * 8, x[u][v]);
            }
        }
    }
    for (; p < kc; ++p) {
        for (int v = 0; v < NV; ++v) {
            _mm256_storeu_ps(dst + p * nr + v * 8, _mm256_loadu_ps(b + p * ldb + v * 8));
        }
    }
}


// Transposes an 8 x 8 block held in eight registers: afterwards r_q holds what was column q.
// Three steps (unpack, shuffle, permute) that each consume their inputs, so it fits the 16 registers.
MENSURFLOW_TARGET_AVX2 inline void transpose8x8(__m256& r0, __m256& r1, __m256& r2, __m256& r3,
                                                __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    const __m256 t7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// A stored column by column (a transposed matrix): element (i, p) is a[i + p * csa], so the 6 rows of
// a panel for one depth step are 6 consecutive floats and packing is a plain copy, no transpose.
// Packs mc rows in 6-row panels. The depth steps are the outer loop and the panels the inner one, so
// each source column is read as one contiguous run (a single cache line per panel and step would be
// slow); the destination panels are written 24 bytes at a time, kc * 24 bytes apart.
MENSURFLOW_TARGET_AVX2 inline void pack_a_colmajor_avx2(const float* a, std::int64_t csa,
                                                        std::int64_t mc, std::int64_t kc,
                                                        float* dst) {
    const std::int64_t full = mc / 6;
    const __m128 zero = _mm_setzero_ps();
    for (std::int64_t p = 0; p < kc; ++p) {
        const float* src = a + p * csa;
        float* out = dst + p * 6;
        for (std::int64_t q = 0; q < full; ++q) {
            const float* s = src + q * 6;
            float* o = out + q * kc * 6;
            _mm_storeu_ps(o, _mm_loadu_ps(s));
            _mm_storel_pi(reinterpret_cast<__m64*>(o + 4),
                          _mm_loadl_pi(zero, reinterpret_cast<const __m64*>(s + 4)));
        }
    }
    const std::int64_t rest = mc - full * 6; // a last panel with fewer rows: zero padded
    if (rest > 0) {
        float* out = dst + full * kc * 6;
        for (std::int64_t p = 0; p < kc; ++p) {
            for (std::int64_t r = 0; r < 6; ++r) {
                out[p * 6 + r] = r < rest ? a[full * 6 + r + p * csa] : 0.0f;
            }
        }
    }
}

// B stored transposed (n rows of length k): element (p, j) is b[p + j * csb]. One full panel of
// NV * 8 columns: 8 depth steps at a time, each group of 8 columns is eight loads of 8 consecutive
// depth steps followed by an 8 x 8 transpose, whose rows are then the panel rows for those steps.
template <int NV>
MENSURFLOW_TARGET_AVX2 inline void pack_b_trans_panel_avx2(const float* b, std::int64_t csb,
                                                           std::int64_t kc, float* dst) {
    constexpr int nr = NV * 8;
    std::int64_t p = 0;
    for (; p + 8 <= kc; p += 8) {
        for (int v = 0; v < NV; ++v) {
            const float* col = b + static_cast<std::int64_t>(v) * 8 * csb + p;
            __m256 r0 = _mm256_loadu_ps(col + 0 * csb);
            __m256 r1 = _mm256_loadu_ps(col + 1 * csb);
            __m256 r2 = _mm256_loadu_ps(col + 2 * csb);
            __m256 r3 = _mm256_loadu_ps(col + 3 * csb);
            __m256 r4 = _mm256_loadu_ps(col + 4 * csb);
            __m256 r5 = _mm256_loadu_ps(col + 5 * csb);
            __m256 r6 = _mm256_loadu_ps(col + 6 * csb);
            __m256 r7 = _mm256_loadu_ps(col + 7 * csb);
            transpose8x8(r0, r1, r2, r3, r4, r5, r6, r7);
            float* out = dst + p * nr + v * 8;
            _mm256_storeu_ps(out + 0 * nr, r0);
            _mm256_storeu_ps(out + 1 * nr, r1);
            _mm256_storeu_ps(out + 2 * nr, r2);
            _mm256_storeu_ps(out + 3 * nr, r3);
            _mm256_storeu_ps(out + 4 * nr, r4);
            _mm256_storeu_ps(out + 5 * nr, r5);
            _mm256_storeu_ps(out + 6 * nr, r6);
            _mm256_storeu_ps(out + 7 * nr, r7);
        }
    }
    for (; p < kc; ++p) { // fewer than 8 depth steps left
        for (int j = 0; j < nr; ++j) {
            dst[p * nr + j] = b[p + static_cast<std::int64_t>(j) * csb];
        }
    }
}
#endif // MENSURFLOW_PACK_AVX2

// ---- Dispatch -----------------------------------------------------------------------------------
//
// Element (i, p) of A is a[i * rsa + p * csa] and element (p, j) of B is b[p * rsb + j * csb], so a
// row-major matrix has (rsa, csa) = (ld, 1), its transpose (1, ld), and a sliced or padded view any
// leading dimension. The fast paths below cover the unit-stride layouts; everything else is scalar.

inline void pack_a(std::int64_t mr, const float* a, std::int64_t rsa, std::int64_t csa,
                   std::int64_t mc, std::int64_t kc, float* dst) {
#ifdef MENSURFLOW_PACK_AVX2
    if (mr == 6) {
        if (csa == 1) { // rows are contiguous along the depth: transpose in registers
            for (std::int64_t i = 0; i < mc; i += 6) {
                pack_a_panel6_avx2(a + i * rsa, rsa, std::min<std::int64_t>(6, mc - i), kc, dst);
                dst += kc * 6;
            }
            return;
        }
        if (rsa == 1) { // the transposed layout: a panel's rows are contiguous
            pack_a_colmajor_avx2(a, csa, mc, kc, dst);
            return;
        }
    }
#endif
    pack_a_scalar(mr, a, rsa, csa, mc, kc, dst);
}

inline void pack_b(std::int64_t nr, const float* b, std::int64_t rsb, std::int64_t csb,
                   std::int64_t kc, std::int64_t nc, float* dst) {
#ifdef MENSURFLOW_PACK_AVX2
    if (nr == 8 || nr == 16 || nr == 24) {
        const bool rows = csb == 1; // a depth step is a contiguous run of columns: plain copy
        const bool transposed = !rows && rsb == 1; // columns are contiguous along the depth
        if (rows || transposed) {
            for (std::int64_t j = 0; j < nc; j += nr) {
                const std::int64_t cols = std::min(nr, nc - j);
                if (cols != nr) {
                    pack_b_scalar(nr, b + j * csb, rsb, csb, kc, cols, dst); // partial panel
                } else if (rows) {
                    if (nr == 8) {
                        pack_b_panel_avx2<1>(b + j, rsb, kc, dst);
                    } else if (nr == 16) {
                        pack_b_panel_avx2<2>(b + j, rsb, kc, dst);
                    } else {
                        pack_b_panel_avx2<3>(b + j, rsb, kc, dst);
                    }
                } else if (nr == 8) {
                    pack_b_trans_panel_avx2<1>(b + j * csb, csb, kc, dst);
                } else if (nr == 16) {
                    pack_b_trans_panel_avx2<2>(b + j * csb, csb, kc, dst);
                } else {
                    pack_b_trans_panel_avx2<3>(b + j * csb, csb, kc, dst);
                }
                dst += kc * nr;
            }
            return;
        }
    }
#endif
    pack_b_scalar(nr, b, rsb, csb, kc, nc, dst);
}

// Row-major matrices (leading dimension lda or ldb): the layout the first version handled.
inline void pack_a(std::int64_t mr, const float* a, std::int64_t lda, std::int64_t mc,
                   std::int64_t kc, float* dst) {
    pack_a(mr, a, lda, 1, mc, kc, dst);
}

inline void pack_b(std::int64_t nr, const float* b, std::int64_t ldb, std::int64_t kc,
                   std::int64_t nc, float* dst) {
    pack_b(nr, b, ldb, 1, kc, nc, dst);
}

} // namespace mensurflow::cpu::detail
