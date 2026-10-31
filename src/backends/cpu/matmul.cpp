// Cache-blocked, row-parallel matrix multiplication.
#include "backends/cpu/matmul.hpp"

#include "backends/cpu/parallel.hpp"

#include <algorithm>
#include <stdexcept>

namespace mensurflow::cpu {

namespace {

// Block sizes, in elements. A block of B (kDepth x kCols floats = 128 KB) is reused by every row
// handled by the calling thread and fits the 256 KB L2 cache of a typical core; the kCols floats
// of one row of C (1 KB) stay in the L1 cache while the depth loop runs over them.
constexpr std::int64_t kDepth = 128;
constexpr std::int64_t kCols = 256;

// Below this many multiply-adds the work is too small to share between threads.
constexpr double kParallelWork = 262144.0;

// Computes rows [row_begin, row_end) of C. Blocks over columns, then depth; both ascend, so each
// C[i,j] accumulates A[i,p] * B[p,j] in increasing p, exactly like the sequential loop.
void multiply_rows(const float* a, const float* b, float* c, std::int64_t row_begin,
                   std::int64_t row_end, std::int64_t n, std::int64_t k) {
    for (std::int64_t i = row_begin; i < row_end; ++i) {
        std::fill(c + i * n, c + (i + 1) * n, 0.0f);
    }
    for (std::int64_t j0 = 0; j0 < n; j0 += kCols) {
        const std::int64_t cols = std::min(kCols, n - j0);
        for (std::int64_t p0 = 0; p0 < k; p0 += kDepth) {
            const std::int64_t depth = std::min(kDepth, k - p0);
            for (std::int64_t i = row_begin; i < row_end; ++i) {
                float* ci = c + i * n + j0;
                const float* ai = a + i * k + p0;
                for (std::int64_t p = 0; p < depth; ++p) {
                    const float aip = ai[p];
                    const float* bp = b + (p0 + p) * n + j0;
                    for (std::int64_t j = 0; j < cols; ++j) {
                        ci[j] += aip * bp[j];
                    }
                }
            }
        }
    }
}

} // namespace

void matmul_f32(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                std::int64_t k) {
    if (m < 0 || n < 0 || k < 0) {
        throw std::invalid_argument("matmul: negative dimension");
    }
    if (m == 0 || n == 0) {
        return;
    }
    if (c == nullptr || (k > 0 && (a == nullptr || b == nullptr))) {
        throw std::invalid_argument("matmul: null pointer");
    }
    // The index being split is a row of C, so the grain is in rows: a grain of m keeps small
    // products serial, 0 shares anything else. align = 1 because one row is already a lot of work.
    const double work = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const std::int64_t grain = work < kParallelWork ? m : 0;
    parallel_for(
        m,
        [=](std::int64_t row_begin, std::int64_t row_end) {
            multiply_rows(a, b, c, row_begin, row_end, n, k);
        },
        grain, 1);
}

} // namespace mensurflow::cpu
