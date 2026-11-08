// CPU BLAS kernels: one loop per operation. MENSURFLOW_PARALLEL_FOR (threads.hpp) splits the loop
// across threads when the build has OpenMP; its `if` keeps small inputs on one thread. matmul is
// the exception: its parallel version is a different, cache-blocked loop, so it has two.
#include "backends/cpu/ops/blas/blas.hpp"

#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cmath>

namespace mensurflow::cpu {

float dot_f32(const float* x, const float* y, std::int64_t n) {
    double sum = 0.0;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold) reduction(+ : sum))
    for (std::int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
    }
    return static_cast<float>(sum);
}

void scal_f32(float alpha, const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = alpha * x[i];
    }
}

float asum_f32(const float* x, std::int64_t n) {
    double sum = 0.0;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold) reduction(+ : sum))
    for (std::int64_t i = 0; i < n; ++i) {
        sum += std::fabs(static_cast<double>(x[i]));
    }
    return static_cast<float>(sum);
}

float nrm2_f32(const float* x, std::int64_t n) {
    double sum = 0.0;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold) reduction(+ : sum))
    for (std::int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    return static_cast<float>(std::sqrt(sum));
}

void copy_f32(const float* x, float* out, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = x[i];
    }
}

void matvec_f32(const float* a, const float* x, float* y, std::int64_t m, std::int64_t k) {
    MENSURFLOW_PARALLEL_FOR(if (m * k > parallel_work_threshold))
    for (std::int64_t i = 0; i < m; ++i) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < k; ++j) {
            sum += static_cast<double>(a[i * k + j]) * static_cast<double>(x[j]);
        }
        y[i] = static_cast<float>(sum);
    }
}

void gemv_f32(const float* a, const float* x, float* y, std::int64_t rows, std::int64_t cols,
              float alpha, float beta, const float* initial, bool transpose) {
    const std::int64_t outputs = transpose ? cols : rows;
    const std::int64_t inputs = transpose ? rows : cols;
    MENSURFLOW_PARALLEL_FOR(if (rows * cols > parallel_work_threshold))
    for (std::int64_t i = 0; i < outputs; ++i) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < inputs; ++j) {
            const float value = transpose ? a[j * cols + i] : a[i * cols + j];
            sum += static_cast<double>(value) * static_cast<double>(x[j]);
        }
        y[i] = static_cast<float>(alpha * sum + (beta == 0.0f ? 0.0 : beta * initial[i]));
    }
}

// One row of out per iteration; the inner loop is a contiguous axpy the compiler vectorizes.
void ger_f32(float alpha, const float* x, const float* y, const float* a, float* out,
             std::int64_t m, std::int64_t n) {
    MENSURFLOW_PARALLEL_FOR(if (m * n > parallel_threshold))
    for (std::int64_t i = 0; i < m; ++i) {
        const float scale = alpha * x[i];
        float* row = out + i * n;
        if (a == nullptr) {
            for (std::int64_t j = 0; j < n; ++j) {
                row[j] = scale * y[j];
            }
        } else {
            const float* initial = a + i * n;
            for (std::int64_t j = 0; j < n; ++j) {
                row[j] = scale * y[j] + initial[j];
            }
        }
    }
}

// The plain i-k-j loop: for each row of A, every A[i,p] scales row p of B into row i of C.
void matmul_f32_sequential(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                           std::int64_t k) {
    for (std::int64_t i = 0; i < m; ++i) {
        float* ci = c + i * n;
        for (std::int64_t j = 0; j < n; ++j) {
            ci[j] = 0.0f;
        }
        for (std::int64_t p = 0; p < k; ++p) {
            const float aip = a[i * k + p];
            const float* bp = b + p * n;
            for (std::int64_t j = 0; j < n; ++j) {
                ci[j] += aip * bp[j];
            }
        }
    }
}

// The same i-k-j products, but each thread takes one contiguous band of rows of C and walks B in
// blocks: a block of B (128 x 256 floats = 128 KB) fits a typical 256 KB L2 cache and is reused by
// every row of the band, and the 256 floats of one row of C stay in L1 while the depth loop runs.
void matmul_f32_parallel(const float* a, const float* b, float* c, std::int64_t m, std::int64_t n,
                         std::int64_t k) {
    constexpr std::int64_t block_depth = 128;
    constexpr std::int64_t block_cols = 256;
    const bool parallel = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) >
                          static_cast<double>(parallel_work_threshold);
    const std::int64_t threads = parallel ? max_threads() : 1;
    const std::int64_t band_rows = std::max<std::int64_t>((m + threads - 1) / threads, 1);
    const std::int64_t bands = (m + band_rows - 1) / band_rows;
    MENSURFLOW_PARALLEL_FOR(if (parallel))
    for (std::int64_t band = 0; band < bands; ++band) {
        const std::int64_t row_begin = band * band_rows;
        const std::int64_t row_end = std::min(row_begin + band_rows, m);
        for (std::int64_t i = row_begin; i < row_end; ++i) {
            for (std::int64_t j = 0; j < n; ++j) {
                c[i * n + j] = 0.0f;
            }
        }
        for (std::int64_t j0 = 0; j0 < n; j0 += block_cols) {
            const std::int64_t cols = std::min(block_cols, n - j0);
            for (std::int64_t p0 = 0; p0 < k; p0 += block_depth) {
                const std::int64_t depth = std::min(block_depth, k - p0);
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
}

} // namespace mensurflow::cpu
