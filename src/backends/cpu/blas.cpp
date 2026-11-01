// CPU BLAS kernels.
#include "backends/cpu/blas.hpp"

#include "backends/cpu/parallel.hpp"

#include <cmath>
#include <stdexcept>

namespace mensurflow::cpu {

float dot_f32(const float* x, const float* y, std::int64_t n) {
    const double total = parallel_sum(n, [=](std::int64_t begin, std::int64_t end) {
        double sum = 0.0;
        for (std::int64_t i = begin; i < end; ++i) {
            sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
        }
        return sum;
    });
    return static_cast<float>(total);
}

void scal_f32(float alpha, const float* x, float* out, std::int64_t n) {
    if (n < 0) {
        throw std::invalid_argument("scal: negative length");
    }
    if (n == 0) {
        return;
    }
    if (x == nullptr || out == nullptr) {
        throw std::invalid_argument("scal: null pointer");
    }
    parallel_for(n, [=](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            out[i] = alpha * x[i];
        }
    });
}

float asum_f32(const float* x, std::int64_t n) {
    if (n < 0) {
        throw std::invalid_argument("asum: negative length");
    }
    if (n > 0 && x == nullptr) {
        throw std::invalid_argument("asum: null pointer");
    }
    return static_cast<float>(parallel_sum(n, [x](std::int64_t begin, std::int64_t end) {
        double result = 0.0;
        for (std::int64_t i = begin; i < end; ++i) {
            result += std::fabs(static_cast<double>(x[i]));
        }
        return result;
    }));
}

float nrm2_f32(const float* x, std::int64_t n) {
    if (n < 0) {
        throw std::invalid_argument("nrm2: negative length");
    }
    if (n > 0 && x == nullptr) {
        throw std::invalid_argument("nrm2: null pointer");
    }
    const double squared = parallel_sum(n, [x](std::int64_t begin, std::int64_t end) {
        double result = 0.0;
        for (std::int64_t i = begin; i < end; ++i) {
            const double value = static_cast<double>(x[i]);
            result += value * value;
        }
        return result;
    });
    return static_cast<float>(std::sqrt(squared));
}

void matvec_f32(const float* a, const float* x, float* y, std::int64_t m, std::int64_t k) {
    if (m < 0 || k < 0) {
        throw std::invalid_argument("matvec: negative dimension");
    }
    if (m == 0) {
        return;
    }
    if (y == nullptr || (k > 0 && (a == nullptr || x == nullptr))) {
        throw std::invalid_argument("matvec: null pointer");
    }
    const double work = static_cast<double>(m) * static_cast<double>(k);
    const std::int64_t grain = work < 262144.0 ? m : 0;
    parallel_for(
        m,
        [=](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                double sum = 0.0;
                for (std::int64_t j = 0; j < k; ++j) {
                    sum += static_cast<double>(a[i * k + j]) * static_cast<double>(x[j]);
                }
                y[i] = static_cast<float>(sum);
            }
        },
        grain, 1);
}

void gemv_f32(const float* a, const float* x, float* y, std::int64_t rows, std::int64_t cols,
              float alpha, float beta, const float* initial, bool transpose) {
    const std::int64_t output = transpose ? cols : rows;
    const std::int64_t input = transpose ? rows : cols;
    if (rows < 0 || cols < 0 || output < 0 || input < 0) {
        throw std::invalid_argument("gemv: negative dimension");
    }
    if (output == 0) {
        return;
    }
    if (y == nullptr || (input > 0 && (a == nullptr || x == nullptr)) ||
        (beta != 0.0f && initial == nullptr)) {
        throw std::invalid_argument("gemv: null pointer");
    }
    parallel_for(output, [=](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            double total = 0.0;
            for (std::int64_t j = 0; j < input; ++j) {
                const float value = transpose ? a[j * cols + i] : a[i * cols + j];
                total += static_cast<double>(value) * static_cast<double>(x[j]);
            }
            y[i] = static_cast<float>(alpha * total +
                                      (beta == 0.0f ? 0.0 : beta * initial[i]));
        }
    });
}

} // namespace mensurflow::cpu
