// CPU BLAS kernels.
#include "backends/cpu/blas.hpp"

#include "backends/cpu/parallel.hpp"

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

} // namespace mensurflow::cpu
