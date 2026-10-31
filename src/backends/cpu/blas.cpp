// CPU BLAS kernels.
#include "backends/cpu/blas.hpp"

#include "backends/cpu/parallel.hpp"

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

} // namespace mensurflow::cpu
