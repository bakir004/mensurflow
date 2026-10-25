// CPU AXPY kernel called by the CPU adapter after common input validation.
// Each independent iteration updates y[i] = alpha * x[i] + y[i].
#include <cstddef>

namespace prs::detail {
void cpu_axpy(float alpha, const float* x, float* y, std::size_t count) {
    // CMake enables OpenMP when available. Small vectors use a single thread to
    // avoid parallel overhead; builds without OpenMP retain the ordinary loop.
#ifdef _OPENMP
#pragma omp parallel for if (count >= 16384)
#endif
    for (std::size_t i = 0; i < count; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}
} // namespace prs::detail
