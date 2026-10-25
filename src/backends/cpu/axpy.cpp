// CPU AXPY loop used by cpu/backend.cpp after the public API validates sizes.
// Each iteration updates one independent element of the caller's output buffer.
#include <cstddef>
namespace prs::detail {
void cpu_axpy(float alpha, const float* x, float* y, std::size_t count) {
// _OPENMP is supplied by the compiler when OpenMP support is enabled by CMake.
#ifdef _OPENMP
// Small inputs stay serial to avoid parallel-region overhead.
#pragma omp parallel for if(count >= 16384)
#endif
    for (std::size_t i = 0; i < count; ++i) y[i] = alpha * x[i] + y[i];
}
}
