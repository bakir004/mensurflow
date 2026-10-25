#include <cstddef>
namespace prs::detail {
void cpu_axpy(float alpha, const float* x, float* y, std::size_t count) {
#ifdef _OPENMP
#pragma omp parallel for if(count >= 16384)
#endif
    for (std::size_t i = 0; i < count; ++i) y[i] = alpha * x[i] + y[i];
}
}
