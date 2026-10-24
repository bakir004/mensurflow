#include "prs/axpy.hpp"

#include <cstddef>
#include <stdexcept>

namespace prs {

#ifdef PRS_HAS_CUDA
void axpy_cuda(float alpha, const float* x, float* y, std::size_t n);
bool cuda_device_available();
#endif

#ifdef PRS_HAS_METAL
void axpy_metal(float alpha, const float* x, float* y, std::size_t n);
bool metal_device_available();
#endif

bool metal_available() {
#ifdef PRS_HAS_METAL
    return metal_device_available();
#else
    return false;
#endif
}

bool cuda_available() {
#ifdef PRS_HAS_CUDA
    return cuda_device_available();
#else
    return false;
#endif
}

void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y,
          Backend backend) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("AXPY: x and y must have equal sizes");
    }

    switch (backend) {
    case Backend::CPU:
        // Independent elements: distribute iterations across CPU threads when
        // OpenMP is available. Small vectors avoid thread-launch overhead.
#pragma omp parallel for if(x.size() >= 16384)
        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] = alpha * x[i] + y[i];
        }
        return;
    case Backend::CUDA:
#ifdef PRS_HAS_CUDA
        axpy_cuda(alpha, x.data(), y.data(), x.size());
        return;
#else
        throw std::runtime_error("AXPY: CUDA support was not built");
#endif
    case Backend::Metal:
#ifdef PRS_HAS_METAL
        axpy_metal(alpha, x.data(), y.data(), x.size());
        return;
#else
        throw std::runtime_error("AXPY: Metal support was not built");
#endif
    }
    throw std::invalid_argument("AXPY: unknown backend");
}

} // namespace prs
