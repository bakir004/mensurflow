// Adapts the CUDA operations to the common BackendInterface in interface.hpp.
// Kernel implementations live in axpy.cu and matmul.cu.
#include "../interface.hpp"
#include <cuda_runtime.h>

namespace prs::detail {
// CUDA support in the build does not guarantee a usable device at runtime.
bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
void cuda_axpy(float, const float*, float*, std::size_t);
void cuda_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
// Keep this adapter class local to this translation unit. Other backends can
// use the same class name without referring to the same type.
namespace {
class Implementation final : public BackendInterface {
public:
    bool available() const override { return cuda_device_available(); }
    void axpy(float alpha, const float* x, float* y, std::size_t count) const override {
        cuda_axpy(alpha, x, y, count);
    }
    void matmul(const float* a, const float* b, float* c,
                std::size_t m, std::size_t n, std::size_t k) const override {
        cuda_matmul(a, b, c, m, n, k);
    }
};
}
const BackendInterface& cuda_backend() {
    // Reuse one adapter and return it through the common interface without a copy.
    // Static lifetime keeps the returned reference valid after this call ends.
    static Implementation instance;
    return instance;
}
} // namespace prs::detail
