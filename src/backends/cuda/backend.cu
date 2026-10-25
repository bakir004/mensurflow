// Adapt the CUDA operations in axpy.cu and matmul.cu to BackendInterface.
// The common dispatcher uses this adapter without depending on CUDA types.
#include "../interface.hpp"
#include <cuda_runtime.h>

namespace prs::detail {
void cuda_axpy(float, const float*, float*, std::size_t);

void cuda_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

// Keep this adapter's class name local to this translation unit so each
// backend can have its own Implementation class.
namespace {
class Implementation final : public BackendInterface {
  public:
    bool available() const override {
        // Runtime availability is separate from CMake finding the CUDA compiler.
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    void axpy(float alpha, const float* x, float* y, std::size_t count) const override {
        cuda_axpy(alpha, x, y, count);
    }

    void matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                std::size_t k) const override {
        cuda_matmul(a, b, c, m, n, k);
    }
};
} // namespace

const BackendInterface& cuda_backend() {
    // The returned reference remains valid for the lifetime of the program.
    // GPU buffers are still allocated separately by each operation.
    static Implementation instance;
    return instance;
}
} // namespace prs::detail
