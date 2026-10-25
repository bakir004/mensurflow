#include "../interface.hpp"
#include <cuda_runtime.h>

namespace prs::detail {
bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
void cuda_axpy(float, const float*, float*, std::size_t);
void cuda_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
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
    static Implementation instance;
    return instance;
}
} // namespace prs::detail
