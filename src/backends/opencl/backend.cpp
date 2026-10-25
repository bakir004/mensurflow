// Adapt the OpenCL operations in axpy.cpp and matmul.cpp to BackendInterface.
// The common dispatcher uses this adapter without depending on OpenCL types.
#include "../interface.hpp"
#include "utils.hpp"

namespace prs::detail {
void opencl_axpy(float, const float*, float*, std::size_t);

void opencl_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

// Keep this adapter's class name local to this translation unit so each
// backend can have its own Implementation class.
namespace {
class Implementation final : public BackendInterface {
  public:
    bool available() const override {
        // Building OpenCL support does not guarantee a usable runtime GPU device.
        return opencl_device().found;
    }

    void axpy(float alpha, const float* x, float* y, std::size_t count) const override {
        opencl_axpy(alpha, x, y, count);
    }

    void matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                std::size_t k) const override {
        opencl_matmul(a, b, c, m, n, k);
    }
};
} // namespace

const BackendInterface& opencl_backend() {
    // The returned reference remains valid for the lifetime of the program.
    // GPU buffers are still allocated separately by each operation.
    static Implementation instance;
    return instance;
}
} // namespace prs::detail