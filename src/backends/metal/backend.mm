// Adapt the Metal operations in axpy.mm and matmul.mm to BackendInterface.
// Objective-C++ keeps Apple API details out of the common C++ dispatcher.
#include "../interface.hpp"
#import <Metal/Metal.h>

namespace prs::detail {
void metal_axpy(float, const float*, float*, std::size_t);

void metal_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

// Give this adapter a translation-unit-local name, independent of other backends.
namespace {
class Implementation final : public BackendInterface {
  public:
    bool available() const override {
        // Building Metal support does not guarantee a usable runtime device.
        @autoreleasepool {
            return MTLCreateSystemDefaultDevice() != nil;
        }
    }

    void axpy(float alpha, const float* x, float* y, std::size_t count) const override {
        metal_axpy(alpha, x, y, count);
    }

    void matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                std::size_t k) const override {
        metal_matmul(a, b, c, m, n, k);
    }
};
} // namespace

const BackendInterface& metal_backend() {
    // Reuse the stateless adapter and keep its returned reference valid.
    static Implementation instance;
    return instance;
}
} // namespace prs::detail
