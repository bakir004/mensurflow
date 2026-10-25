// Adapts the Metal operations to the common BackendInterface in interface.hpp.
// Shader implementations and execution wrappers live in axpy.mm and matmul.mm.
#include "../interface.hpp"
#import <Metal/Metal.h>

namespace prs::detail {
// Check runtime device availability; the autorelease pool releases temporary
// Objective-C objects created while performing the check.
bool metal_device_available() {
    @autoreleasepool { return MTLCreateSystemDefaultDevice() != nil; }
}
void metal_axpy(float, const float*, float*, std::size_t);
void metal_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
// Keep this adapter class local so other backends may also name theirs Implementation.
namespace {
class Implementation final : public BackendInterface {
public:
    bool available() const override { return metal_device_available(); }
    void axpy(float alpha, const float* x, float* y, std::size_t count) const override {
        metal_axpy(alpha, x, y, count);
    }
    void matmul(const float* a, const float* b, float* c,
                std::size_t m, std::size_t n, std::size_t k) const override {
        metal_matmul(a, b, c, m, n, k);
    }
};
}
const BackendInterface& metal_backend() {
    // Reuse one adapter and expose it without copying or leaking its concrete type.
    // Static lifetime keeps the returned interface reference valid between calls.
    static Implementation instance;
    return instance;
}
} // namespace prs::detail
