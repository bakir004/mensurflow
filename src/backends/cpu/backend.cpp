// CPU adapter: connects BackendInterface to the loops in axpy.cpp and matmul.cpp.
// src/ops.cpp accesses this adapter through cpu_backend().
#include "../interface.hpp"

namespace prs::detail {
// Implementations are defined in neighboring source files and linked together.
void cpu_axpy(float, const float*, float*, std::size_t);
void cpu_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
// Keep this Implementation class local; other backends use the same class name.
namespace {
class Implementation final : public BackendInterface {
public:
    // CPU execution needs no optional device or GPU runtime.
    bool available() const override { return true; }
    // override checks that these methods match the shared interface exactly.
    void axpy(float alpha, const float* x, float* y, std::size_t count) const override {
        cpu_axpy(alpha, x, y, count);
    }
    void matmul(const float* a, const float* b, float* c,
                std::size_t m, std::size_t n, std::size_t k) const override {
        cpu_matmul(a, b, c, m, n, k);
    }
};
}
// Reuse one adapter; returning a base reference avoids copying or exposing its type.
const BackendInterface& cpu_backend() {
    // Static lifetime keeps the returned reference valid after this call ends.
    static Implementation instance;
    return instance;
}
} // namespace prs::detail
