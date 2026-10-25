// CPU adapter for BackendInterface. The arithmetic lives in axpy.cpp and
// matmul.cpp; this file exposes it through the shared backend dispatch contract.
#include "../interface.hpp"

namespace prs::detail {
// Definitions live in the neighboring CPU implementation files.
void cpu_axpy(float, const float*, float*, std::size_t);

void cpu_matmul(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);

// Keep this Implementation name local so other backends can use the same name.
namespace {
class Implementation final : public BackendInterface {
  public:
    bool available() const override { return true; }

    void axpy(float alpha, const float* x, float* y, std::size_t count) const override {
        cpu_axpy(alpha, x, y, count);
    }

    void matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                std::size_t k) const override {
        cpu_matmul(a, b, c, m, n, k);
    }
};
} // namespace

const BackendInterface& cpu_backend() {
    // Construct once and keep alive for all references returned by this accessor.
    static Implementation instance;
    return instance;
}
} // namespace prs::detail
