// Internal contract shared by the CPU, CUDA, and Metal adapters. Public vector
// operations validate inputs before dispatching through this interface.
#pragma once
#include <cstddef>

// The same named namespace is reopened by each implementation file.
namespace prs::detail {
// Pure virtual (= 0) methods require each concrete backend to provide the same
// operations. const methods keep the adapter unchanged while writing outputs.
class BackendInterface {
  public:
    // Permit safe destruction through a pointer to the common base class.
    virtual ~BackendInterface() = default;

    virtual bool available() const = 0;

    // Pointers refer to host memory; GPU implementations manage their own copies.
    // Sizes have been checked, and matmul output cannot overlap either input.
    virtual void axpy(float alpha, const float* x, float* y, std::size_t count) const = 0;

    virtual void matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                        std::size_t k) const = 0;
};

// Accessors expose each file-local adapter as a base-class reference without
// copying it. Definitions retain a static instance for a stable lifetime.
// GPU accessors are referenced only when their backend was compiled in.
const BackendInterface& cpu_backend();

const BackendInterface& cuda_backend();

const BackendInterface& metal_backend();
} // namespace prs::detail
