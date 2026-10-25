// Internal contract shared by the CPU, CUDA, and Metal backend adapters.
// Public callers use include/prs/ops.hpp; src/ops.cpp uses this interface.
#pragma once
#include <cstddef>

// This named namespace is reopened by backend files to share internal symbols.
namespace prs::detail {
// Pure virtual methods (= 0) require every concrete backend to provide them.
class BackendInterface {
public:
    // Supports correct destruction if a derived object is deleted via the base.
    virtual ~BackendInterface() = default;
    // Runtime device availability; compilation support is checked separately.
    virtual bool available() const = 0;
    // Pointers refer to validated host buffers; backends do not own these arrays.
    // The trailing const applies to the backend object; y remains writable.
    virtual void axpy(float alpha, const float* x, float* y, std::size_t count) const = 0;
    // Dimensions and output capacity are already checked by src/ops.cpp.
    virtual void matmul(const float* a, const float* b, float* c,
                        std::size_t m, std::size_t n, std::size_t k) const = 0;
};
// Declarations only: each backend file defines its accessor and concrete class.
// References expose the common interface without copying the static instances.
// GPU accessors are used only when that backend is compiled into the library.
const BackendInterface& cpu_backend();
const BackendInterface& cuda_backend();
const BackendInterface& metal_backend();
} // namespace prs::detail
