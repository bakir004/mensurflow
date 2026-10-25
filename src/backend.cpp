// Implements the public availability, naming, and automatic selection helpers.
// Backend adapters perform device checks; this file hides optional build support.
#include "prs/backend.hpp"
#include "backends/interface.hpp"
#include <stdexcept>
#include <string>

namespace prs {
// CMake defines PRS_HAS_CUDA only when CUDA sources are part of the library.
bool cuda_available() {
#ifdef PRS_HAS_CUDA
    return detail::cuda_backend().available();
#else
    return false;
#endif
}
// The compile-time guard keeps CPU-only builds independent of Metal symbols.
bool metal_available() {
#ifdef PRS_HAS_METAL
    return detail::metal_backend().available();
#else
    return false;
#endif
}
// Auto is always usable because it can select the always-available CPU backend.
bool backend_available(Backend backend) {
    switch (backend) {
    case Backend::Auto:
    case Backend::CPU: return true;
    case Backend::CUDA: return cuda_available();
    case Backend::Metal: return metal_available();
    }
    throw std::invalid_argument("Unknown backend");
}
// Keep display names separate from backend selection and implementation details.
const char* backend_name(Backend backend) {
    switch (backend) {
    case Backend::CPU: return "CPU";
    case Backend::CUDA: return "CUDA";
    case Backend::Metal: return "Metal";
    case Backend::Auto: return "Auto";
    }
    throw std::invalid_argument("Unknown backend");
}
// Resolve Auto once for dispatch; explicit unavailable requests must report errors.
Backend resolve_backend(Backend backend) {
    if (backend == Backend::Auto) {
        if (cuda_available()) return Backend::CUDA;
        if (metal_available()) return Backend::Metal;
        return Backend::CPU;
    }
    // Availability does not guarantee that a later GPU allocation will succeed.
    if (!backend_available(backend)) {
        throw std::runtime_error(std::string(backend_name(backend)) + " backend/device unavailable");
    }
    return backend;
}
} // namespace prs
