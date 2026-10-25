// Public backend discovery, names, and Auto selection. Build definitions guard
// references to optional GPU adapters; their available() methods check devices.
#include "prs/backend.hpp"
#include "backends/interface.hpp"
#include <stdexcept>
#include <string>

namespace prs {
bool cuda_available() {
    // Without the backend's build definition, no CUDA symbols are referenced.
#ifdef PRS_HAS_CUDA
    return detail::cuda_backend().available();
#else
    return false;
#endif
}

bool metal_available() {
#ifdef PRS_HAS_METAL
    return detail::metal_backend().available();
#else
    return false;
#endif
}

bool backend_available(Backend backend) {
    switch (backend) {
    case Backend::Auto:
    case Backend::CPU:
        return true;
    case Backend::CUDA:
        return cuda_available();
    case Backend::Metal:
        return metal_available();
    }
    throw std::invalid_argument("Unknown backend");
}

const char* backend_name(Backend backend) {
    switch (backend) {
    case Backend::CPU:
        return "CPU";
    case Backend::CUDA:
        return "CUDA";
    case Backend::Metal:
        return "Metal";
    case Backend::Auto:
        return "Auto";
    }
    throw std::invalid_argument("Unknown backend");
}

Backend resolve_backend(Backend backend) {
    // Auto uses a fixed priority; it does not benchmark the operation or device.
    if (backend == Backend::Auto) {
        if (cuda_available()) {
            return Backend::CUDA;
        }
        if (metal_available()) {
            return Backend::Metal;
        }
        return Backend::CPU;
    }
    if (!backend_available(backend)) {
        // Explicit selection never silently falls back to a different backend.
        throw std::runtime_error(std::string(backend_name(backend)) +
                                 " backend/device unavailable");
    }
    return backend;
}
} // namespace prs
