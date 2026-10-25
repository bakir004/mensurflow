// Public backend selection API: choose CPU, CUDA, Metal, or automatic selection.
// Implemented in src/backend.cpp; individual device checks live in each backend.
#pragma once

namespace prs {
// Existing explicit backend values are preserved.
enum class Backend { CPU, CUDA, Metal, Auto };

// A GPU backend must be compiled into the library and have a usable device.
bool cuda_available();
bool metal_available();
// CPU and Auto are always available; an unknown enum value is rejected.
bool backend_available(Backend backend);
// Auto prefers CUDA, then Metal, then CPU. Explicit unavailable choices throw.
// Selection is based on availability, not input size or measured performance.
Backend resolve_backend(Backend backend = Backend::Auto);
// Human-readable labels used by tests, benchmarks, and error messages.
const char* backend_name(Backend backend);
} // namespace prs
