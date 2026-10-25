// Public backend choices and capability queries. src/backend.cpp resolves Auto
// and combines build-time backend support with runtime device availability.
#pragma once

namespace prs {
// Existing explicit backend values are preserved.
enum class Backend { CPU, CUDA, Metal, Auto, OpenCL };

bool cuda_available();

bool metal_available();

bool opencl_available();

// CPU and Auto are always available; GPU choices also require a usable device.
bool backend_available(Backend backend);

// Auto prefers CUDA, then Metal, then OpenCL, then CPU. Explicit unavailable choices throw.
// This policy checks availability rather than estimating operation performance.
Backend resolve_backend(Backend backend = Backend::Auto);

// Return a stable display name; an invalid enum value throws invalid_argument.
const char* backend_name(Backend backend);
} // namespace prs
