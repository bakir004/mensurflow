#pragma once

namespace prs {
// Existing explicit backend values are preserved.
enum class Backend { CPU, CUDA, Metal, Auto };

bool cuda_available();
bool metal_available();
bool backend_available(Backend backend);
// Auto prefers CUDA, then Metal, then CPU. Explicit unavailable choices throw.
Backend resolve_backend(Backend backend = Backend::Auto);
const char* backend_name(Backend backend);
} // namespace prs
