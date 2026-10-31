// Backend registration and device lookup.
#include "core/registry.hpp"

#include <exception>
#include <stdexcept>

namespace mensurflow::detail {

Registry::Registry() {
    // A backend whose runtime fails to initialize (missing driver, no device) is skipped, not
    // fatal: the library must keep working on CPU on machines without that hardware.
    auto try_register = [this](void (*register_fn)(Registry&)) {
        try {
            register_fn(*this);
        } catch (const std::exception&) {
        }
    };
    register_cpu_backend(*this); // CPU is required; let a failure here propagate.
#ifdef MENSURFLOW_HAS_CUDA
    try_register(register_cuda_backend);
#endif
#ifdef MENSURFLOW_HAS_ROCM
    try_register(register_rocm_backend);
#endif
#ifdef MENSURFLOW_HAS_METAL
    try_register(register_metal_backend);
#endif
    (void)try_register; // Unused when only the CPU backend is built.
}

Registry& Registry::instance() {
    static Registry registry;
    return registry;
}

void Registry::add(std::unique_ptr<Backend> backend) {
    if (backend->device_count() <= 0) {
        return;
    }
    backend->register_kernels(dispatcher_);
    backends_.push_back(std::move(backend));
}

Backend* Registry::find(DeviceType type) const {
    for (const auto& backend : backends_) {
        if (backend->type() == type) {
            return backend.get();
        }
    }
    return nullptr;
}

Backend& Registry::get(Device device) const {
    Backend* backend = find(device.type);
    if (backend == nullptr || device.index < 0 || device.index >= backend->device_count()) {
        throw std::runtime_error("device " + to_string(device) + " is not available");
    }
    return *backend;
}

std::vector<Device> Registry::devices() const {
    std::vector<Device> out;
    for (const auto& backend : backends_) {
        for (int i = 0; i < backend->device_count(); ++i) {
            out.push_back({backend->type(), i});
        }
    }
    return out;
}

} // namespace mensurflow::detail
