// Registry: the backends built into this binary that found usable hardware, plus the op table
// they filled. Created on first use. Backends are chosen at build time (MENSURFLOW_CUDA etc.);
// at run time a compiled-in backend is dropped if its runtime or device is missing.
#pragma once

#include "backends/backend.hpp"
#include "core/dispatcher.hpp"

#include <memory>
#include <vector>

namespace mensurflow::detail {

class Registry {
  public:
    static Registry& instance();

    // nullptr if no backend of this type is available.
    Backend* find(DeviceType type) const;

    // Throws runtime_error("device cuda:0 is not available") for an unknown type or index.
    Backend& get(Device device) const;

    // Every usable device, CPU first.
    std::vector<Device> devices() const;

    Dispatcher& dispatcher() { return dispatcher_; }

    const Dispatcher& dispatcher() const { return dispatcher_; }

    // Used by the register_*_backend functions. Registers the backend's kernels and keeps it only
    // when it has at least one device.
    void add(std::unique_ptr<Backend> backend);

  private:
    Registry();

    std::vector<std::unique_ptr<Backend>> backends_;
    Dispatcher dispatcher_;
};

// One per backend, defined in that backend's source file. The ones for backends that were not
// compiled in are never referenced.
void register_cpu_backend(Registry& registry);
void register_cuda_backend(Registry& registry);
void register_rocm_backend(Registry& registry);
void register_metal_backend(Registry& registry);

} // namespace mensurflow::detail
