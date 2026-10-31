// Storage owns one memory allocation that one or more Tensors view. The memory is a backend
// Buffer, so the same Storage holds host memory on CPU and device memory on a GPU.
// Backends live for the whole program, so a Storage must not outlive static destruction.
#pragma once

#include "backends/backend.hpp"
#include "mensurflow/device.hpp"

#include <cstddef>
#include <memory>

namespace mensurflow::detail {

struct Storage {
    // Allocates through the device's backend; throws runtime_error if the device is unavailable.
    Storage(Device device, std::size_t nbytes);
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // The bytes if the CPU can address them directly (CPU memory, unified-memory GPUs);
    // nullptr for device-only memory or when nbytes == 0.
    std::byte* host_data() const;

    // Copy `bytes` bytes at byte `offset` from host memory into the buffer, or out of it. Both
    // are complete on return; copy_to_host also waits for queued device work.
    void copy_from_host(std::size_t offset, const void* src, std::size_t bytes);
    void copy_to_host(void* dst, std::size_t offset, std::size_t bytes) const;

    Device device;
    std::size_t nbytes;
    std::shared_ptr<Buffer> buffer;

  private:
    Backend* backend_;
};

} // namespace mensurflow::detail
