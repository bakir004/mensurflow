// Internal interface every backend (CPU, CUDA, ROCm, Metal, later Vulkan) implements. Nothing above
// the dispatcher includes a vendor header: backends hide their runtime behind these classes.
#pragma once

#include "mensurflow/device.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace mensurflow::detail {

class Dispatcher;

// One device memory allocation. Freed when the last Storage using it goes away.
class Buffer {
  public:
    virtual ~Buffer() = default;

    virtual std::size_t size() const = 0;

    // The backend's own handle: a plain pointer for CPU/CUDA/ROCm, the MTLBuffer for Metal.
    // Null for a zero-byte buffer. Only that backend's kernels interpret it.
    virtual void* native() const = 0;

    // Address the CPU can read and write directly, or nullptr if the memory is device-only.
    // Lets Tensor skip a staging copy on CPU and on unified-memory devices.
    virtual void* host_ptr() const { return nullptr; }
};

// Orders work on one device and moves bytes between host and device.
class Stream {
  public:
    virtual ~Stream() = default;

    // Both copies are complete when they return: src may be reused, dst may be read.
    // d2h also waits for all earlier work on this stream, so reading a result needs no
    // separate synchronize().
    virtual void copy_h2d(Buffer& dst, std::size_t offset, const void* src, std::size_t bytes) = 0;
    virtual void copy_d2h(void* dst, const Buffer& src, std::size_t offset, std::size_t bytes) = 0;

    // Blocks until every operation queued so far has finished.
    virtual void synchronize() = 0;
};

struct DeviceInfo {
    std::string name;
    std::size_t memory_bytes = 0; // 0 when unknown
};

class Backend {
  public:
    virtual ~Backend() = default;

    virtual DeviceType type() const = 0;

    // Usable devices of this type; 0 means the backend is built in but has no hardware.
    virtual int device_count() const = 0;
    virtual DeviceInfo info(int index) const = 0;

    virtual std::shared_ptr<Buffer> allocate(int index, std::size_t bytes) = 0;

    // The stream kernels for this device run on. One per device for now.
    virtual Stream& stream(int index) = 0;

    // Adds this backend's kernels to the op table.
    virtual void register_kernels(Dispatcher& dispatcher) = 0;
};

} // namespace mensurflow::detail
