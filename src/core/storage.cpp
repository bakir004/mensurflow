// Storage forwards allocation and host transfers to the device's backend.
#include "core/storage.hpp"

#include "core/registry.hpp"

#include <cstring>
#include <stdexcept>

namespace mensurflow::detail {

Storage::Storage(Device device, std::size_t nbytes)
    : device(device), nbytes(nbytes), backend_(&Registry::instance().get(device)) {
    buffer = backend_->allocate(device.index, nbytes);
}

std::byte* Storage::host_data() const { return static_cast<std::byte*>(buffer->host_ptr()); }

void Storage::copy_from_host(std::size_t offset, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (offset > nbytes || bytes > nbytes - offset) {
        throw std::out_of_range("Storage::copy_from_host: range exceeds the allocation");
    }
    backend_->stream(device.index).copy_h2d(*buffer, offset, src, bytes);
}

void Storage::copy_to_host(void* dst, std::size_t offset, std::size_t bytes) const {
    if (bytes == 0) {
        return;
    }
    if (offset > nbytes || bytes > nbytes - offset) {
        throw std::out_of_range("Storage::copy_to_host: range exceeds the allocation");
    }
    backend_->stream(device.index).copy_d2h(dst, *buffer, offset, bytes);
}

} // namespace mensurflow::detail
