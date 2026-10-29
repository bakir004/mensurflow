// Storage owns one memory allocation that one or more Tensors view.
// For now it is always host memory; the backend registry will replace the raw pointer
// with a backend Buffer so the same Storage can hold GPU memory.
#pragma once

#include "mensurflow/device.hpp"

#include <cstddef>

namespace mensurflow::detail {

struct Storage {
    // 64-byte alignment fits a cache line and the widest SIMD loads (AVX-512).
    static constexpr std::size_t alignment = 64;

    Storage(Device device, std::size_t nbytes);
    ~Storage();
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Device device;
    std::size_t nbytes;
    std::byte* data = nullptr; // nullptr when nbytes == 0
};

} // namespace mensurflow::detail
