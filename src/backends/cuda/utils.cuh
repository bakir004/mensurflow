// Shared CUDA helpers: turn runtime errors into exceptions and own device
// allocations used by the AXPY and matrix multiplication implementations.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace prs::detail {
// Include the failed operation in errors reported through the public API.
inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// RAII releases device memory when an operation returns or throws.
// Copying is disabled so a device allocation always has one owner.
struct DeviceBuffer {
    float* data = nullptr;

    explicit DeviceBuffer(std::size_t bytes) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(data); }

    DeviceBuffer(const DeviceBuffer&) = delete;

    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
} // namespace prs::detail
