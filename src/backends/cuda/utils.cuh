// Shared CUDA helpers for axpy.cu and matmul.cu: error reporting and buffer ownership.
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace prs::detail {
// Attach the failed operation's name to CUDA errors and expose them as exceptions.
inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
// Own one GPU allocation. Destruction releases it even when a later step throws.
// Copying is disabled to prevent two objects from freeing the same allocation.
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
