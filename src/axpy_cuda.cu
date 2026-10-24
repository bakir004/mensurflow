#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace prs {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

// Own device memory so errors do not leak allocations.
struct DeviceBuffer {
    float* data = nullptr;
    explicit DeviceBuffer(std::size_t bytes) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void axpy_kernel(float alpha, const float* x, float* y,
                            std::size_t n) {
    // Grid-stride loop: each thread handles independent vector elements.
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                            + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = start; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

} // namespace

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void axpy_cuda(float alpha, const float* x, float* y, std::size_t n) {
    if (!cuda_device_available()) {
        throw std::runtime_error("AXPY: no accessible CUDA device");
    }
    if (n == 0) return;
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error("AXPY: allocation size overflow");
    }

    const std::size_t bytes = n * sizeof(float);
    DeviceBuffer dx(bytes);
    DeviceBuffer dy(bytes);
    check(cudaMemcpy(dx.data, x, bytes, cudaMemcpyHostToDevice), "copy x to GPU");
    check(cudaMemcpy(dy.data, y, bytes, cudaMemcpyHostToDevice), "copy y to GPU");

    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(
        std::min<std::size_t>(1 + (n - 1) / threads, 65535));
    axpy_kernel<<<blocks, threads>>>(alpha, dx.data, dy.data, n);
    check(cudaGetLastError(), "launch AXPY");
    check(cudaDeviceSynchronize(), "execute AXPY");
    check(cudaMemcpy(y, dy.data, bytes, cudaMemcpyDeviceToHost), "copy y to CPU");
}

} // namespace prs
