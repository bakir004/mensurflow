#include "utils.cuh"
#include <algorithm>
#include <limits>

namespace prs::detail {
namespace {
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

void cuda_axpy(float alpha, const float* x, float* y, std::size_t n) {
    if (n == 0) return;
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error("AXPY: allocation size overflow");
    }

    const std::size_t bytes = n * sizeof(float);
    DeviceBuffer dx(bytes);
    DeviceBuffer dy(bytes);
    cuda_check(cudaMemcpy(dx.data, x, bytes, cudaMemcpyHostToDevice), "copy x to GPU");
    cuda_check(cudaMemcpy(dy.data, y, bytes, cudaMemcpyHostToDevice), "copy y to GPU");

    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(
        std::min<std::size_t>(1 + (n - 1) / threads, 65535));
    axpy_kernel<<<blocks, threads>>>(alpha, dx.data, dy.data, n);
    cuda_check(cudaGetLastError(), "launch AXPY");
    cuda_check(cudaDeviceSynchronize(), "execute AXPY");
    cuda_check(cudaMemcpy(y, dy.data, bytes, cudaMemcpyDeviceToHost), "copy y to CPU");
}

} // namespace prs::detail
