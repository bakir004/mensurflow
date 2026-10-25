// Implement synchronous, row-major CUDA matrix multiplication C = A * B.
// backend.cu dispatches here after the common API validates input shapes.
#include "utils.cuh"
#include <cstddef>

namespace prs::detail {
namespace {
// One 16x16 block computes an output tile. Shared input tiles let threads
// reuse values instead of repeatedly reading them from device global memory.
__global__ void matmul_kernel(const float* a, const float* b, float* c, std::size_t m,
                              std::size_t n, std::size_t k) {
    __shared__ float at[16][16];
    __shared__ float bt[16][16];
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * 16 + y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * 16 + x;
    float sum = 0;
    for (std::size_t base = 0; base < k; base += 16) {
        // Zero-padding makes partial edge tiles work for arbitrary dimensions.
        at[y][x] = (row < m && base + x < k) ? a[row * k + base + x] : 0;
        bt[y][x] = (base + y < k && col < n) ? b[(base + y) * n + col] : 0;
        // Finish cooperative loading before any thread reads the shared tiles.
        __syncthreads();
        for (unsigned p = 0; p < 16; ++p) {
            sum += at[y][p] * bt[p][x];
        }
        // No thread may overwrite a tile until every thread has used it.
        __syncthreads();
    }
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}
} // namespace

void cuda_matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                 std::size_t k) {
    // The public API handles empty shapes and allocation-size overflow first.
    const auto gx = 1 + (n - 1) / 16, gy = 1 + (m - 1) / 16;
    int device = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
    if (gx > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        gy > static_cast<std::size_t>(properties.maxGridSize[1])) {
        throw std::runtime_error("Matmul: dimensions exceed CUDA grid limits");
    }
    const auto ab = m * k * sizeof(float), bb = k * n * sizeof(float), cb = m * n * sizeof(float);
    // Each call owns fresh device buffers; C is fully overwritten by the kernel.
    DeviceBuffer da(ab), db(bb), dc(cb);
    cuda_check(cudaMemcpy(da.data, a, ab, cudaMemcpyHostToDevice), "copy A to GPU");
    cuda_check(cudaMemcpy(db.data, b, bb, cudaMemcpyHostToDevice), "copy B to GPU");
    matmul_kernel<<<dim3(static_cast<unsigned>(gx), static_cast<unsigned>(gy)), dim3(16, 16)>>>(
        da.data, db.data, dc.data, m, n, k);
    cuda_check(cudaGetLastError(), "launch matmul");
    // This synchronous API returns a completed result in the caller's host C.
    cuda_check(cudaDeviceSynchronize(), "execute matmul");
    cuda_check(cudaMemcpy(c, dc.data, cb, cudaMemcpyDeviceToHost), "copy C to CPU");
}
} // namespace prs::detail
