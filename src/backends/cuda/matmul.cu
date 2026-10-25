// CUDA matrix multiplication for row-major A(M x K), B(K x N), and C(M x N).
// Uses 16 x 16 shared-memory tiles; utils.cuh manages device memory and errors.
#include "utils.cuh"
#include <cstddef>

namespace prs::detail {
// Keep the kernel local to this translation unit.
namespace {
__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              std::size_t m, std::size_t n, std::size_t k) {
    // One block computes a 16 x 16 output tile. Each thread accumulates one C
    // element while the block reuses input tiles held in fast shared memory.
    __shared__ float at[16][16];
    __shared__ float bt[16][16];
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * 16 + y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * 16 + x;
    float sum = 0;
    for (std::size_t base = 0; base < k; base += 16) {
        // Zero-fill partial edge tiles so arbitrary matrix sizes are supported.
        at[y][x] = (row < m && base+x < k) ? a[row*k + base+x] : 0;
        bt[y][x] = (base+y < k && col < n) ? b[(base+y)*n + col] : 0;
        // All tile entries must be loaded before any thread reads them.
        __syncthreads();
        for (unsigned p = 0; p < 16; ++p) sum += at[y][p] * bt[p][x];
        // All threads must finish reading before the next iteration overwrites tiles.
        __syncthreads();
    }
    if (row < m && col < n) c[row*n + col] = sum;
}
}
void cuda_matmul(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t n, std::size_t k) {
    // Shared validation handles empty dimensions and size overflow before this
    // wrapper is called. Here, check the active device's two-dimensional grid.
    const auto gx = 1 + (n-1)/16, gy = 1 + (m-1)/16;
    int device = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
    if (gx > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        gy > static_cast<std::size_t>(properties.maxGridSize[1])) {
        throw std::runtime_error("Matmul: dimensions exceed CUDA grid limits");
    }
    // Allocate per-call GPU storage and copy only the input matrices to it.
    // The kernel overwrites every valid C element, so C needs no initial copy.
    const auto ab = m*k*sizeof(float), bb = k*n*sizeof(float), cb = m*n*sizeof(float);
    DeviceBuffer da(ab), db(bb), dc(cb);
    cuda_check(cudaMemcpy(da.data, a, ab, cudaMemcpyHostToDevice), "copy A to GPU");
    cuda_check(cudaMemcpy(db.data, b, bb, cudaMemcpyHostToDevice), "copy B to GPU");
    // Launch one 16 x 16 thread block per output tile.
    matmul_kernel<<<dim3(static_cast<unsigned>(gx), static_cast<unsigned>(gy)), dim3(16,16)>>>(
        da.data, db.data, dc.data, m, n, k);
    // Surface launch/execution errors and finish the GPU work before copying C back.
    cuda_check(cudaGetLastError(), "launch matmul");
    cuda_check(cudaDeviceSynchronize(), "execute matmul");
    cuda_check(cudaMemcpy(c, dc.data, cb, cudaMemcpyDeviceToHost), "copy C to CPU");
}
} // namespace prs::detail
