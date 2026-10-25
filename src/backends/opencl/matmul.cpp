// Implement synchronous, row-major OpenCL matrix multiplication C = A * B.
// backend.cpp dispatches here after the common API validates input shapes.
#include "utils.hpp"

namespace prs::detail {
namespace {
cl_kernel kernel() {
    // Cache a kernel whose 16x16 groups cooperatively load input tiles into local memory.
    // Barriers protect shared tile reads, and zero-padding handles edge tiles.
    static cl_kernel instance = opencl_kernel(R"opencl(
__kernel void matmul(__global const float* a, __global const float* b, __global float* c,
                     const uint m, const uint n, const uint k) {
    __local float at[16][16];
    __local float bt[16][16];
    uint x = get_local_id(0), y = get_local_id(1);
    uint row = get_group_id(1)*16 + y, col = get_group_id(0)*16 + x;
    float sum = 0;
    for (uint base = 0; base < k; base += 16) {
        at[y][x] = (row < m && base+x < k) ? a[row*k + base+x] : 0;
        bt[y][x] = (base+y < k && col < n) ? b[(base+y)*n + col] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint p = 0; p < 16; ++p) sum += at[y][p] * bt[p][x];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < m && col < n) c[row*n + col] = sum;
}
)opencl",
                                              "matmul");
    return instance;
}
} // namespace

void opencl_matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                   std::size_t k) {
    // Roundup and the kernel's base += 16 must also fit uint32.
    constexpr auto limit = std::numeric_limits<std::uint32_t>::max() - 15;
    if (m > limit || n > limit || k > limit) {
        throw std::runtime_error("Matmul: dimensions exceed OpenCL index limits");
    }
    auto& ctx = opencl_context();
    auto state = kernel();
    if (opencl_group_limit(state) < 256) {
        throw std::runtime_error("OpenCL device cannot support 16x16 work-groups");
    }
    // Copy host inputs into fresh buffers; the kernel overwrites all of C.
    OpenCLBuffer da(CL_MEM_READ_ONLY, m * k, a), db(CL_MEM_READ_ONLY, k * n, b);
    OpenCLBuffer dc(CL_MEM_WRITE_ONLY, m * n);
    opencl_arg(state, 0, da.handle);
    opencl_arg(state, 1, db.handle);
    opencl_arg(state, 2, dc.handle);
    opencl_arg(state, 3, static_cast<cl_uint>(m));
    opencl_arg(state, 4, static_cast<cl_uint>(n));
    opencl_arg(state, 5, static_cast<cl_uint>(k));
    // Dimension 0 walks columns and dimension 1 walks rows. OpenCL 1.2 needs global sizes that
    // are multiples of the group size, so round up; bounds checks skip threads outside C.
    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] = {(n + 15) / 16 * 16, (m + 15) / 16 * 16};
    opencl_check(
        clEnqueueNDRangeKernel(ctx.queue, state, 2, nullptr, global, local, 0, nullptr, nullptr),
        "Launch matmul");
    // Make GPU writes visible before copying the result into host storage.
    opencl_wait();
    opencl_read(dc, c, m * n);
}
} // namespace prs::detail