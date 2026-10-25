// Implement synchronous OpenCL AXPY: y = alpha * x + y.
// backend.cpp dispatches here; utils.hpp provides the shared OpenCL infrastructure.
#include "utils.hpp"
#include <algorithm>

namespace prs::detail {
// Keep kernel construction private to this operation's translation unit.
namespace {
cl_kernel kernel() {
    // Each work-item updates one element; the bounds check covers padding in the last group.
    // size_t keeps the comparison correct even when the padded global size exceeds 32 bits.
    // Compile the embedded kernel once and reuse it.
    static cl_kernel instance = opencl_kernel(R"opencl(
__kernel void axpy(__global const float* x, __global float* y, const float alpha, const uint n) {
    size_t i = get_global_id(0);
    if (i < n) y[i] = alpha * x[i] + y[i];
}
)opencl",
                                              "axpy");
    return instance;
}
} // namespace

void opencl_axpy(float alpha, const float* x, float* y, std::size_t n) {
    auto& ctx = opencl_context();
    auto state = kernel();
    // Both inputs start in host memory; this API does not retain GPU buffers between calls,
    // so allocation and copies are part of each operation. y is both read and written.
    OpenCLBuffer dx(CL_MEM_READ_ONLY, n, x);
    OpenCLBuffer dy(CL_MEM_READ_WRITE, n, y);
    const cl_uint count = static_cast<cl_uint>(n);
    // Argument indices must match the embedded kernel's parameter order.
    opencl_arg(state, 0, dx.handle);
    opencl_arg(state, 1, dy.handle);
    opencl_arg(state, 2, alpha);
    opencl_arg(state, 3, count);
    const std::size_t group = std::min<std::size_t>(256, opencl_group_limit(state));
    // OpenCL 1.2 needs a global size that is a multiple of the group size; round up and let
    // the kernel's bounds check skip the extra work-items.
    const std::size_t global = (n + group - 1) / group * group;
    opencl_check(
        clEnqueueNDRangeKernel(ctx.queue, state, 1, nullptr, &global, &group, 0, nullptr, nullptr),
        "Launch AXPY");
    // Wait before reading GPU output back into the caller's y storage.
    opencl_wait();
    opencl_read(dy, y, n);
}
} // namespace prs::detail
