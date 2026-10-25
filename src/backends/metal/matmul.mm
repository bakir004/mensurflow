// Implement synchronous, row-major Metal matrix multiplication C = A * B.
// backend.mm dispatches here after the common API validates input shapes.
#include "utils.hpp"
#include <cstring>

namespace prs::detail {
namespace {
id<MTLComputePipelineState> pipeline() {
    // Cache a shader whose 16x16 groups cooperatively load input tiles.
    // Barriers protect shared tile reads, and zero-padding handles edge tiles.
    static id<MTLComputePipelineState> instance = metal_pipeline(@R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint3& dims [[buffer(3)]],
                   uint2 group [[threadgroup_position_in_grid]],
                   uint2 t [[thread_position_in_threadgroup]]) {
    threadgroup float at[16][16];
    threadgroup float bt[16][16];
    uint m = dims.x, n = dims.y, k = dims.z;
    uint row = group.y*16 + t.y, col = group.x*16 + t.x;
    float sum = 0;
    for (uint base = 0; base < k; base += 16) {
        at[t.y][t.x] = (row < m && base+t.x < k) ? a[row*k + base+t.x] : 0;
        bt[t.y][t.x] = (base+t.y < k && col < n) ? b[(base+t.y)*n + col] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < 16; ++p) sum += at[t.y][p] * bt[p][t.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < n) c[row*n + col] = sum;
}
)metal", @"matmul");
    return instance;
}
} // namespace

void metal_matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                  std::size_t k) {
    // Keep temporary Objective-C resources scoped to this completed operation.
    @autoreleasepool {
        // Roundup and the shader's base += 16 must also fit uint32.
        constexpr auto limit = std::numeric_limits<std::uint32_t>::max() - 15;
        if (m > limit || n > limit || k > limit) {
            throw std::runtime_error("Matmul: dimensions exceed Metal index limits");
        }
        auto& ctx = metal_context();
        auto state = pipeline();
        if (state.maxTotalThreadsPerThreadgroup < 256) {
            throw std::runtime_error("Metal device cannot support 16x16 threadgroups");
        }
        // Copy host inputs into fresh buffers; the shader overwrites all of C.
        id<MTLBuffer> da = metal_buffer(a, m * k), db = metal_buffer(b, k * n);
        id<MTLBuffer> dc = metal_buffer(nullptr, m * n);
        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        if (!command) {
            throw std::runtime_error("Cannot create Metal command buffer");
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            throw std::runtime_error("Cannot create Metal encoder");
        }
        // uint3 in MSL has 16-byte size/alignment; supply four host uints.
        const std::uint32_t dims[4] = {static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(n),
                                       static_cast<std::uint32_t>(k), 0};
        [encoder setComputePipelineState:state];
        [encoder setBuffer:da offset:0 atIndex:0];
        [encoder setBuffer:db offset:0 atIndex:1];
        [encoder setBuffer:dc offset:0 atIndex:2];
        [encoder setBytes:dims length:sizeof(dims) atIndex:3];
        // Round up to cover the matrix; bounds checks skip threads outside the output.
        [encoder dispatchThreadgroups:MTLSizeMake((n + 15) / 16, (m + 15) / 16, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        // Make GPU writes visible before copying the result into host storage.
        metal_wait(command);
        std::memcpy(c, dc.contents, m * n * sizeof(float));
    }
}
} // namespace prs::detail
