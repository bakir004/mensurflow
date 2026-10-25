// Metal matrix multiplication for row-major A(M x K), B(K x N), and C(M x N).
// utils.hpp supplies the shared device, resource checks, and command completion.
#include "utils.hpp"
#include <cstring>

namespace prs::detail {
// Keep the shader pipeline accessor local to this translation unit.
namespace {
id<MTLComputePipelineState> pipeline() {
    // Compile once and cache this pipeline. The embedded shader assigns a 16 x 16
    // output tile to each threadgroup, with one output element per thread.
    // Threads cooperatively load input tiles, zero-fill edges, and synchronize
    // before reading a tile and before replacing it with the next K slice.
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
}
void metal_matmul(const float* a, const float* b, float* c,
                  std::size_t m, std::size_t n, std::size_t k) {
    // Bound temporary Objective-C object lifetimes to this completed operation.
    @autoreleasepool {
        // Roundup and the shader's base += 16 must also fit uint32.
        constexpr auto limit = std::numeric_limits<std::uint32_t>::max() - 15;
        if (m > limit || n > limit || k > limit) {
            throw std::runtime_error("Matmul: dimensions exceed Metal index limits");
        }
        auto& ctx = metal_context();
        auto state = pipeline();
        // The tiled shader needs 256 threads to cooperate in each threadgroup.
        if (state.maxTotalThreadsPerThreadgroup < 256) {
            throw std::runtime_error("Metal device cannot support 16x16 threadgroups");
        }
        // Copy A and B into checked buffers; C is output-only and needs no initial data.
        id<MTLBuffer> da = metal_buffer(a, m*k), db = metal_buffer(b, k*n);
        id<MTLBuffer> dc = metal_buffer(nullptr, m*n);
        // Record one compute dispatch in a fresh command buffer.
        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        if (!command) throw std::runtime_error("Cannot create Metal command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) throw std::runtime_error("Cannot create Metal encoder");
        // uint3 in MSL has 16-byte size/alignment; supply four host uints.
        const std::uint32_t dims[4] = {static_cast<std::uint32_t>(m),
            static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k), 0};
        // Argument indices correspond to the shader's [[buffer(...)]] declarations.
        [encoder setComputePipelineState:state];
        [encoder setBuffer:da offset:0 atIndex:0];
        [encoder setBuffer:db offset:0 atIndex:1];
        [encoder setBuffer:dc offset:0 atIndex:2];
        [encoder setBytes:dims length:sizeof(dims) atIndex:3];
        // Round up to cover partial tiles; shader bounds checks protect their edges.
        [encoder dispatchThreadgroups:MTLSizeMake((n+15)/16, (m+15)/16, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        // The public API is synchronous: finish GPU work before copying C to the caller.
        metal_wait(command);
        std::memcpy(c, dc.contents, m*n*sizeof(float));
    }
}
} // namespace prs::detail
