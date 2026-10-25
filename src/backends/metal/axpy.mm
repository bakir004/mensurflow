// Metal AXPY: compile the shader, dispatch vector work, and copy y back to the caller.
// Shared device, buffer, pipeline, and completion helpers are in utils.hpp.
#include "utils.hpp"
#include <algorithm>
#include <cstring>

namespace prs::detail {
// Keep the pipeline accessor private to this translation unit.
namespace {
id<MTLComputePipelineState> pipeline() {
    // Embedded Metal source: each GPU thread updates one vector element.
        NSString* source = @R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void axpy(device const float* x [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 constant float& alpha [[buffer(2)]],
                 constant uint& n [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = alpha * x[i] + y[i];
}
)metal";
    // Compile the shader and create its pipeline only on the first call.
    static id<MTLComputePipelineState> instance = metal_pipeline(source, @"axpy");
    return instance;
}
}
void metal_axpy(float alpha, const float* x, float* y, std::size_t n) {
    // Release temporary Objective-C objects after this synchronous operation.
    @autoreleasepool {
        auto& ctx = metal_context();
        auto state = pipeline();
        // These helpers check resource limits and copy the host vectors into buffers.
        id<MTLBuffer> dx = metal_buffer(x, n), dy = metal_buffer(y, n);
        // A command buffer holds GPU work; its encoder records this kernel's settings.
        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        if (!command) throw std::runtime_error("Cannot create Metal command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) throw std::runtime_error("Cannot create Metal encoder");
        const std::uint32_t count = static_cast<std::uint32_t>(n);
        // Argument indices match the shader's [[buffer(...)]] declarations above.
        [encoder setComputePipelineState:state];
        [encoder setBuffer:dx offset:0 atIndex:0];
        [encoder setBuffer:dy offset:0 atIndex:1];
        [encoder setBytes:&alpha length:sizeof(alpha) atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        // Request one GPU thread per element, using a device-supported group size.
        const NSUInteger threads = std::min<NSUInteger>(256, state.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        // Wait before copying: the public API returns with y fully updated.
        metal_wait(command);
        std::memcpy(y, dy.contents, n*sizeof(float));
    }
}
} // namespace prs::detail
