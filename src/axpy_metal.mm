#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace prs {
namespace {

std::runtime_error metal_error(const char* operation, NSError* error) {
    const char* message = error ? [[error localizedDescription] UTF8String] : "unknown error";
    return std::runtime_error(std::string(operation) + ": " + message);
}

// Compiled once on first use; ARC manages the Objective-C objects.
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;

    MetalContext() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("AXPY: no Metal device available");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("AXPY: cannot create Metal command queue");

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
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw metal_error("Compile Metal AXPY", error);
        id<MTLFunction> function = [library newFunctionWithName:@"axpy"];
        if (!function) throw std::runtime_error("AXPY: Metal function not found");
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) throw metal_error("Create Metal pipeline", error);
    }
};

MetalContext& context() {
    static MetalContext instance;
    return instance;
}

} // namespace

bool metal_device_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

void axpy_metal(float alpha, const float* x, float* y, std::size_t n) {
    @autoreleasepool {
        auto& ctx = context();
        if (n == 0) return;
        if (n > std::numeric_limits<std::uint32_t>::max() ||
            n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error("AXPY: vector too large for Metal backend");
        }
        const std::size_t bytes = n * sizeof(float);
        if (bytes > ctx.device.maxBufferLength) {
            throw std::runtime_error("AXPY: vector exceeds Metal buffer limit");
        }
        // Shared storage is visible to both CPU and GPU on Apple Silicon.
        // Copies preserve the API's independent host-vector ownership.
        id<MTLBuffer> dx = [ctx.device newBufferWithBytes:x length:bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> dy = [ctx.device newBufferWithBytes:y length:bytes
                                               options:MTLResourceStorageModeShared];
        if (!dx || !dy) throw std::runtime_error("AXPY: Metal buffer allocation failed");

        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        if (!command) throw std::runtime_error("AXPY: cannot create Metal command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) throw std::runtime_error("AXPY: cannot create Metal encoder");
        const std::uint32_t count = static_cast<std::uint32_t>(n);
        [encoder setComputePipelineState:ctx.pipeline];
        [encoder setBuffer:dx offset:0 atIndex:0];
        [encoder setBuffer:dy offset:0 atIndex:1];
        [encoder setBytes:&alpha length:sizeof(alpha) atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        const NSUInteger threads = std::min<NSUInteger>(256, ctx.pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw metal_error("Execute Metal AXPY", command.error);
        }
        std::memcpy(y, dy.contents, bytes);
    }
}

} // namespace prs
