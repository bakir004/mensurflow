// Shared Metal infrastructure for axpy.mm and matmul.mm: runtime compilation,
// a reusable device/queue, checked shared buffers, and synchronous error handling.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace prs::detail {
// Convert an Objective-C error description into the C++ exception used by the API.
inline std::runtime_error metal_error(const char* operation, NSError* error) {
    const char* message = error ? [[error localizedDescription] UTF8String] : "unknown error";
    return std::runtime_error(std::string(operation) + ": " + message);
}
// Operations share a device and command queue rather than creating them per call.
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    MetalContext() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal device available");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Cannot create Metal command queue");
    }
};
inline MetalContext& metal_context() {
    // Construct lazily on first use and keep the context available for later calls.
    static MetalContext context;
    return context;
}
inline id<MTLComputePipelineState> metal_pipeline(NSString* source, NSString* name) {
    // Compile Metal source at runtime, select its named kernel, and prepare it for
    // dispatch. Each operation caches the pipeline returned by this helper.
    auto& ctx = metal_context();
    NSError* error = nil;
    id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:nil error:&error];
    if (!library) throw metal_error("Compile Metal shader", error);
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw std::runtime_error("Metal function not found");
    id<MTLComputePipelineState> pipeline =
        [ctx.device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) throw metal_error("Create Metal pipeline", error);
    return pipeline;
}
inline void metal_check_count(std::size_t count) {
    // Shaders use uint addressing; reject buffers that would wrap an index.
    if (count > std::numeric_limits<std::uint32_t>::max() ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        count*sizeof(float) > metal_context().device.maxBufferLength) {
        throw std::runtime_error("Operation exceeds Metal buffer/index limits");
    }
}
inline id<MTLBuffer> metal_buffer(const float* input, std::size_t count) {
    // Shared storage is CPU-accessible, but newBufferWithBytes still copies input.
    // A null input requests output storage without an initial host-data copy.
    metal_check_count(count);
    auto& ctx = metal_context();
    id<MTLBuffer> buffer = input
        ? [ctx.device newBufferWithBytes:input length:count*sizeof(float)
                                options:MTLResourceStorageModeShared]
        : [ctx.device newBufferWithLength:count*sizeof(float)
                                 options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
    return buffer;
}
inline void metal_wait(id<MTLCommandBuffer> command) {
    // Submit the recorded commands and report GPU failures before returning results.
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw metal_error("Execute Metal operation", command.error);
    }
}
} // namespace prs::detail
