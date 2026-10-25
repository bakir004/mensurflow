// Shared Objective-C++ Metal helpers for device/queue access, shader compilation,
// buffer limits, allocation, and synchronous error reporting in both operations.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace prs::detail {
// Convert Apple's error objects into the exceptions used by the C++ API.
inline std::runtime_error metal_error(const char* operation, NSError* error) {
    const char* message = error ? [[error localizedDescription] UTF8String] : "unknown error";
    return std::runtime_error(std::string(operation) + ": " + message);
}

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    MetalContext() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("No Metal device available");
        }
        queue = [device newCommandQueue];
        if (!queue) {
            throw std::runtime_error("Cannot create Metal command queue");
        }
    }
};

inline MetalContext& metal_context() {
    // Lazily initialize one device and queue shared by AXPY and matmul calls.
    static MetalContext context;
    return context;
}

// Operation-local pipeline() functions cache the compiled result of this helper.
inline id<MTLComputePipelineState> metal_pipeline(NSString* source, NSString* name) {
    auto& ctx = metal_context();
    NSError* error = nil;
    id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        throw metal_error("Compile Metal shader", error);
    }
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        throw std::runtime_error("Metal function not found");
    }
    id<MTLComputePipelineState> pipeline =
        [ctx.device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        throw metal_error("Create Metal pipeline", error);
    }
    return pipeline;
}

inline void metal_check_count(std::size_t count) {
    // Shaders use uint addressing; reject buffers that would wrap an index.
    if (count > std::numeric_limits<std::uint32_t>::max() ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        count * sizeof(float) > metal_context().device.maxBufferLength) {
        throw std::runtime_error("Operation exceeds Metal buffer/index limits");
    }
}

inline id<MTLBuffer> metal_buffer(const float* input, std::size_t count) {
    metal_check_count(count);
    auto& ctx = metal_context();
    // Shared buffers are CPU-accessible, but newBufferWithBytes still copies
    // input data; output-only buffers need no initial host contents.
    id<MTLBuffer> buffer = input ? [ctx.device newBufferWithBytes:input
                                                           length:count * sizeof(float)
                                                          options:MTLResourceStorageModeShared]
                                 : [ctx.device newBufferWithLength:count * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    if (!buffer) {
        throw std::runtime_error("Metal buffer allocation failed");
    }
    return buffer;
}

inline void metal_wait(id<MTLCommandBuffer> command) {
    // Public operations return only after GPU writes are ready for host copying.
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw metal_error("Execute Metal operation", command.error);
    }
}
} // namespace prs::detail
