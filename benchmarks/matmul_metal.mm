#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

// Serial, cache-friendly loop order. Release builds allow compiler vectorization.
void cpu_matmul(const std::vector<float>& a, const std::vector<float>& b,
                std::vector<float>& c, unsigned n) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned k = 0; k < n; ++k) {
            const float value = a[i * n + k];
            for (unsigned j = 0; j < n; ++j) {
                c[i * n + j] += value * b[k * n + j];
            }
        }
    }
}

struct MetalMatmul {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;

    MetalMatmul() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal GPU available");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Cannot create command queue");
        NSString* source = @R"metal(
#include <metal_stdlib>
using namespace metal;

// A workgroup computes a 16x16 output tile, reusing input tiles in shared memory.
kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant uint& n [[buffer(3)]],
                   uint2 group [[threadgroup_position_in_grid]],
                   uint2 t [[thread_position_in_threadgroup]]) {
    threadgroup float at[16][16];
    threadgroup float bt[16][16];
    uint row = group.y * 16 + t.y;
    uint col = group.x * 16 + t.x;
    float sum = 0;
    for (uint base = 0; base < n; base += 16) {
        at[t.y][t.x] = (row < n && base + t.x < n) ? a[row*n + base+t.x] : 0;
        bt[t.y][t.x] = (base + t.y < n && col < n) ? b[(base+t.y)*n + col] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; ++k) sum += at[t.y][k] * bt[k][t.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < n && col < n) c[row*n + col] = sum;
}
)metal";
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        id<MTLFunction> function = [library newFunctionWithName:@"matmul"];
        if (!function) throw std::runtime_error("Metal matmul function missing");
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
        if (pipeline.maxTotalThreadsPerThreadgroup < 256) {
            throw std::runtime_error("GPU cannot support 16x16 threadgroups");
        }
    }

    void run(const std::vector<float>& a, const std::vector<float>& b,
             std::vector<float>& c, unsigned n) {
        @autoreleasepool {
            const auto bytes = a.size() * sizeof(float);
            id<MTLBuffer> da = [device newBufferWithBytes:a.data() length:bytes
                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> db = [device newBufferWithBytes:b.data() length:bytes
                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> dc = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!da || !db || !dc) throw std::runtime_error("Metal allocation failed");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!command) throw std::runtime_error("Cannot create command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("Cannot create compute encoder");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:da offset:0 atIndex:0];
            [encoder setBuffer:db offset:0 atIndex:1];
            [encoder setBuffer:dc offset:0 atIndex:2];
            [encoder setBytes:&n length:sizeof(n) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((n+15)/16, (n+15)/16, 1)
                    threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error("Metal matmul execution failed");
            }
            std::memcpy(c.data(), dc.contents, bytes);
        }
    }
};

template <typename F> double time_ms(F operation) {
    const auto start = Clock::now();
    operation();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void benchmark(MetalMatmul& gpu, unsigned n) {
    std::vector<float> a(n*n), b(n*n), cpu(n*n), metal(n*n);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 8;
        b[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8;
    }
    cpu_matmul(a, b, cpu, n);
    gpu.run(a, b, metal, n);
    std::vector<double> cpu_times, gpu_times;
    for (int trial = 0; trial < 5; ++trial) {
        auto run_cpu = [&] { cpu_matmul(a, b, cpu, n); };
        auto run_gpu = [&] { gpu.run(a, b, metal, n); };
        if (trial % 2 == 0) {
            cpu_times.push_back(time_ms(run_cpu));
            gpu_times.push_back(time_ms(run_gpu));
        } else {
            gpu_times.push_back(time_ms(run_gpu));
            cpu_times.push_back(time_ms(run_cpu));
        }
    }
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(metal[i]) ||
            std::abs(cpu[i] - metal[i]) > 1e-4f * std::max(1.0f, std::abs(cpu[i]))) {
            throw std::runtime_error("Matrix multiplication correctness check failed");
        }
    }
    std::sort(cpu_times.begin(), cpu_times.end());
    std::sort(gpu_times.begin(), gpu_times.end());
    std::cout << std::setw(10) << n << std::setw(14) << cpu_times[2]
              << std::setw(14) << gpu_times[2] << std::setw(14)
              << cpu_times[2] / gpu_times[2] << '\n';
}
} // namespace

int main() {
    @autoreleasepool {
        try {
            MetalMatmul gpu;
            std::cout << "Custom float32 matrix multiplication, C = A * B\n"
                      << "CPU: serial i-k-j loop; Metal: tiled 16x16 kernel.\n"
                      << "Median of 5 calls after warmup. GPU includes allocations and copies.\n"
                      << "Shader compilation excluded; not a comparison with optimized CPU BLAS.\n\n"
                      << std::setw(10) << "N x N" << std::setw(14) << "CPU ms"
                      << std::setw(14) << "Metal ms" << std::setw(14) << "GPU speedup"
                      << '\n' << std::fixed << std::setprecision(3);
            benchmark(gpu, 64);
            benchmark(gpu, 1024);
            std::cout << "Correctness checks passed.\n";
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
