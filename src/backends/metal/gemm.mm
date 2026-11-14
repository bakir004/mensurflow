// Metal matmul kernels (see gemm.hpp). The shader source below is compiled once per tile
// configuration, with the sizes passed as preprocessor constants, because threadgroup arrays and
// the per-thread accumulator arrays need compile-time sizes.
#include "backends/metal/gemm.hpp"

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace mensurflow::metal {

namespace {

// KERNEL 0: tiled, KERNEL 1: simdgroup_matrix. Both stage a BK-deep slice of A (BM x BK) and of B
// (BK x BN) in threadgroup memory per step, zero-padded past the edges of the matrices, so the
// inner loops never check bounds; only the final store does.
const char* const shader_source = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Dims {
    uint m;
    uint n;
    uint k;
};

#if KERNEL == 0

// Thread (tr, tc) owns rows tr + i * RT and columns tc + j * CT of the tile: neighbouring threads
// read neighbouring columns of B from threadgroup memory and write neighbouring columns of C.
kernel void gemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                 device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                 uint2 group [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint RT = BM / TM;
    constexpr uint CT = BN / TN;
    constexpr uint THREADS = RT * CT;
    threadgroup float As[BK * BM]; // transposed: As[p * BM + r]
    threadgroup float Bs[BK * BN]; // Bs[p * BN + c]
    const uint row0 = group.y * BM;
    const uint col0 = group.x * BN;
    const uint tr = tid / CT;
    const uint tc = tid % CT;
    float acc[TM][TN];
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    for (uint k0 = 0; k0 < d.k; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += THREADS) {
            const uint r = e / BK;
            const uint p = e % BK;
            const uint gr = row0 + r;
            const uint gp = k0 + p;
            As[p * BM + r] = gr < d.m && gp < d.k ? A[gr * d.k + gp] : 0.0f;
        }
        for (uint e = tid; e < BK * BN; e += THREADS) {
            const uint p = e / BN;
            const uint c = e % BN;
            const uint gp = k0 + p;
            const uint gc = col0 + c;
            Bs[e] = gp < d.k && gc < d.n ? B[gp * d.n + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < BK; ++p) {
            float a[TM];
            float b[TN];
            for (uint i = 0; i < TM; ++i) {
                a[i] = As[p * BM + tr + i * RT];
            }
            for (uint j = 0; j < TN; ++j) {
                b[j] = Bs[p * BN + tc + j * CT];
            }
            for (uint i = 0; i < TM; ++i) {
                for (uint j = 0; j < TN; ++j) {
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < TM; ++i) {
        const uint r = row0 + tr + i * RT;
        for (uint j = 0; j < TN; ++j) {
            const uint c = col0 + tc + j * CT;
            if (r < d.m && c < d.n) {
                C[r * d.n + c] = acc[i][j];
            }
        }
    }
}

#elif KERNEL == 1

// Simdgroup (sr, sc) owns a WM x WN block of the tile as FM x FN 8x8 accumulators. On an M3 Pro,
// up to 8 accumulators per simdgroup run at full speed and 16 at a third of it (2048^3: 3.0 vs
// 0.9 TFLOP/s); padding the staged rows against bank conflicts and unrolling made no difference.
kernel void gemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                 device float* C [[buffer(2)]], constant Dims& d [[buffer(3)]],
                 uint2 group [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint THREADS = SG_M * SG_N * 32;
    constexpr uint WM = BM / SG_M;
    constexpr uint WN = BN / SG_N;
    constexpr uint FM = WM / 8;
    constexpr uint FN = WN / 8;
    constexpr uint AS = BK; // row stride of As
    constexpr uint BS = BN; // row stride of Bs
    threadgroup float As[BM * AS];
    threadgroup float Bs[BK * BS];
    threadgroup float edge[SG_M * SG_N * 64]; // one 8x8 block per simdgroup, for partial stores
    const uint row0 = group.y * BM;
    const uint col0 = group.x * BN;
    const uint sr = sg / SG_N;
    const uint sc = sg % SG_N;
    simdgroup_float8x8 acc[FM][FN];
    for (uint i = 0; i < FM; ++i) {
        for (uint j = 0; j < FN; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const bool rows_inside = row0 + BM <= d.m;
    const bool cols_inside = col0 + BN <= d.n;
    for (uint k0 = 0; k0 < d.k; k0 += BK) {
        const bool depth_inside = k0 + BK <= d.k;
        if (rows_inside && depth_inside) {
            // Four floats per load; packed_float4 needs only float alignment.
            for (uint e = tid; e < BM * BK / 4; e += THREADS) {
                const uint r = e / (BK / 4);
                const uint p = e % (BK / 4) * 4;
                const float4 v = float4(*reinterpret_cast<device const packed_float4*>(
                    A + (row0 + r) * d.k + k0 + p));
                *reinterpret_cast<threadgroup float4*>(As + r * AS + p) = v;
            }
        } else {
            for (uint e = tid; e < BM * BK; e += THREADS) {
                const uint r = e / BK;
                const uint p = e % BK;
                const uint gr = row0 + r;
                const uint gp = k0 + p;
                As[r * AS + p] = gr < d.m && gp < d.k ? A[gr * d.k + gp] : 0.0f;
            }
        }
        if (cols_inside && depth_inside) {
            for (uint e = tid; e < BK * BN / 4; e += THREADS) {
                const uint p = e / (BN / 4);
                const uint c = e % (BN / 4) * 4;
                const float4 v = float4(*reinterpret_cast<device const packed_float4*>(
                    B + (k0 + p) * d.n + col0 + c));
                *reinterpret_cast<threadgroup float4*>(Bs + p * BS + c) = v;
            }
        } else {
            for (uint e = tid; e < BK * BN; e += THREADS) {
                const uint p = e / BN;
                const uint c = e % BN;
                const uint gp = k0 + p;
                const uint gc = col0 + c;
                Bs[p * BS + c] = gp < d.k && gc < d.n ? B[gp * d.n + gc] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[FM];
            simdgroup_float8x8 b[FN];
            for (uint i = 0; i < FM; ++i) {
                simdgroup_load(a[i], As + (sr * WM + i * 8) * AS + kk, AS);
            }
            for (uint j = 0; j < FN; ++j) {
                simdgroup_load(b[j], Bs + kk * BS + sc * WN + j * 8, BS);
            }
            for (uint i = 0; i < FM; ++i) {
                for (uint j = 0; j < FN; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* scratch = edge + sg * 64;
    for (uint i = 0; i < FM; ++i) {
        const uint r = row0 + sr * WM + i * 8;
        for (uint j = 0; j < FN; ++j) {
            const uint c = col0 + sc * WN + j * 8;
            if (r + 8 <= d.m && c + 8 <= d.n) {
                simdgroup_store(acc[i][j], C + ulong(r) * d.n + c, d.n);
            } else if (r < d.m && c < d.n) {
                // A block that crosses the edge of C: through threadgroup memory, element by element.
                simdgroup_store(acc[i][j], scratch, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    const uint rr = r + e / 8;
                    const uint cc = c + e % 8;
                    if (rr < d.m && cc < d.n) {
                        C[rr * d.n + cc] = scratch[e];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

#endif
)METAL";

constexpr std::int64_t max_threadgroup_bytes = 32 * 1024;

MetalGemmConfig& config_state() {
    static MetalGemmConfig config;
    return config;
}

std::int64_t threads_of(const MetalGemmConfig& c) {
    return c.kernel == 0 ? (c.bm / c.tm) * (c.bn / c.tn) : c.sg_m * c.sg_n * 32;
}

using PipelineKey = std::array<std::int64_t, 8>;

PipelineKey key_of(const MetalGemmConfig& c) {
    if (c.kernel == 0) {
        return {0, c.bm, c.bn, c.bk, c.tm, c.tn, 0, 0};
    }
    return {1, c.bm, c.bn, c.bk, 0, 0, c.sg_m, c.sg_n};
}

std::mutex pipeline_mutex;
std::map<PipelineKey, id<MTLComputePipelineState>> pipelines;

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, const MetalGemmConfig& c) {
    const PipelineKey key = key_of(c);
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    const auto found = pipelines.find(key);
    if (found != pipelines.end()) {
        return found->second;
    }
    @autoreleasepool {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.preprocessorMacros = @{
            @"KERNEL" : @(c.kernel),
            @"BM" : @(c.bm),
            @"BN" : @(c.bn),
            @"BK" : @(c.bk),
            @"TM" : @(c.tm),
            @"TN" : @(c.tn),
            @"SG_M" : @(c.sg_m),
            @"SG_N" : @(c.sg_n),
        };
        // IEEE behaviour for infinities and NaNs, like the CPU kernels.
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:shader_source];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil) {
            throw std::runtime_error(std::string("Metal matmul: the shader did not compile: ") +
                                     [[error localizedDescription] UTF8String]);
        }
        id<MTLFunction> function = [library newFunctionWithName:@"gemm"];
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
        if (state == nil) {
            throw std::runtime_error(std::string("Metal matmul: no pipeline: ") +
                                     [[error localizedDescription] UTF8String]);
        }
        if (static_cast<std::int64_t>([state maxTotalThreadsPerThreadgroup]) < threads_of(c)) {
            throw std::runtime_error(
                "Metal matmul: this tile configuration needs " + std::to_string(threads_of(c)) +
                " threads per threadgroup, the device allows " +
                std::to_string([state maxTotalThreadsPerThreadgroup]) + " for it");
        }
        pipelines.emplace(key, state);
        return state;
    }
}

void run(id<MTLCommandBuffer> command_buffer) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
        throw std::runtime_error(std::string("Metal matmul failed: ") +
                                 [[[command_buffer error] localizedDescription] UTF8String]);
    }
}

void gemm_mps(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLBuffer> a,
              std::size_t a_offset, id<MTLBuffer> b, std::size_t b_offset, id<MTLBuffer> c,
              std::size_t c_offset, std::int64_t m, std::int64_t n, std::int64_t k) {
    @autoreleasepool {
        auto matrix = [](id<MTLBuffer> buffer, std::size_t offset, std::int64_t rows,
                         std::int64_t cols) {
            MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
                matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
                                 columns:static_cast<NSUInteger>(cols)
                                rowBytes:static_cast<NSUInteger>(cols) * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:descriptor];
        };
        MPSMatrixMultiplication* product =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:NO
                                             transposeRight:NO
                                                 resultRows:static_cast<NSUInteger>(m)
                                              resultColumns:static_cast<NSUInteger>(n)
                                            interiorColumns:static_cast<NSUInteger>(k)
                                                      alpha:1.0
                                                       beta:0.0];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        [product encodeToCommandBuffer:command_buffer
                            leftMatrix:matrix(a, a_offset, m, k)
                           rightMatrix:matrix(b, b_offset, k, n)
                          resultMatrix:matrix(c, c_offset, m, n)];
        run(command_buffer);
    }
}

} // namespace

void validate(const MetalGemmConfig& c) {
    auto fail = [](const std::string& rule) {
        throw std::invalid_argument("Metal matmul configuration: " + rule);
    };
    if (c.kernel < 0 || c.kernel > 2) {
        fail("kernel must be 0 (tiled), 1 (simdgroup) or 2 (MPS)");
    }
    if (c.kernel == 2) {
        return; // MPS has no tile sizes of ours
    }
    if (c.bm <= 0 || c.bn <= 0 || c.bk <= 0) {
        fail("bm, bn and bk must be positive");
    }
    std::int64_t floats = (c.bm + c.bn) * c.bk;
    if (c.kernel == 0) {
        if (c.tm <= 0 || c.tn <= 0 || c.bm % c.tm != 0 || c.bn % c.tn != 0) {
            fail("tm and tn must be positive and divide bm and bn");
        }
        if (threads_of(c) > 1024) {
            fail("(bm / tm) * (bn / tn) must be at most 1024 threads");
        }
        if (c.tm * c.tn > 64) {
            fail("tm * tn must be at most 64 accumulators per thread");
        }
    } else {
        if (c.sg_m <= 0 || c.sg_n <= 0 || c.sg_m * c.sg_n > 32) {
            fail("sg_m and sg_n must be positive with sg_m * sg_n at most 32");
        }
        if (c.bm % (8 * c.sg_m) != 0 || c.bn % (8 * c.sg_n) != 0 || c.bk % 8 != 0) {
            fail("bm must be a multiple of 8 * sg_m, bn of 8 * sg_n, and bk of 8");
        }
        floats += c.sg_m * c.sg_n * 64;
    }
    if (floats * static_cast<std::int64_t>(sizeof(float)) > max_threadgroup_bytes) {
        fail("the staged tiles need " + std::to_string(floats * 4) +
             " bytes of threadgroup memory, more than 32 KB");
    }
}

MetalGemmConfig metal_gemm_config() { return config_state(); }

void set_metal_gemm_config(const MetalGemmConfig& config) {
    validate(config);
    config_state() = config;
}

void gemm_f32(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLBuffer> a,
              std::size_t a_offset, id<MTLBuffer> b, std::size_t b_offset, id<MTLBuffer> c,
              std::size_t c_offset, std::int64_t m, std::int64_t n, std::int64_t k,
              const MetalGemmConfig& requested) {
    validate(requested);
    constexpr std::int64_t limit = std::numeric_limits<std::uint32_t>::max();
    if (m * k > limit || k * n > limit || m * n > limit) {
        throw std::runtime_error("Metal matmul: a matrix has 2^32 elements or more");
    }
    if (requested.kernel == 2) {
        gemm_mps(device, queue, a, a_offset, b, b_offset, c, c_offset, m, n, k);
        return;
    }
    MetalGemmConfig config = requested;
    if (config.kernel == 1 && ![device supportsFamily:MTLGPUFamilyApple7]) {
        // No simdgroup_matrix: the tiled kernel with its best sizes on the M3 Pro.
        config = MetalGemmConfig{};
        config.kernel = 0;
        config.bm = 64;
        config.bn = 64;
    }
    id<MTLComputePipelineState> state = pipeline(device, config);
    @autoreleasepool {
        const struct {
            std::uint32_t m, n, k;
        } dims{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(n),
               static_cast<std::uint32_t>(k)};

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:state];
        [encoder setBuffer:a offset:a_offset atIndex:0];
        [encoder setBuffer:b offset:b_offset atIndex:1];
        [encoder setBuffer:c offset:c_offset atIndex:2];
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
        const MTLSize groups =
            MTLSizeMake(static_cast<NSUInteger>((n + config.bn - 1) / config.bn),
                        static_cast<NSUInteger>((m + config.bm - 1) / config.bm), 1);
        [encoder
             dispatchThreadgroups:groups
            threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(threads_of(config)), 1, 1)];
        [encoder endEncoding];
        run(command_buffer);
    }
}

} // namespace mensurflow::metal
