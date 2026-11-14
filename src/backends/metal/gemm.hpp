// Metal matmul for float32: row-major C[m,n] = A[m,k] * B[k,n] on the GPU, with three kernels to
// choose from and tile sizes that can be changed at run time (the knobs an auto-tuner turns):
//
//   kernel 0  tiled: each threadgroup computes a bm x bn tile of C, staging bk-deep slices of A
//             and B in threadgroup memory; each thread accumulates tm x tn outputs in registers.
//   kernel 1  simdgroup (the default): the same threadgroup tiles, but each of the sg_m x sg_n
//             simdgroups multiplies 8 x 8 blocks with simdgroup_matrix (Apple GPU family 7+; on
//             older GPUs it runs kernel 0 with 64 x 64 x 16 tiles and 4 x 4 per thread). Slices
//             are loaded four floats at a time when the tile lies inside the matrix.
//   kernel 2  MPS: MPSMatrixMultiplication, Apple's own kernel, as the reference to beat. It has no
//             tile sizes of its own.
//
// The defaults (simdgroup, 32 x 64 x 16, 2 x 2 simdgroups) were the fastest of the configurations in
// matmul_tile_benchmark on an M3 Pro, at about 0.8x MPS from 2048^3 up.
//
// Each tile configuration is compiled once, on first use, from the shader source with the tile
// sizes as preprocessor constants, and cached. The tiled kernel adds the products of each C[i,j]
// in increasing p with fused multiply-adds; simdgroup_matrix and MPS do not document their order.
// Results agree with the CPU within float rounding, not bit for bit.
#pragma once

#include <cstdint>

namespace mensurflow::metal {

struct MetalGemmConfig {
    std::int64_t kernel = 1; // 0 tiled, 1 simdgroup, 2 MPS
    std::int64_t bm = 32;    // rows of C per threadgroup
    std::int64_t bn = 64;    // columns of C per threadgroup
    std::int64_t bk = 16;    // depth of the A and B slices staged in threadgroup memory
    std::int64_t tm = 4;     // tiled: rows of C per thread
    std::int64_t tn = 4;     // tiled: columns of C per thread
    std::int64_t sg_m = 2;   // simdgroup: simdgroups along the rows of the tile
    std::int64_t sg_n = 2;   // simdgroup: simdgroups along the columns of the tile
};

// Throws std::invalid_argument naming the rule a configuration breaks: kernel 0-2; positive sizes;
// tiled: bm % tm == bn % tn == 0, (bm / tm) * (bn / tn) threads at most 1024 and tm * tn at most 64;
// simdgroup: bm % (8 sg_m) == bn % (8 sg_n) == bk % 8 == 0 and sg_m * sg_n at most 32; both: the
// staged slices (and the simdgroup edge scratch) at most 32 KB of threadgroup memory.
void validate(const MetalGemmConfig& config);

// The configuration matmul uses on a Metal device. The setter is for tuning and benchmarks: it is not
// thread-safe and must not be called while a matmul is running. Throws like validate().
MetalGemmConfig metal_gemm_config();
void set_metal_gemm_config(const MetalGemmConfig& config);

} // namespace mensurflow::metal

#ifdef __OBJC__
#import <Metal/Metal.h>

namespace mensurflow::metal {

// Encodes and runs C = A * B on `queue` and waits for it. Each matrix is a contiguous row-major
// block starting `*_offset` bytes into its buffer; m, n and k are positive and each matrix has
// fewer than 2^32 elements. Throws std::runtime_error if the shader does not compile, the
// configuration needs more threads than the device allows, or the GPU reports an error.
void gemm_f32(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLBuffer> a,
              std::size_t a_offset, id<MTLBuffer> b, std::size_t b_offset, id<MTLBuffer> c,
              std::size_t c_offset, std::int64_t m, std::int64_t n, std::int64_t k,
              const MetalGemmConfig& config);

} // namespace mensurflow::metal
#endif
