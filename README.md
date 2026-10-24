# PRS: CPU/CUDA/Metal AXPY

First library operation: **AXPY**, `y[i] = alpha * x[i] + y[i]`.
Uses float32 vectors, with the same API for CPU, NVIDIA CUDA, and Apple Metal. This is a starting
kernel library, not a graph engine. Autotuning is not implemented yet.

## Build and test

Requires CMake >= 3.18 and a C++17 compiler. For GPU execution, install the NVIDIA
CUDA Toolkit (including `nvcc`) and a compatible NVIDIA driver on your NVIDIA
machine. Modern macOS does not support CUDA; use the Metal backend on your M3
MacBook Pro instead. Metal is enabled by default on macOS and needs Apple's
Command Line Tools (`xcode-select --install`). The Metal shader is compiled at
runtime, so no standalone `metal` compiler or full Xcode installation is needed.

```sh
cd PRS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CMake disables the NVIDIA backend if it cannot find a CUDA compiler; Metal can
still run on macOS. To force CPU-only, add
`-DPRS_ENABLE_CUDA=OFF -DPRS_ENABLE_METAL=OFF`. If needed, specify your GPU architecture
with e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` (replace 86 with your GPU's compute
capability without the decimal). OpenMP is optional; without it CPU execution
uses a serial loop. With it, large vectors use multiple CPU threads.

The test prints a small example (`12 24 36`) and checks empty vectors, block
boundaries, larger inputs, aliasing, zero alpha, and size errors. CUDA tests
are explicitly skipped if no device/backend is available; this does not verify
CUDA correctness. Run on an NVIDIA machine and check for `CUDA tests passed`.
On your Mac, the same test suite runs on Metal and prints `Metal tests passed`.

## Use

```cpp
#include "prs/axpy.hpp"

std::vector<float> x{1, 2, 3};
std::vector<float> y{10, 20, 30};
prs::axpy(2.0f, x, y, prs::Backend::CPU); // y = {12, 24, 36}

if (prs::metal_available()) {
    y = {10, 20, 30};
    prs::axpy(2.0f, x, y, prs::Backend::Metal); // Mac GPU
}

if (prs::cuda_available()) {
    y = {10, 20, 30};
    prs::axpy(2.0f, x, y, prs::Backend::CUDA); // same result
}
```

Link the `prs` CMake target from your executable with
`target_link_libraries(your_program PRIVATE prs)`.

## Implementation

- `include/prs/axpy.hpp`: public API and backend selection.
- `src/axpy.cpp`: validation, dispatch, CPU loop (optionally OpenMP).
- `src/axpy_cuda.cu`: device allocation, input copies, CUDA kernel, output copy.
- `src/axpy_metal.mm`: Objective-C++ Metal host code and embedded MSL shader.
  Caches the compiled pipeline, uses shared buffers, and waits before copying
  results back to the host vector. Runs on Apple Silicon GPUs.
- CUDA uses 256 threads/block and a grid-stride loop with bounds checking.
- Calls are synchronous: `y` contains results when the call returns.
- Explicit GPU requests throw on an unavailable backend/device; no silent fallback.

GPU calls currently allocate and copy on every invocation. This makes usage easy
but may be slower than CPU for small inputs. A future performance-oriented API
could retain tensors on the GPU, reuse allocations, and autotune block sizes.

## CPU vs. GPU benchmark

After a Release build, run `./build/axpy_benchmark`. It compares 1,024 and
16,777,216 elements using 11 trials, alternating backend order, with correctness
checks outside timing. Small-vector trials average 100 calls to reduce timer
noise; large-vector trials use one call. The reported time is the median trial
per-call time. Three warmup calls exclude first-use shader compilation.

Example measured on Apple M3 Pro (CPU serial loop, no OpenMP):

| Elements | CPU time | Metal time | Result |
|---|---:|---:|---|
| 1,024 | 0.000058 ms | 0.120322 ms | CPU faster |
| 16,777,216 | 1.803334 ms | 17.506208 ms | CPU about 9.7x faster |

These are **end-to-end API timings**, not isolated GPU kernel timings. The GPU
path allocates buffers, copies both inputs, submits work, waits, and copies the
output on every call. For this low-arithmetic-intensity operation, that overhead
outweighs GPU compute benefits even at the larger size. Shared Apple Silicon
memory does not eliminate the explicit copies in this implementation. Results
vary with system load; they do not imply that CPUs always outperform GPUs.

## A workload where the GPU wins: matrix multiplication

On macOS, run `./build/matmul_benchmark` after a Release build. This standalone
example computes float32 `C = A * B` with a serial, cache-friendly C++ i-k-j loop
and a custom Metal kernel using 16x16 shared-memory tiles. It is an example,
not yet part of the public `prs` API. No CUDA matmul backend is included.

Measured on Apple M3 Pro (median of five calls after warmup):

| Square matrix size | CPU | Metal GPU | GPU speedup |
|---|---:|---:|---:|
| 64 x 64 | 0.019 ms | 0.535 ms | 0.035x (CPU wins) |
| 1024 x 1024 | 69.862 ms | 6.611 ms | **10.568x** |

GPU timings include buffer allocations, input copies, execution, synchronization,
and output copy; shader compilation is excluded. Results are checked against
CPU output outside timing. Matrix multiplication does O(N^3) arithmetic on
O(N^2) input/output data, giving the GPU much more useful work per byte than AXPY.
This compares our custom implementations, **not** optimized CPU BLAS such as
Apple Accelerate, which may substantially change the comparison.


