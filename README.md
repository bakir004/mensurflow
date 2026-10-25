# PRS: CPU/CUDA/Metal operations

<!-- Project guide: build instructions, public API usage, source layout, and
     performance methodology. For a code walkthrough, start at include/prs/ops.hpp,
     follow src/ops.cpp, then src/backends/interface.hpp and the chosen backend. -->

Float32 **AXPY**, `y[i] = alpha * x[i] + y[i]`, and rectangular **matrix multiplication**,
`C[M,N] = A[M,K] * B[K,N]`, share one API for CPU, NVIDIA CUDA, and Apple Metal.
This is a starting kernel library, not a graph engine. Autotuning is not implemented yet.

## Build and run

Run `./build.sh` on Linux/macOS or `build.bat` on Windows. These scripts configure
`build-auto` in Release mode, build in parallel, and run both benchmarks with
automatic GPU detection. They can be invoked from any working directory and stop
on errors. Windows needs CMake, the CUDA Toolkit, and a supported MSVC C++ toolchain
(e.g. an x64 Visual Studio Developer Command Prompt). Without a GPU backend, the
benchmarks still run using CPU. The scripts build/run benchmarks; run `ctest`
separately for the full test suite.

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
#include "prs/ops.hpp"

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

// Row-major A[2,3] * B[3,2] = C[2,2]. C must be sized before the call.
std::vector<float> a{1, 2, 3, 4, 5, 6};
std::vector<float> b{7, 8, 9, 10, 11, 12};
std::vector<float> c(4);
prs::matmul(a, b, c, 2, 2, 3); // default Auto; c = {58, 64, 139, 154}
```

`Backend::Auto` prefers available CUDA, then Metal, otherwise CPU. Explicit GPU
requests throw if unavailable; there is no silent fallback. `axpy` retains its
CPU default for compatibility; pass `Backend::Auto` for automatic selection.
`resolve_backend()` and `backend_available()` expose selection and availability.
The original `prs/axpy.hpp` remains a compatibility include.

Both operations are synchronous and use host vectors. AXPY allows x/y aliasing;
matmul allows C to alias A/B by computing into a temporary. Matrices are tightly
packed row-major: A has M*K elements, B K*N, C M*N. K=0 fills C with zeros;
M=0 or N=0 requires an empty C. Exact input/output sizes and dimension/byte
overflow are checked before dispatch. Invalid sizes throw `std::invalid_argument`,
overflow throws `std::overflow_error`, and GPU failures throw `std::runtime_error`.
An explicit unavailable backend is rejected even for an empty operation.

Link the `prs` CMake target from your executable with
`target_link_libraries(your_program PRIVATE prs)`.

## Implementation

```text
include/prs/
  backend.hpp                 # Backend enum, availability, Auto selection
  ops.hpp                     # Public AXPY and matmul API
  axpy.hpp                    # Backward-compatible include
src/
  ops.cpp                     # Validation and dispatch
  backend.cpp                 # Availability and backend selection
  backends/
    interface.hpp             # BackendInterface: available(), axpy(), matmul()
    cpu/{backend,axpy,matmul}.cpp
    cuda/{backend,axpy,matmul}.cu
    cuda/utils.cuh            # Shared CUDA errors and allocation ownership
    metal/{backend,axpy,matmul}.mm
    metal/utils.hpp           # Cached device/queue, pipeline and buffer helpers
benchmarks/{axpy,matmul}_benchmark.cpp
tests/{axpy,matmul}_test.cpp
```

Each backend implements the same internal interface. Metal caches device/queue
and operation pipelines, uses shared buffers, and waits before returning host
results. The CPU matmul implementation is a serial cache-friendly i-k-j loop;
CUDA and Metal use rectangular 16x16 shared-memory tiled kernels.
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

Run `./build/matmul_benchmark` after a Release build. This backend-independent
benchmark calls the public library API, using a serial C++ i-k-j CPU baseline
and an automatically selected backend. It checks a 17x23 output with K=19 to
exercise rectangular partial tiles, plus 64x64 and 1024x1024 square cases.
On CPU-only systems it compares two CPU runs. All results are validated outside
timing, after warmup and every timed trial. Tests additionally cover empty/zero-K
matrices, aliases, invalid sizes, overflow, and each available backend.

### NVIDIA RTX 3060 on Arch Linux

The NVIDIA driver alone is not enough to build CUDA code; install the toolkit
and its supported host compiler first (the current Arch package uses GCC 15):

```sh
sudo pacman -S cuda gcc15
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure -V
cmake --build build-cuda --target run_benchmarks
```

CMake finds `nvcc` on PATH or in standard toolkit locations, detects local NVIDIA
compute capabilities using `nvidia-smi`, and selects the versioned host GCC
required by Arch's CUDA package. Explicit compiler/architecture settings override
auto-detection; use `CMAKE_CUDA_ARCHITECTURES` when building for another machine.
On macOS the default backend is Metal. Without a CUDA compiler, CPU targets still
build and both benchmarks fall back to CPU. Detection does not install drivers
or the toolkit. Use a fresh build directory after installing a toolkit if an old
configuration cached CUDA as unavailable.

Verify that the tests print `CUDA tests passed`, not a CUDA skip. Both benchmarks
include allocations, host/device transfers, and synchronization in GPU timings;
AXPY may remain faster on the CPU because it does little computation per byte.
The matrix benchmark uses a custom CUDA kernel, not cuBLAS or tensor cores.

Historical standalone implementation measured on Apple M3 Pro (median of five calls after warmup):

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


