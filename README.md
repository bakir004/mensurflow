# PRS: CPU/CUDA AXPY

First library operation: **AXPY**, `y[i] = alpha * x[i] + y[i]`.
Uses float32 vectors, with the same API for CPU and GPU. This is a starting
kernel library, not a graph engine. Autotuning is not implemented yet.

## Build and test

Requires CMake >= 3.18 and a C++17 compiler. For GPU execution, install the NVIDIA
CUDA Toolkit (including `nvcc`) and a compatible NVIDIA driver on your NVIDIA
machine. Modern macOS does not support CUDA; build/run the GPU backend on Linux
or Windows with supported NVIDIA hardware.

```sh
cd PRS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CMake automatically builds CPU-only if it cannot find a CUDA compiler. To force
CPU-only, add `-DPRS_ENABLE_CUDA=OFF`. If needed, specify your GPU architecture
with e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` (replace 86 with your GPU's compute
capability without the decimal). OpenMP is optional; without it CPU execution
uses a serial loop. With it, large vectors use multiple CPU threads.

The test prints a small example (`12 24 36`) and checks empty vectors, block
boundaries, larger inputs, aliasing, zero alpha, and size errors. CUDA tests
are explicitly skipped if no device/backend is available; this does not verify
GPU correctness. Run on an NVIDIA machine and check for `CUDA tests passed`.

## Use

```cpp
#include "prs/axpy.hpp"

std::vector<float> x{1, 2, 3};
std::vector<float> y{10, 20, 30};
prs::axpy(2.0f, x, y, prs::Backend::CPU); // y = {12, 24, 36}

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
- CUDA uses 256 threads/block and a grid-stride loop with bounds checking.
- Calls are synchronous: `y` contains results when the call returns.
- Explicit CUDA requests throw on an unavailable backend/device; no silent fallback.

GPU calls currently allocate and copy on every invocation. This makes usage easy
but may be slower than CPU for small inputs. A future performance-oriented API
could retain tensors on the GPU, reuse allocations, and autotune block sizes.
