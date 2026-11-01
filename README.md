# Mensurflow

Mensurflow is a C++17 tensor and numerical-kernel library.

It provides a common `Tensor` API, device selection, backend dispatch, CPU
kernels, and optional CUDA, ROCm, and Metal integrations. Vulkan is planned but
is not yet implemented.

## Current status

| Area | Status |
|---|---|
| Tensor API | Implemented |
| CPU backend | Implemented |
| CUDA backend | Optional; partial |
| ROCm backend | Optional; partial and hardware validation pending |
| Metal backend | Optional; partial and hardware validation pending |
| Vulkan backend | Planned |
| Graph runtime | Planned |
| Autograd and training | Planned |
| Mixed precision | Planned |

The current implementation is primarily a float32 tensor library. GPU support
varies by operation and backend.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler
- OpenMP is optional
- CUDA Toolkit is optional
- ROCm and rocBLAS are optional
- Metal and MPSGraph are optional and available only on macOS

## Build and test

### Windows

Use an x64 Visual Studio Developer Command Prompt:

```powershell
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --config Release
ctest --test-dir build-release -C Release --output-on-failure
```

### Linux and macOS

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --config Release
ctest --test-dir build-release --config Release --output-on-failure
```

For a deterministic CPU-only build:

```powershell
cmake -S . -B build-cpu `
  -DCMAKE_BUILD_TYPE=Release `
  -DMENSURFLOW_CUDA=OFF `
  -DMENSURFLOW_ROCM=OFF `
  -DMENSURFLOW_METAL=OFF
```

The available CMake options are:

| Option | Values | Default |
|---|---|---|
| `MENSURFLOW_CUDA` | `AUTO`, `ON`, `OFF` | `AUTO` |
| `MENSURFLOW_ROCM` | `AUTO`, `ON`, `OFF` | `AUTO` |
| `MENSURFLOW_METAL` | `AUTO`, `ON`, `OFF` | `AUTO` |
| `MENSURFLOW_OPENMP` | `ON`, `OFF` | `ON` |
| `MENSURFLOW_NATIVE` | `ON`, `OFF` | `OFF` |

`AUTO` enables a backend when its development SDK is found. `ON` requires the
backend SDK and fails configuration when it is unavailable. `OFF` disables the
backend.

OpenMP parallelizes suitable CPU kernels. `MENSURFLOW_NATIVE=ON` enables
machine-specific compiler optimizations and should not be used for portable
builds.

## Public API

Include the umbrella header:

```cpp
#include "mensurflow/mensurflow.hpp"
```

Example:

```cpp
#include "mensurflow/mensurflow.hpp"

#include <iostream>

int main() {
    using namespace mensurflow;

    const auto x = Tensor::from_vector<float>({1.0f, 2.0f, 3.0f}, {3});
    const auto y = Tensor::from_vector<float>({10.0f, 20.0f, 30.0f}, {3});
    const Tensor result = add(x, y);

    for (float value : result.to_vector<float>()) {
        std::cout << value << ' ';
    }
}
```

Expected output:

```text
11 22 33
```

Operations run on the device where their input tensors reside. Results are
out-of-place and inputs are not modified.

## Supported operations

### BLAS-style operations

- `dot`
- `matvec`
- `matmul`
- `softmax`
- `argmax`
- `axpy`
- `bias_add`

### Reductions

- `sum` — full-tensor sum; an empty tensor returns zero
- `mean` — full-tensor arithmetic mean; empty tensors are invalid
- `max` — full-tensor maximum; empty tensors are invalid

### Elementwise operations

- `add`
- `sub`
- `mul`
- `div`
- `neg`
- `abs`
- `square`
- `relu`
- `sqrt`
- `exp`
- `sigmoid`

`matvec` and `matmul` currently have CPU implementations only. Unsupported backend and
operation combinations throw `NotImplemented`; they do not silently fall back
to another device.

## Tensor operations

The current `Tensor` API includes:

- Construction with `Tensor::empty` and `Tensor::from_host`
- Typed construction with `Tensor::from_vector`
- Device transfer with `Tensor::to`
- Host transfer with `copy_to_host` and `to_vector`
- Scalar reads with `item`
- Views with `transpose`, `slice`, `expand`, `unsqueeze`, and `view`
- Value constructors with `zeros`, `ones`, `full`, `constant`, `eye`, and
  their `*_like` variants
- Packing non-contiguous data with `contiguous`
- Metadata queries for shape, strides, dtype, device, dimensions, and size

Views share storage and do not copy data. Operations that require contiguous
inputs create packed copies internally.

## Devices and backend selection

```cpp
auto selected = default_device();
auto devices = list_devices();

Tensor x = Tensor::empty({1024}, DType::F32, selected);
```

`Device` values use a device type and index, for example `cpu`, `cuda:0`, or
`rocm:1`. `default_device()` uses the first available backend according to the
library's selection policy. CPU is always available.

The `MENSURFLOW_DEVICE` environment variable overrides automatic selection:

```text
MENSURFLOW_DEVICE=cpu
MENSURFLOW_DEVICE=cuda:0
MENSURFLOW_DEVICE=rocm:0
MENSURFLOW_DEVICE=metal:0
```

An explicitly selected unavailable device produces an error. The library does
not silently redirect an operation to another device.

## Backend coverage

| Operation family | CPU | CUDA | ROCm | Metal | Vulkan |
|---|---:|---:|---:|---:|---:|
| Basic elementwise arithmetic | Yes | Partial | Partial | Partial | No |
| Unary elementwise operations | Yes | No | No | No | No |
| `dot` | Yes | Yes | Implemented, hardware validation pending | Implemented, hardware validation pending | No |
| `matvec` | Yes | No | No | No | No |
| `matmul` | Yes | No | No | No | No |
| `softmax` | Yes | No | No | No | No |
| `argmax` | Yes | No | No | No | No |
| Tensor transfers | Yes | Yes | Yes | Yes | No |

GPU-specific capabilities depend on the SDK, driver, and hardware available at
runtime.

## Testing

Run the complete configured test suite with:

```powershell
ctest --test-dir build-release -C Release --output-on-failure
```

CPU tests run on every supported build. GPU-specific tests run only when the
corresponding backend and hardware are available. A skipped GPU test does not
count as GPU correctness evidence.

The GitHub Actions workflow in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs CPU-only Release
configure, build, and CTest checks on Ubuntu and Windows.

## Benchmarks

Benchmarks are built with the project and measure the current tensor and CPU
kernel implementations:

- `elementwise_benchmark`
- `dot_benchmark`
- `matvec_benchmark`
- `matmul_benchmark`
- `softmax_benchmark`
- `argmax_benchmark`
- `cpu_parallel_benchmark`

Build the benchmark targets with:

```powershell
cmake --build build-release --config Release
```

Benchmark results and instructions are documented in
[`docs/benchmarks/README.md`](docs/benchmarks/README.md).

## Formatting

The project uses `.clang-format` for C++, CUDA, and Objective-C++ sources.
Configure and build the formatting target when `clang-format` is available:

```powershell
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target format
```

The tracked pre-commit hook can be enabled with:

```powershell
git config --local core.hooksPath .githooks
```

The hook formats staged source files. Documentation-only changes do not require
formatting.

## Project layout

```text
include/mensurflow/       Public headers
src/core/                 Tensor, device, shape, storage, and dispatcher
src/ops/                  Validation and public operation dispatch
src/backends/cpu/         CPU implementations
src/backends/cuda/        CUDA backend
src/backends/rocm/        ROCm backend
src/backends/metal/       Metal backend
tests/                    Correctness tests
benchmarks/               Performance benchmarks
docs/plan/                Architecture and future roadmap
```

## Roadmap

The detailed plans are:

- [`docs/plan/ARCHITECTURE.md`](docs/plan/ARCHITECTURE.md)
- [`docs/plan/KERNELS_ROADMAP.md`](docs/plan/KERNELS_ROADMAP.md)
- [`docs/plan/VULKAN_DESIGN.md`](docs/plan/VULKAN_DESIGN.md)

The next implementation milestones are:

1. Add an end-to-end dense-inference example and coverage tests.
2. Decide whether optional GEMM parameters (`alpha`, `beta`, and transpose
   flags) are needed before backend work.
3. Improve CPU GEMM and reduction performance.
4. Integrate the Vulkan backend.
5. Add graph execution, neural-network layers, and autograd.
