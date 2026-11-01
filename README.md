# mensurflow

A C++17 library for tensors, BLAS and ML ops that is meant to detect the user's hardware
and run on the best backend: cuBLAS on NVIDIA, rocBLAS on AMD, MPS on Apple, our own
kernels on CPU, and **Vulkan compute** as the portable GPU fallback.

> **Status: early.** Only the core types exist so far: `DType`, `Device`, `Shape`/`Strides`
> with broadcasting, and a CPU-only `Tensor` with zero-copy views. There are no ops,
> no dispatcher, and no GPU backends yet. See [Roadmap](#roadmap).

The design lives in [`docs/plan/`](docs/plan/README.md); start with
[ARCHITECTURE.md](docs/plan/ARCHITECTURE.md).

## Build and test

Requires CMake >= 3.20 and a C++17 compiler. Ninja and ccache are used if installed.

```sh
./test.sh              # Linux/macOS: configure Debug build in build/, compile, run all tests
./test.sh -R view      # extra arguments go to ctest (here: only tests matching "view")
```

On Windows, run `test.bat` (same arguments). The scripts also enable the formatting
git hook (see [Formatting](#formatting)).

Manual equivalent:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Link the library from your own CMake project with
`target_link_libraries(your_program PRIVATE mensurflow)`.

## Use

```cpp
#include "mensurflow/tensor.hpp"

using namespace mensurflow;

// Copy host data into a tensor (dtype is taken from the element type: float -> F32).
Tensor m = Tensor::from_vector(std::vector<float>{0, 1, 2, 3, 4, 5}, {2, 3});
// m = [[0, 1, 2],
//      [3, 4, 5]]     shape {2, 3}, strides {3, 1}

// Views share m's memory; only shape/strides/offset change.
Tensor t   = m.transpose(0, 1);          // [[0, 3], [1, 4], [2, 5]]
Tensor col = m.slice(1, 1, 2);           // column 1: [[1], [4]]
Tensor r   = m.view({3, 2});             // [[0, 1], [2, 3], [4, 5]]

Tensor bias = Tensor::from_vector(std::vector<float>{10, 20, 30}, {3});
Tensor b2   = bias.expand({2, 3});       // stride 0 repeats the row, no copy
Tensor c    = bias.unsqueeze(1);         // shape {3, 1}, broadcasts as a column

std::vector<float> out = t.to_vector<float>();  // {0, 3, 1, 4, 2, 5}
```

### Concepts

- **Shape** is the size of each dimension, outermost first. `{}` is a scalar.
- **Strides** are how many *elements* to jump in memory per index step. An element's
  position is `offset + sum(index[i] * stride[i])`.
- **Views** (`transpose`, `slice`, `expand`, `unsqueeze`, `view`) build new
  shape/strides/offset over the same storage, so they never copy data.
- **Broadcasting** follows NumPy: align shapes from the right; each pair of sizes must
  be equal or one of them must be 1. Stretched dimensions get stride 0.
- **Copying a `Tensor` shares memory** (like PyTorch). Memory is freed when the last
  tensor using it goes away.
- **Host** means the CPU and normal RAM. `from_host`/`from_vector` copy into the
  tensor's device memory, and `copy_to_host`/`to_vector` copy back out. Only `cpu:0`
  exists today, so both are `memcpy`.

### Error handling

| Situation | Exception |
|---|---|
| Invalid shape, size mismatch, bad broadcast, wrong dtype in `to_vector`, `view` of a non-contiguous tensor | `std::invalid_argument` |
| Dimension index out of range | `std::out_of_range` |
| Device other than `cpu:0` (no backends yet) | `std::runtime_error` |
| Using a default-constructed (undefined) `Tensor` | `std::logic_error` |

## Source layout

```text
include/mensurflow/        # public API (never includes vendor headers)
  dtype.hpp                # DType, dtype_size, dtype_name, dtype_of<T>
  device.hpp               # DeviceType, Device, parse_device ("cuda:1")
  shape.hpp                # Shape, Strides, numel, contiguity, broadcasting
  tensor.hpp               # Tensor: creation, metadata, views, host copies
src/core/
  dtype.cpp, device.cpp, shape.cpp, tensor.cpp
  storage.hpp/.cpp         # internal: one 64-byte aligned allocation shared by views
tests/
  test_util.hpp            # require / require_throws / run helpers
  core/*_test.cpp          # one executable per test, registered with CTest
docs/
  plan/                    # architecture, kernel roadmap, Vulkan design
  SPECS.md, KERNELS.md, TECHNOLOGY_OVERVIEW.md, FORMATTING.md
test.sh / test.bat         # build + run tests
```

## Roadmap

Following [ARCHITECTURE.md §8](docs/plan/ARCHITECTURE.md):

- [x] Core types: `DType`, `Device`, `Shape`/`Strides`, broadcasting
- [x] CPU `Tensor` with views and strided host copies
- [ ] Dispatcher and backend registry, device detection (`default_device`, `MENSURFLOW_DEVICE`)
- [ ] First CPU ops (elementwise, matmul) through the dispatcher
- [ ] Vulkan backend (RAII core, VMA, shaders compiled to SPIR-V at build time)
- [ ] Milestone 1 kernels on CPU + Vulkan; train an MLP on MNIST
- [ ] Vendor backends (CUDA/cuBLAS, ROCm/rocBLAS, Metal/MPS), autograd, `nn` modules

## Formatting

Code is formatted with `clang-format` (`.clang-format`) by a git pre-commit hook, which
`test.sh`/`test.bat` enable automatically. Setup and manual enabling are described in
[docs/FORMATTING.md](docs/FORMATTING.md).

```sh
git config --local mensurflow.formatOnCommit false   # disable for this clone
git -c mensurflow.formatOnCommit=false commit ...    # skip for one commit
```
