# CPU dot product: sequential, SIMD, and threads

Standalone C++17 example computing:

```text
dot(a, b) = sum(a[i] * b[i])
```

It does not build or require the main project's GPU backends.

## Methods

1. **Sequential scalar:** ordinary loop, one thread. Auto-vectorization is
   explicitly disabled for this source file with GCC, Clang, and MSVC so it is a
   genuinely scalar baseline. Other optimized C++ loops can auto-vectorize!
2. **SIMD only:** `omp simd reduction`, without creating a thread team.
3. **SIMD + threads:** `omp parallel for simd reduction`, with static scheduling.
4. **Unrolled SIMD + threads:** four independent accumulators, plus tail handling.
   This may reduce dependency costs, but can also lose to compiler-generated code.
5. **Optional BLAS:** `cblas_ddot` from an installed compatible BLAS library,
   such as OpenBLAS or Accelerate. Automatically skipped if unavailable.

**There is no universal “superoptimal” implementation.** BLAS is a useful optimized
baseline, not a guarantee of winning. Dot product has little computation per byte,
so memory bandwidth often limits large inputs. Small inputs favor fewer threads.

## Requirements

- CMake 3.18+, a C++17 compiler, and an OpenMP runtime.
- A Release build for meaningful measurements.
- Optional CBLAS library and headers for the vendor comparison.

### Linux (Ubuntu/Debian)

```sh
sudo apt install build-essential cmake
# Optional vendor baseline:
sudo apt install libopenblas-dev
```

GCC provides OpenMP support through `-fopenmp`, configured by CMake. For Clang,
install its OpenMP runtime development package, typically `libomp-dev`.

### macOS

Install Apple's Command Line Tools and the OpenMP runtime:

```sh
xcode-select --install  # only if not already installed
brew install cmake libomp
```

CMake locates libomp in common Homebrew locations. Accelerate supplies the optional
BLAS baseline without another package. An older SDK may emit a deprecation warning
for its legacy CBLAS interface; the example uses it for broad SDK compatibility.

### Windows

Install CMake and Visual Studio Build Tools with **Desktop development with C++**.
Run the commands below from an x64 Developer Command Prompt or Developer PowerShell.
CMake uses `/openmp:experimental` for MSVC because classic `/openmp` does not enable
SIMD directives. Use a recent compiler; GCC/Clang are the recommended toolchains
for comparing OpenMP SIMD support consistently. Windows/MSVC has not been tested
on the development machine.

Alternatively, MSYS2 UCRT64 provides GCC and OpenMP:

```sh
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
cmake -S openmp -B build-openmp -G Ninja -DCMAKE_BUILD_TYPE=Release
```

Use the MSYS2 UCRT64 terminal for that toolchain. OpenBLAS is optional; if installed,
CMake must be able to find both its CBLAS headers and libraries.

## Build and run

From the repository root, on Linux/macOS or a single-configuration generator:

```sh
cmake -S openmp -B build-openmp -DCMAKE_BUILD_TYPE=Release
cmake --build build-openmp --config Release --parallel
ctest --test-dir build-openmp -C Release --output-on-failure
./build-openmp/dot_product
./build-openmp/dot_product 16000000 11 8
```

With Visual Studio's multi-configuration generator on Windows:

```powershell
cmake -S openmp -B build-openmp
cmake --build build-openmp --config Release --parallel
ctest --test-dir build-openmp -C Release --output-on-failure
.\build-openmp\Release\dot_product.exe 16000000 11 8
```

Arguments:

```text
dot_product [elements=8000000] [trials=9] [OpenMP threads=runtime default]
```

The two input arrays use `16 * elements` bytes total. Choose smaller inputs on
memory-constrained machines. Sizes/trials/threads must be positive integers;
element counts are limited to `INT_MAX` for the common CBLAS interface.

Optional configuration:

```sh
# Enable instructions supported by this machine (GCC/Clang only).
# Do not distribute this binary to CPUs with different instruction support.
cmake -S openmp -B build-openmp -DDOT_NATIVE=ON

# Disable the optional BLAS comparison.
cmake -S openmp -B build-openmp -DDOT_ENABLE_BLAS=OFF
```

Use a fresh build directory when changing compiler/toolchain. The project does not
use `-ffast-math`. OpenMP SIMD reductions still allow a different summation order.
SIMD width and actual generated instructions depend on compiler and target flags;
a directive does not guarantee a particular hardware instruction sequence.

## Reading the results

The program prints median, minimum, and maximum wall-clock execution time for each
method, speedup versus scalar, and effective input bandwidth in decimal GB/s.

- Inputs are initialized using a fixed seed outside timing.
- Every method gets two warmup calls.
- Trial order rotates between methods to reduce fixed-order bias.
- Every result is checked outside timing against a long-double reference, with a
  tolerance scaled by input magnitude. Long double has platform-dependent precision.
- Allocation, initialization, reference calculation, and validation are excluded.
- Thread-team startup/dispatch and completion costs within a kernel call are
  included. Warmups exclude process/runtime first-use effects.
- Effective bandwidth counts both input vectors; cached inputs need not come from
  DRAM, so this is not a direct DRAM bandwidth measurement.
- OpenMP requested and observed team sizes are printed. OS scheduling can still
  prevent all workers from executing simultaneously.

The threads argument controls **our OpenMP methods**, not necessarily BLAS. Control
vendor threading separately (e.g. `OPENBLAS_NUM_THREADS` or `MKL_NUM_THREADS`,
depending on the installed library) and record those settings. BLAS may use one or
many threads internally, so it is not automatically an equal-thread-count contest.

For useful comparisons, run small, medium, and large sizes, then sweep thread
counts. Close heavy background workloads and record CPU, compiler, build flags,
BLAS implementation, and thread settings. Very small inputs need more trials or
batched timing for reliable numbers; the included small tests check correctness,
not meaningful speedup. Never treat a speedup from one machine as universal.
