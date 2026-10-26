# Vulkan Compute matrix multiplication

Standalone C++17 example of float32, contiguous row-major matrix multiplication:

```text
A[M,K] × B[K,N] → C[M,N]
```

The shader computes 16×16 output tiles, stages inputs in shared workgroup memory,
and handles rectangular matrices and partial tiles. The C++ host program creates
buffers, descriptors, a compute pipeline, submits work, waits, and checks every
output against a CPU reference. This is a learning example, **not yet a backend
of the main PRS library** and not a replacement for optimized BLAS.

## Can it run on macOS? Does Vulkan translate to Metal?

**Yes, through MoltenVK.** macOS does not provide a native Vulkan driver.

```text
C++ Vulkan calls → Vulkan loader → MoltenVK → Metal → Apple GPU
GLSL compute shader → glslc → SPIR-V → MoltenVK / SPIRV-Cross → Metal Shading Language
```

MoltenVK implements a portable subset of Vulkan, not every possible Vulkan
feature. This example uses basic storage buffers, float32 math, workgroup shared
memory, and barriers supported by the intended portability path. It enables the
portability enumeration flag and advertised portability subset extension needed
to discover/use MoltenVK devices.

This translation does not turn the host source into a standalone Metal program.
The executable still uses Vulkan and needs a working Vulkan loader/driver setup.
The same shader can run through native Vulkan drivers on compatible Windows/Linux
GPUs. Source portability does not guarantee identical performance across GPUs.

## What to install

All platforms need:

1. **C++17 compiler and CMake 3.18+.**
2. **Vulkan development headers and loader library.**
3. **`glslc`**, the shader compiler, to compile GLSL to SPIR-V during the build.
4. **A Vulkan driver/ICD** for the device: vendor/Mesa on Windows/Linux, MoltenVK
   on macOS. A loader alone is not a GPU driver.

The [LunarG Vulkan SDK](https://vulkan.lunarg.com/) is the simplest integrated
source of headers, loader, shader tools, diagnostics, and validation layers.
On macOS it also supplies MoltenVK. On Windows/Linux you still need an appropriate
GPU driver separately. The Vulkan SDK itself does not require a paid license.

### macOS

A Homebrew-only setup is also available and was tested on Apple M3 Pro:

```sh
brew install cmake vulkan-headers vulkan-loader molten-vk shaderc vulkan-tools
vulkaninfo --summary  # should show the Apple GPU and MoltenVK driver
```

With these Homebrew packages, driver discovery worked without extra environment
variables. Alternatively, use the integrated LunarG SDK setup below:

- Install Apple's Command Line Tools: `xcode-select --install` if needed.
- Install CMake, for example `brew install cmake`.
- Download and install the **macOS Vulkan SDK** from LunarG, including MoltenVK
  and shader tools.
- In the terminal where you will build/run, source the SDK environment script:

```sh
# Replace <version> and location with the actual SDK installation.
source "$HOME/VulkanSDK/<version>/setup-env.sh"
vulkaninfo --summary
glslc --version
```

Keep that environment active when running the program; it supplies SDK/tool paths
and driver discovery settings. If the installation layout differs, follow that
SDK version's getting-started instructions. No window or swapchain is used by
this example.

### Linux (Ubuntu/Debian)

```sh
sudo apt install build-essential cmake libvulkan-dev glslc vulkan-tools
```

Install a driver appropriate for the hardware. For many AMD/Intel systems:

```sh
sudo apt install mesa-vulkan-drivers
```

NVIDIA systems generally use the distribution's supported NVIDIA driver package.
You can alternatively use the Linux LunarG SDK. Confirm device detection:

```sh
vulkaninfo --summary
glslc --version
```

Software drivers may also appear. The example prefers a discrete/integrated GPU
and prints a warning if it selects a CPU/software Vulkan device.

### Windows

- Install Visual Studio Build Tools with **Desktop development with C++**.
- Install CMake and the **Windows Vulkan SDK** from LunarG.
- Install/update the NVIDIA, AMD, or Intel graphics driver for your GPU.
- Open a new x64 Developer Command Prompt/PowerShell after installation.
- Confirm `vulkaninfo --summary` and `glslc --version` work. Add the SDK `Bin`
  directory to PATH if the installer did not do so.

`VULKAN_SDK` should point to the installed SDK so CMake can find its libraries and
tools. The SDK includes developer tooling; normal execution still requires the
Vulkan loader and a compatible driver.

## Build and run

From the repository root:

```sh
cmake -S vulkan -B build-vulkan -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release --parallel
./build-vulkan/vulkan_matmul
./build-vulkan/vulkan_matmul 127 93 65
ctest --test-dir build-vulkan -C Release --output-on-failure
```

Visual Studio's multi-configuration generator places the executable under Release:

```powershell
cmake -S vulkan -B build-vulkan
cmake --build build-vulkan --config Release --parallel
.\build-vulkan\Release\vulkan_matmul.exe 127 93 65
ctest --test-dir build-vulkan -C Release --output-on-failure
```

Arguments are **`M K N`**, all positive; the default is `128 128 128`. Zero-sized
matrices are deliberately excluded from this example. Very large matrices may
exceed device memory/storage-buffer/dispatch limits or make CPU validation slow.

CMake runs `glslc --target-env=vulkan1.0 -O` automatically. If needed, specify
`-DGLSLC_EXECUTABLE=/absolute/path/to/glslc`. The executable embeds the build-time
shader path: keep `matmul.comp.spv` in that build directory. Packaging a relocatable
application and its dependencies is separate work.

## Implementation and limitations

- `matmul.comp`: tiled GPU kernel, 256 invocations per workgroup.
- `main.cpp`: headless Vulkan setup, dispatch, synchronization, and validation.
- `CMakeLists.txt`: independent executable, shader compilation, correctness tests.
- The example uses **host-visible, host-coherent storage-buffer memory** to keep
  upload/readback logic small. If a selected device cannot provide it, execution
  fails clearly. A production backend should also support staging buffers and
  device-local allocations, especially on discrete GPUs.
- Host coherence avoids explicit flush/invalidate operations; it does **not**
  remove the need for a compute-to-host memory barrier and completion wait.
- Matrices must fit 32-bit shader indexing and the device storage-buffer limits.
- Float32 GPU results are checked against double-precision CPU accumulation with
  absolute + relative tolerance. Floating-point reduction/FMA behavior can differ.
- All Vulkan resources are cleaned up, including on errors.

## Timing and performance

The reported time measures **one queue submission through fence completion**.
It is not an isolated shader timestamp and not full end-to-end latency. It excludes
instance/device creation, allocation, shader/pipeline creation, host input/output
copies, and CPU validation. First-dispatch overhead can still be included.

Do not compare that number directly with the main project's end-to-end benchmark
as if the timing scopes were equal. For a serious benchmark, add warmups, repeated
trials, separate device timestamps, persistent allocations, and fair CPU/vendor
baselines. Tile sizes and memory strategies need measurements on each device.

## Troubleshooting

- **CMake cannot find Vulkan / `glslc`:** install the SDK/development packages,
  load the SDK environment, then reconfigure in a fresh build directory.
- **No devices / incompatible driver:** run `vulkaninfo --summary`. On macOS,
  confirm the SDK's MoltenVK driver environment is active. On Linux/Windows,
  verify the graphics driver actually provides Vulkan.
- **CPU device selected:** hardware Vulkan may be unavailable or misconfigured;
  a successful software run is not evidence of GPU execution.
- **Validation:** with the SDK installed, optionally enable its validation layer:
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./build-vulkan/vulkan_matmul`.
  This Unix-style environment syntax differs in PowerShell.

Verified locally on Apple M3 Pro through MoltenVK 1.4.2: both CTest cases,
`127 × 93` times `93 × 65`, and `1024 × 1024` times `1024 × 1024` passed CPU-reference
checks. Other hardware/driver combinations still need testing.

Runtime correctness must be verified on each target GPU. Merely compiling the
host code or shader does not demonstrate driver compatibility or GPU correctness.
