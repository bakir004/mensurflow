# Architecture & Folder Structure

Goal: one C++ API for tensors, BLAS and ML ops. At runtime the library detects
the hardware and sends each op to the best available implementation:

| Hardware              | Primary backend                          | Fallback chain           |
|-----------------------|------------------------------------------|--------------------------|
| NVIDIA GPU            | CUDA + cuBLAS / cuBLASLt (+ cuDNN, opt.) | Vulkan → CPU             |
| AMD GPU               | HIP + rocBLAS / hipBLASLt (+ MIOpen)     | Vulkan → CPU             |
| Apple Silicon         | Metal + MPS / MPSGraph                   | Vulkan (MoltenVK) → CPU  |
| Intel / ARM / Qualcomm / anything else with a Vulkan driver | **Vulkan** (our own kernels) | CPU |
| No GPU                | CPU (own kernels + optional OpenBLAS / Accelerate) | —              |

Vulkan is the **portable backend**: it is the only GPU backend where *we* write
every kernel, so it must cover every op. Vendor backends cover what their
libraries do well and fall back per op (see "Per-op fallback" below).

---

## 1. Layers

```
 ┌──────────────────────────────────────────────────────────────┐
 │  User code: model, training loop                             │
 ├──────────────────────────────────────────────────────────────┤
 │  prs::nn  (Linear, Conv2d, LayerNorm, Attention, optimizers) │  backend-agnostic
 │  prs::autograd (tape, backward functions)                    │  backend-agnostic
 ├──────────────────────────────────────────────────────────────┤
 │  prs::ops  (matmul, add, softmax, ... ) — validation, shape  │  backend-agnostic
 │            inference, broadcasting, dtype promotion          │
 ├──────────────────────────────────────────────────────────────┤
 │  Dispatcher: (op, backend, dtype) → kernel function pointer  │
 ├──────────────────────────────────────────────────────────────┤
 │  Runtime: Device, Stream, Allocator, Buffer, Event           │  per backend
 ├────────┬────────┬────────┬─────────┬─────────────────────────┤
 │  CPU   │  CUDA  │  ROCm  │  Metal  │  Vulkan                 │
 └────────┴────────┴────────┴─────────┴─────────────────────────┘
```

Rules:

- Everything above the dispatcher **never includes a vendor header**.
  No `<cuda_runtime.h>`, `<vulkan/vulkan.h>`, or `Metal.h` in `include/`.
- Each backend compiles to its own target (static, or a shared plugin — see §4).
- Validation happens once, in `prs::ops`. Kernels may assume valid shapes,
  contiguous-or-described strides, and non-overlapping outputs.

---

## 2. Core public types

The current API takes `std::vector<float>` and copies to the GPU on every call.
That is fine for single kernels but kills ML workloads (a training step does
hundreds of ops). The library must move to **device-resident tensors**:

```cpp
namespace prs {

enum class DeviceType { CPU, CUDA, ROCm, Metal, Vulkan };
enum class DType { F32, F16, BF16, I32, I64, U8, Bool };

struct Device {                     // value type, cheap to copy
    DeviceType type;
    int index = 0;                  // which GPU of this type
};

Device default_device();            // autodetect (policy in §3)
std::vector<Device> list_devices();

class Tensor {                      // shared ownership of storage, like PyTorch
  public:
    static Tensor empty(Shape, DType, Device = default_device());
    static Tensor from_host(const float*, Shape, Device = default_device());

    Tensor to(Device) const;        // copy across devices
    void copy_to_host(float* dst) const;  // synchronizes the stream

    const Shape& shape() const;
    const Strides& strides() const; // views: transpose/slice without copying
    DType dtype() const;
    Device device() const;

  private:
    std::shared_ptr<Storage> storage_;  // backend buffer + allocator handle
    std::size_t offset_;
    Shape shape_;
    Strides strides_;
    DType dtype_;
};

// Ops are free functions; they enqueue on the current stream and return immediately.
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);       // broadcasting
Tensor softmax(const Tensor& x, int dim);
// ...

void synchronize(Device = default_device());

} // namespace prs
```

Keep the existing `prs::axpy(std::vector...)` / `prs::matmul(std::vector...)`
as a thin **convenience layer** on top of `Tensor` (upload → op → download), so
current tests and benchmarks keep working.

### Execution model

- **Eager + asynchronous.** Each op records work into the device's current
  `Stream` (CUDA stream / HIP stream / `MTLCommandBuffer` / Vulkan command buffer)
  and returns. Only `copy_to_host`, `synchronize`, or reading a scalar blocks.
- Later (phase 3+), add an optional **graph mode** (`prs::Graph g; g.capture(...)`)
  that maps to CUDA Graphs / MPSGraph / pre-recorded Vulkan command buffers and
  enables fusion. Do not design for it now, but don't make ops depend on
  immediate execution either.

---

## 3. Device detection & selection

Detection order inside `default_device()`:

1. `PRS_DEVICE` environment variable overrides everything (`cuda:0`, `vulkan:1`, `cpu`).
2. CUDA plugin loaded and `cudaGetDeviceCount() > 0` → CUDA.
3. ROCm plugin loaded and `hipGetDeviceCount() > 0` → ROCm.
4. Apple platform and `MTLCreateSystemDefaultDevice() != nil` → Metal.
5. Vulkan loader present and a physical device with a compute queue exists → Vulkan.
   Prefer `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`, then integrated, never `CPU`
   (llvmpipe/SwiftShader).
6. CPU.

Vulkan also gives us a free hardware inventory: `VkPhysicalDeviceProperties::vendorID`
(`0x10DE` NVIDIA, `0x1002` AMD, `0x8086` Intel, `0x106B` Apple, `0x13B5` ARM,
`0x5143` Qualcomm). Use it for `prs::list_devices()` diagnostics and to print
"NVIDIA GPU found but CUDA runtime missing — using Vulkan" style hints.

Explicit selection (`Device{DeviceType::CUDA}`) never silently falls back —
same policy the current `resolve_backend` already uses.

---

## 4. Shipping: static vs. plugin backends

Problem: a binary linked against `libcudart` won't even start on a machine
without CUDA. Two options:

| Option | How | When |
|---|---|---|
| **A. Compile-time** (current) | `PRS_ENABLE_CUDA` etc., `#ifdef PRS_HAS_CUDA` | Now. Simple, good for the course. |
| **B. Runtime plugins** | Each backend is `libprs_cuda.so` / `.dylib` / `.dll` exporting `prs_backend_register()`; core `dlopen`s whatever is next to it | When one binary must run on many machines. |

Design the backend interface so B is a drop-in change: the core only talks to a
`BackendRegistry` filled via `register_backend(std::unique_ptr<Backend>)`. With
option A, each backend calls it from a static registration function; with B the
plugin calls it from `prs_backend_register()`.

---

## 5. Dispatcher & per-op fallback

Vendor libraries don't cover everything (cuBLAS has no softmax, MPS has no
fused AdamW, ...). So dispatch is **per op, not per backend**:

```cpp
namespace prs::detail {

using KernelFn = void (*)(const OpArgs&, Stream&);

struct KernelKey { OpId op; DeviceType device; DType dtype; };

class Dispatcher {
  public:
    void register_kernel(KernelKey, KernelFn, int priority = 0);
    KernelFn lookup(KernelKey) const;   // throws NotImplemented if missing
};

} // namespace prs::detail
```

Each backend registers what it has. Lookup order for e.g. `softmax` on a
CUDA tensor:

1. A vendor-library kernel (cuDNN softmax), if registered.
2. Our own kernel for that backend (`backends/cuda/kernels/softmax.cu`).
3. **CPU fallback**: copy inputs to host, run the CPU kernel, copy back, and
   log a one-time warning. Off by default, on with `PRS_ENABLE_CPU_FALLBACK=1`
   (same idea as PyTorch's `PYTORCH_ENABLE_MPS_FALLBACK`). Useful while a
   backend is still incomplete; never something to rely on for speed.
4. Otherwise throw `NotImplemented("softmax", CUDA, F16)`.

Ops are **not** silently redirected from CUDA to Vulkan: the memory lives in a
different API, and sharing it needs external-memory interop
(`VK_KHR_external_memory` + `cudaImportExternalMemory`) — possible later, not
worth it early. The Vulkan fallback in the table at the top applies to
*device selection*: if no vendor runtime is present, Auto picks Vulkan.

Within a backend, multiple implementations of the same op are normal
(e.g. Vulkan GEMM: naive / tiled / cooperative-matrix). They register with a
priority and an `is_supported(args, device_caps)` predicate.

A test (`tests/coverage_test.cpp`) prints the op × backend × dtype matrix and
fails CI if the Vulkan or CPU column has holes.

---

## 6. Folder structure

```
PRS/
├── CMakeLists.txt                  # top-level: options, add_subdirectory(...)
├── cmake/
│   ├── FindROCm.cmake
│   ├── CompileShaders.cmake        # glslc/slangc → SPIR-V → embedded C++ arrays
│   └── Sanitizers.cmake
├── include/prs/                    # PUBLIC API — no vendor headers allowed
│   ├── prs.hpp                     # umbrella include
│   ├── device.hpp                  # Device, DeviceType, list_devices, synchronize
│   ├── dtype.hpp
│   ├── tensor.hpp                  # Tensor, Shape, Strides
│   ├── ops/
│   │   ├── blas.hpp                # axpy, dot, gemv, gemm, trsm, ...
│   │   ├── elementwise.hpp         # add, mul, relu, gelu, exp, ...
│   │   ├── reduction.hpp           # sum, mean, max, argmax, ...
│   │   ├── nn.hpp                  # softmax, layer_norm, conv2d, attention, ...
│   │   ├── shape.hpp               # reshape, permute, cat, gather, ...
│   │   └── random.hpp
│   ├── autograd/
│   │   ├── variable.hpp
│   │   └── function.hpp
│   ├── nn/
│   │   ├── module.hpp
│   │   ├── layers.hpp              # Linear, Conv2d, LayerNorm, MultiheadAttention
│   │   ├── loss.hpp
│   │   └── optim.hpp               # SGD, Adam, AdamW
│   └── compat/legacy.hpp           # current std::vector API (axpy, matmul)
│
├── src/
│   ├── core/
│   │   ├── tensor.cpp
│   │   ├── shape.cpp               # broadcasting, stride math
│   │   ├── dispatcher.cpp
│   │   ├── registry.cpp            # backend registration, device detection
│   │   └── device_selection.cpp
│   ├── ops/                        # validation + dispatch for every op
│   │   ├── blas.cpp
│   │   ├── elementwise.cpp
│   │   └── ...
│   ├── autograd/
│   ├── nn/
│   │
│   └── backends/
│       ├── backend.hpp             # internal Backend / Stream / Allocator interfaces
│       │
│       ├── cpu/
│       │   ├── backend.cpp
│       │   ├── blas/               # own kernels; optional OpenBLAS/Accelerate shim
│       │   ├── nn/
│       │   └── simd/               # AVX2/AVX-512/NEON helpers
│       │
│       ├── cuda/
│       │   ├── backend.cu          # device, stream, caching allocator
│       │   ├── cublas.cu           # gemm/gemv/... → cuBLAS / cuBLASLt
│       │   └── kernels/            # .cu for ops cuBLAS lacks (softmax, norms, ...)
│       │
│       ├── rocm/
│       │   ├── backend.hip
│       │   ├── rocblas.hip         # → rocBLAS / hipBLASLt
│       │   └── kernels/            # HIP ports; most CUDA kernels hipify 1:1
│       │
│       ├── metal/
│       │   ├── backend.mm
│       │   ├── mps.mm              # MPSMatrixMultiplication, MPSGraph ops
│       │   └── kernels/            # .metal shaders for missing ops
│       │
│       ├── vulkan/                 # see VULKAN_DESIGN.md for details
│       │   ├── backend.cpp         # registers kernels, implements Backend
│       │   ├── core/               # RAII wrappers & runtime
│       │   │   ├── handle.hpp      # UniqueHandle<T, Deleter>
│       │   │   ├── instance.hpp/.cpp
│       │   │   ├── device.hpp/.cpp      # physical device pick, features, queues
│       │   │   ├── allocator.hpp/.cpp   # VMA + caching buffer pool
│       │   │   ├── buffer.hpp/.cpp
│       │   │   ├── stream.hpp/.cpp      # command buffer batching, timeline semaphore
│       │   │   ├── pipeline.hpp/.cpp    # pipeline cache, specialization constants
│       │   │   ├── descriptors.hpp/.cpp
│       │   │   └── staging.hpp/.cpp     # host↔device ring buffer
│       │   ├── ops/                # C++ side of each op: pick variant, push consts, dispatch
│       │   │   ├── gemm.cpp
│       │   │   ├── elementwise.cpp
│       │   │   └── ...
│       │   └── shaders/            # GLSL compute shaders, compiled at build time
│       │       ├── common/         # shared .glsl includes (indexing, dtype macros, subgroup reduce)
│       │       ├── blas/           # axpy.comp, dot.comp, gemv.comp, gemm_tiled.comp, gemm_coopmat.comp
│       │       ├── elementwise/    # unary.comp, binary_broadcast.comp
│       │       ├── reduction/
│       │       ├── nn/             # softmax.comp, layernorm.comp, conv2d_im2col.comp, flash_attn.comp
│       │       └── random/
│       │
│       └── opencl/                 # existing; frozen. Vulkan replaces its role.
│
├── tests/
│   ├── reference/                  # naive double-precision CPU references
│   ├── ops/                        # one file per op family, parametrized over all devices
│   ├── autograd/                   # numeric gradient checks
│   ├── coverage_test.cpp           # op × device × dtype matrix
│   └── vulkan/                     # Vulkan-specific: RAII leaks, validation-layer clean runs
├── benchmarks/
│   ├── blas/
│   ├── nn/
│   └── models/                     # MLP, small CNN, tiny transformer end-to-end
├── examples/
│   ├── mlp_mnist.cpp
│   └── tiny_gpt.cpp
├── tools/
│   ├── embed_spirv.py              # .spv → .hpp (constexpr uint32_t arrays)
│   └── gen_shader_variants.py      # dtype × tile-size permutations
└── docs/
    └── plan/                       # these documents
```

The current `vulkan/` standalone prototype becomes `src/backends/vulkan/` —
its `matmul.comp` is the starting `shaders/blas/gemm_tiled.comp`, and its
`Context` struct gets split into the RAII classes described in
[VULKAN_DESIGN.md](VULKAN_DESIGN.md).

---

## 7. Internal backend interface

Replaces today's `BackendInterface` (which has one virtual per op and doesn't
scale to 100+ ops):

```cpp
namespace prs::detail {

class Buffer {                       // opaque, owned by an Allocator
  public:
    virtual ~Buffer() = default;
    virtual std::size_t size() const = 0;
};

class Stream {
  public:
    virtual ~Stream() = default;
    virtual void copy_h2d(Buffer& dst, std::size_t off, const void* src, std::size_t n) = 0;
    virtual void copy_d2h(void* dst, const Buffer& src, std::size_t off, std::size_t n) = 0;
    virtual void copy_d2d(Buffer& dst, const Buffer& src, std::size_t n) = 0;
    virtual void synchronize() = 0;
};

class Backend {
  public:
    virtual ~Backend() = default;
    virtual DeviceType type() const = 0;
    virtual int device_count() const = 0;
    virtual DeviceInfo info(int index) const = 0;     // name, vendor, memory, caps
    virtual std::shared_ptr<Buffer> allocate(int index, std::size_t bytes) = 0;
    virtual Stream& current_stream(int index) = 0;
    virtual void register_kernels(Dispatcher&) = 0;   // fills op table
};

} // namespace prs::detail
```

Kernels are plain functions registered into the `Dispatcher`, so adding an op
never touches the `Backend` class.

---

## 8. Phased plan

| Phase | Deliverable |
|---|---|
| 0 | Move Vulkan prototype into `src/backends/vulkan` with RAII core; `Tensor` + `Device` + dispatcher; legacy API on top. |
| 1 | P0 kernels (see KERNELS_ROADMAP.md) on CPU + Vulkan; CUDA/Metal/ROCm through vendor BLAS for GEMM. Train an MLP on MNIST on every backend. |
| 2 | P1 kernels, autograd, `nn` modules, Adam. Small CNN. F16 on Vulkan. |
| 3 | P2: attention (flash), conv fast paths, cooperative-matrix GEMM, graph capture + fusion. Tiny transformer. |
| 4 | P3: LAPACK-style solvers, quantization, multi-GPU. |
