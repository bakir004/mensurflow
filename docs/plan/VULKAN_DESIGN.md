# Vulkan Backend Design

How to structure the Vulkan compute backend: RAII ownership, device setup,
memory, command submission, pipelines and descriptors, and shader-writing
practices. It starts from the current `vulkan/main.cpp` prototype and fixes
the things that won't scale.

What the prototype gets right: one owner for all handles, copy disabled,
`vkDeviceWaitIdle` before destruction, barriers not skipped by early returns
in the shader.

What has to change for a library:

| Prototype | Library |
|---|---|
| One `Context` struct owns everything | Long-lived `Device` + per-op cached pipelines + pooled buffers |
| One `VkDeviceMemory` per buffer | VMA sub-allocation + caching pool (drivers limit allocations, often to 4096) |
| SPIR-V read from a file path at runtime | SPIR-V embedded in the binary |
| Record → submit → wait fence per op | Many dispatches per command buffer, timeline semaphore, wait only when the host needs data |
| Descriptor pool + set per run | Push descriptors or buffer device address |
| Fixed 16×16 workgroup in the shader | Specialization constants, tuned per device |

---

## 1. RAII

### 1.1 Generic handle wrapper

Every Vulkan object has exactly one owner, is move-only, and is destroyed in
the owner's destructor. One template covers all handle types:

```cpp
// src/backends/vulkan/core/handle.hpp
namespace prs::vk {

// Owns one Vulkan handle whose destroy function takes (VkDevice, T, allocator).
template <typename T, void (*Destroy)(VkDevice, T, const VkAllocationCallbacks*)>
class DeviceHandle {
  public:
    DeviceHandle() = default;
    DeviceHandle(VkDevice device, T handle) : device_(device), handle_(handle) {}
    ~DeviceHandle() { reset(); }

    DeviceHandle(const DeviceHandle&) = delete;
    DeviceHandle& operator=(const DeviceHandle&) = delete;
    DeviceHandle(DeviceHandle&& other) noexcept
        : device_(other.device_), handle_(std::exchange(other.handle_, VK_NULL_HANDLE)) {}
    DeviceHandle& operator=(DeviceHandle&& other) noexcept {
        if (this != &other) {
            reset();
            device_ = other.device_;
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    void reset() {
        if (handle_ != VK_NULL_HANDLE) {
            Destroy(device_, handle_, nullptr);
            handle_ = VK_NULL_HANDLE;
        }
    }
    T get() const { return handle_; }
    explicit operator bool() const { return handle_ != VK_NULL_HANDLE; }

  private:
    VkDevice device_ = VK_NULL_HANDLE;
    T handle_ = VK_NULL_HANDLE;
};

using ShaderModule        = DeviceHandle<VkShaderModule, vkDestroyShaderModule>;
using Pipeline            = DeviceHandle<VkPipeline, vkDestroyPipeline>;
using PipelineLayout      = DeviceHandle<VkPipelineLayout, vkDestroyPipelineLayout>;
using DescriptorSetLayout = DeviceHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
using DescriptorPool      = DeviceHandle<VkDescriptorPool, vkDestroyDescriptorPool>;
using CommandPool         = DeviceHandle<VkCommandPool, vkDestroyCommandPool>;
using Fence               = DeviceHandle<VkFence, vkDestroyFence>;
using Semaphore           = DeviceHandle<VkSemaphore, vkDestroySemaphore>;
using QueryPool           = DeviceHandle<VkQueryPool, vkDestroyQueryPool>;
using PipelineCache       = DeviceHandle<VkPipelineCache, vkDestroyPipelineCache>;

} // namespace prs::vk
```

`VkInstance`, `VkDevice`, `VkDebugUtilsMessengerEXT` and VMA allocations have
different destroy signatures — give each a small dedicated class.

**Alternative:** `vulkan-hpp`'s `vk::raii::*` does all of this already.
It's a fine choice; the hand-written version above is ~60 lines, has no
template-heavy compile times, and is easier to explain in a report. Pick one
and use it everywhere — don't mix raw handles and wrappers.

### 1.2 Ownership tree and destruction order

C++ destroys members in **reverse declaration order**, so declare them in
creation order and destruction is automatically correct:

```
Instance                      (process-wide, lazily created, one per process)
 └── DebugMessenger           (debug builds only)
 └── Device                   (one per physical GPU used)
      ├── VmaAllocator
      │    └── BufferPool     (caches freed buffers)
      ├── PipelineCache       (saved to disk on destruction)
      ├── PipelineRegistry    (map<key, Pipeline + layout>)
      ├── Stream(s)
      │    ├── CommandPool
      │    ├── command buffers (ring)
      │    └── timeline Semaphore
      └── StagingRing
```

```cpp
class Device {
    // Order matters: later members are destroyed first.
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    UniqueDevice device_;              // vkDestroyDevice — destroyed LAST
    UniqueVmaAllocator allocator_;     // needs device_
    BufferPool buffer_pool_;           // needs allocator_
    PipelineCache pipeline_cache_;
    PipelineRegistry pipelines_;       // needs pipeline_cache_
    std::vector<std::unique_ptr<Stream>> streams_;
    StagingRing staging_;              // destroyed FIRST
  public:
    ~Device();                         // body: vkDeviceWaitIdle(device_.get());
};
```

`~Device()` runs its body **before** members are destroyed, so calling
`vkDeviceWaitIdle` there guarantees the GPU is done with everything.

### 1.3 Lifetime of GPU-referenced resources

The tricky RAII case: a `Tensor` goes out of scope on the host while a
submitted command buffer still reads it. Rules:

1. A `Buffer`'s destructor never calls `vkDestroyBuffer` directly. It returns
   the buffer to the `BufferPool` tagged with the stream's **current timeline
   value** (`last_use`).
2. The pool only hands that buffer out again (or really frees it) once the
   timeline semaphore has reached `last_use`.
3. Same for descriptor pools and command buffers: recycle by timeline value.

This is the same idea as PyTorch's CUDA caching allocator `record_stream`.

### 1.4 Errors

```cpp
inline void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw VulkanError(r, what);  // message includes string_VkResult(r)
}
```

- `VK_ERROR_DEVICE_LOST`: mark the `Device` as lost; every later call throws
  immediately. Don't try to recover.
- `VK_ERROR_OUT_OF_DEVICE_MEMORY` during allocation: empty the buffer pool,
  retry once, then throw `prs::OutOfMemory`.
- Constructors that fail halfway are safe because each already-created member
  is a RAII object that cleans itself up — no `goto cleanup`.

---

## 2. Instance & device setup

### 2.1 Instance

- API version: request **Vulkan 1.2** minimum (timeline semaphores, buffer
  device address, 8/16-bit storage, subgroup properties are core). Use 1.3
  features (synchronization2, maintenance4) when available.
- macOS (MoltenVK): enable `VK_KHR_portability_enumeration` and set
  `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`; enable
  `VK_KHR_portability_subset` on the device if listed.
- Debug builds / `PRS_VULKAN_VALIDATION=1`: enable `VK_LAYER_KHRONOS_validation`
  and `VK_EXT_debug_utils`; route messages to our logger; **treat validation
  errors as test failures** in CI.
- Use **volk** (meta-loader) so the library doesn't hard-link `libvulkan`:
  if `volkInitialize()` fails, the Vulkan backend reports `device_count() == 0`
  instead of the program failing to load.

### 2.2 Physical device selection

Score each device:

1. Must have a queue family with `VK_QUEUE_COMPUTE_BIT`.
2. Prefer discrete > integrated > virtual; reject `VK_PHYSICAL_DEVICE_TYPE_CPU`
   (llvmpipe, SwiftShader) unless `PRS_VULKAN_ALLOW_CPU=1` (useful for CI).
3. Tie-break on device-local heap size.

Prefer a **dedicated compute queue family** (compute without graphics — AMD
and NVIDIA expose one) for work, and a transfer-only family for big uploads
if present.

### 2.3 Capabilities struct

Query once, store in a `DeviceCaps` struct, and let kernels pick variants from it:

```cpp
struct DeviceCaps {
    uint32_t vendor_id;
    uint32_t subgroup_size;              // 32 NVIDIA, 32/64 AMD, 8–32 Intel, 32 Apple
    uint32_t min_subgroup_size, max_subgroup_size;  // VK_EXT_subgroup_size_control
    bool subgroup_arithmetic, subgroup_shuffle;
    uint32_t max_shared_memory;          // maxComputeSharedMemorySize, ≥16 KiB
    uint32_t max_workgroup_invocations;  // ≥128 guaranteed
    uint32_t max_push_constants;         // ≥128 bytes guaranteed
    bool fp16_arith, fp16_storage, int8_storage, bf16;
    bool atomic_float_add, atomic_float16_add;
    bool cooperative_matrix;             // + list of supported (M,N,K,types)
    bool integer_dot_product;
    bool buffer_device_address;
    bool push_descriptor;
    bool timestamp_queries;  float timestamp_period_ns;
    bool unified_memory;                 // integrated GPU / Apple: device-local + host-visible heap
};
```

Enable only features that exist and are needed, by chaining
`VkPhysicalDeviceVulkan11Features`, `...12Features`, `...13Features` and
extension feature structs into `VkDeviceCreateInfo::pNext`.

---

## 3. Memory

### 3.1 Use VMA

Use [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator).
It handles memory type selection, sub-allocation out of large blocks, and
alignment. Writing this yourself is a lot of work with no payoff.

| Buffer kind | VMA usage | Notes |
|---|---|---|
| Tensor storage | `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE` | `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST | SHADER_DEVICE_ADDRESS` |
| Upload staging | `AUTO_PREFER_HOST` + `HOST_ACCESS_SEQUENTIAL_WRITE` | persistently mapped |
| Readback staging | `AUTO_PREFER_HOST` + `HOST_ACCESS_RANDOM` | host-cached memory, much faster to read |
| Unified-memory devices | device-local **and** host-visible | skip staging entirely: map and memcpy |

### 3.2 Caching buffer pool

`vkAllocateMemory` / `vmaCreateBuffer` are slow and ML allocates constantly.
Keep freed buffers in size buckets (round up to powers of two below 1 MiB,
to 2 MiB multiples above) and reuse them, gated by the timeline value (§1.3).
Expose `prs::empty_cache()` and memory statistics for debugging.

### 3.3 Transfers

- **StagingRing**: one large persistently-mapped host buffer used as a ring.
  `upload(dst, data, bytes)` memcpys into the ring and records
  `vkCmdCopyBuffer` on the stream — no allocation per copy.
- Uploads larger than the ring go in chunks.
- Readback: record copy → submit → wait timeline → memcpy out. This is the
  only place the host normally blocks.
- If the memory type is not `HOST_COHERENT`, call `vmaFlushAllocation` /
  `vmaInvalidateAllocation`.

---

## 4. Command submission (Streams)

The biggest performance mistake is **submit + wait per op**. A GEMM on a small
matrix takes microseconds; a submit + fence wait costs 20–100 µs.

```cpp
class Stream {
  public:
    // Ops call these while recording into the current command buffer.
    void dispatch(const PipelineEntry&, std::span<const BufferBinding>,
                  const void* push_constants, uint32_t pc_size, Dim3 groups);
    void copy(const Buffer& src, Buffer& dst, VkDeviceSize bytes);
    void barrier();                 // compute→compute RAW hazard (see below)

    uint64_t flush();               // end + submit current cmd buffer, return timeline value
    void wait(uint64_t value);      // vkWaitSemaphores on the timeline
    void synchronize() { wait(flush()); }

  private:
    CommandPool pool_;
    std::vector<VkCommandBuffer> ring_;  // reused when their timeline value is reached
    Semaphore timeline_;                 // VK_SEMAPHORE_TYPE_TIMELINE
    uint64_t next_value_ = 1;
    uint32_t dispatches_in_cmd_ = 0;     // auto-flush after N (e.g. 64–256) to keep GPU busy
};
```

- One command buffer collects many dispatches. Flush when (a) the host needs
  data, (b) a dispatch-count threshold is hit, or (c) `synchronize()`.
- Use **timeline semaphores** instead of fences: one monotonic counter per
  stream, trivially answers "is buffer X still in use?".
- Allocate command pools with `VK_COMMAND_POOL_CREATE_TRANSIENT_BIT` and reset
  per command buffer (or reset the whole pool per frame of work).
- One stream per thread at most; `VkQueue` submission must be externally
  synchronized — protect `vkQueueSubmit` with a mutex if several streams share a queue.

### 4.1 Barriers

Between two dispatches where the second reads what the first wrote:

```cpp
VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
```

- Start simple: **a global memory barrier after every dispatch**. Correct,
  and cheap compared to the dispatch itself.
- Optimization later: track the last writer of each buffer and only emit a
  barrier when a dispatch reads/writes a buffer written since the last barrier.
  Independent ops can then overlap.
- Transfer → compute (after an upload) and compute → transfer (before a
  readback) need their own stage/access masks.
- Prefer `vkCmdPipelineBarrier2` (synchronization2) when available.

---

## 5. Pipelines, descriptors, push constants

### 5.1 Binding strategy

Two good choices; pick based on device support and fall back:

1. **Buffer device address** (preferred, Vulkan 1.2 core): pass `uint64_t`
   GPU pointers in push constants. No descriptor sets at all; makes
   variable-input kernels (concat, multi-tensor Adam) trivial.
   ```glsl
   #extension GL_EXT_buffer_reference : require
   layout(buffer_reference, std430, buffer_reference_align = 16) buffer F32Buf { float v[]; };
   layout(push_constant) uniform P { F32Buf a; F32Buf b; F32Buf c; uint n; } pc;
   ```
2. **Push descriptors** (`VK_KHR_push_descriptor`): `vkCmdPushDescriptorSetKHR`
   per dispatch, no pools. Supported nearly everywhere including MoltenVK.
3. Last resort: a per-stream `DescriptorPool` reset when its timeline value is
   reached, one set per dispatch.

Don't allocate a descriptor pool per op (as the prototype does).

### 5.2 Push constants

- Pass sizes, strides, scalars (α, β, ε) and buffer addresses via push
  constants. **Max 128 bytes** is the only guaranteed size — design every
  kernel's struct to fit.
- Shapes/strides for rank-8 broadcasting don't fit: put them in a small
  per-dispatch uniform/storage buffer written through the staging ring.
- Keep the C++ struct and the GLSL block in sync with `static_assert(sizeof(...))`
  and identical field order; std430 / scalar layout rules apply.

### 5.3 Specialization constants

Make tile sizes, workgroup size and dtype-dependent constants specialization
constants so one SPIR-V module produces many tuned pipelines:

```glsl
layout(local_size_x_id = 0, local_size_y_id = 1) in;
layout(constant_id = 2) const uint TILE_K = 16;
layout(constant_id = 3) const uint WPT = 4;   // work per thread
```

### 5.4 Pipeline registry & cache

```cpp
struct PipelineKey {
    ShaderId shader;                      // enum generated from the shaders/ tree
    std::array<uint32_t, 8> spec_consts;
    bool operator==(const PipelineKey&) const = default;
};
```

- Create pipelines **lazily** on first use and keep them in an
  `unordered_map<PipelineKey, PipelineEntry>` (entry = pipeline + layout).
- One `VkPipelineCache` per device, loaded from and saved to
  `~/.cache/prs/vulkan/<vendor>-<device>-<driver>.bin`. Validate the header
  (vendorID, deviceID, pipelineCacheUUID) before using it.
- Thread safety: a mutex around the map; pipeline creation itself is thread-safe.

---

## 6. Shaders

### 6.1 Language & build

- **GLSL** compiled with `glslc` (already in use). Fine for everything we need,
  incl. subgroups, buffer references, cooperative matrices
  (`GL_KHR_cooperative_matrix`). Slang is a nice alternative (generics for
  dtypes) but adds a dependency — not needed now.
- Compile at **build time**: CMake → `glslc --target-env=vulkan1.2 -O` →
  `spirv-val` → `tools/embed_spirv.py` generates `shaders_generated.hpp` with
  `constexpr uint32_t gemm_tiled_f32[] = {...}`. No runtime file paths.
- Generate dtype variants with defines, not copy-paste:
  `glslc -DDTYPE=float16_t -DACC=float -DSUFFIX=f16 unary.comp`.
  `tools/gen_shader_variants.py` lists the matrix (shader × dtype × flags).
- Shared code lives in `shaders/common/*.glsl` and is pulled in with
  `#extension GL_GOOGLE_include_directive : require`.

### 6.2 Kernel-writing rules

**Correctness**

- Every global load/store is bounds-checked; dispatch sizes are rounded up.
- Never `return` before a `barrier()` — threads outside the range must still
  hit the barrier (prototype already does this right).
- `barrier()` in GLSL compute syncs execution **and** shared memory; for
  global memory visibility within a workgroup add `memoryBarrierBuffer()`.
- No cross-workgroup synchronization exists inside one dispatch. Multi-pass
  algorithms (big reductions, scans) are multiple dispatches with a barrier
  between, or use atomics + a "last workgroup finishes" counter.
- Integer overflow: index with `uint` and make the C++ side reject tensors
  above 2³² elements, or use 64-bit indexing (`GL_EXT_shader_explicit_arithmetic_types_int64`)
  in a variant.
- Large grids: `maxComputeWorkGroupCount[0]` can be as low as 65535; use a
  grid-stride loop or 2D/3D dispatch for big elementwise ops.

**Performance**

- Workgroup size a multiple of the subgroup size; 64, 128 or 256 threads for
  1D kernels is a safe default on all vendors.
- Coalesced access: consecutive `gl_LocalInvocationID.x` → consecutive addresses.
- Vectorized loads (`vec4` / `f16vec4`) for memory-bound kernels; handle the
  tail separately.
- Reductions: `subgroupAdd` / `subgroupMax` first, then shared memory across
  subgroups, then one atomic or a second pass. Don't hardcode a subgroup size
  of 32 — read `gl_SubgroupSize` or pin it with `VK_EXT_subgroup_size_control`.
- Shared-memory tiles: pad the inner dimension (`[16][17]`) to avoid bank
  conflicts when reading columns.
- GEMM: each thread computes a register micro-tile (e.g. 4×4 or 8×4),
  double-buffer shared tiles, unroll the inner K loop.
- Accumulate F16/BF16 inputs in F32.
- `precise` / no fast-math surprises: SPIR-V has no global fast-math flag, but
  drivers may contract `a*b+c` into FMA — tests need tolerances, not exact equality.

### 6.3 Example: the elementwise template

```glsl
#version 450
#extension GL_GOOGLE_include_directive : require
#include "common/dtype.glsl"     // defines T, ACC, LOAD(), STORE()

layout(local_size_x_id = 0) in;
layout(set = 0, binding = 0, std430) readonly  buffer X { T x[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Y { T y[]; };
layout(push_constant) uniform P { uint n; uint op; float alpha; } pc;

ACC apply(ACC v) {
    switch (pc.op) {                 // uniform branch: all threads take the same path
    case OP_RELU:    return max(v, ACC(0));
    case OP_EXP:     return exp(v);
    case OP_SIGMOID: return ACC(1) / (ACC(1) + exp(-v));
    default:         return v * ACC(pc.alpha);
    }
}

void main() {
    // Grid-stride loop: supports n larger than the maximum dispatch size.
    uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    for (uint i = gl_GlobalInvocationID.x; i < pc.n; i += stride)
        y[i] = T(apply(ACC(x[i])));
}
```

A uniform `switch` is nearly free and keeps the pipeline count down; hot ops
(ReLU, GELU) can later get their own specialization constant instead.

### 6.4 The C++ side of one op

```cpp
// src/backends/vulkan/ops/elementwise.cpp
void unary(const OpArgs& args, detail::Stream& s) {
    auto& stream = static_cast<vk::Stream&>(s);
    auto& dev = stream.device();
    const auto& pipe = dev.pipelines().get({ShaderId::unary_f32, {256}});

    UnaryPush pc{static_cast<uint32_t>(args.numel()), op_code(args.op), args.alpha};
    uint32_t groups = std::min(ceil_div(pc.n, 256u), dev.caps().max_group_count_x);
    stream.dispatch(pipe, {binding(args.in(0)), binding(args.out(0))}, &pc, sizeof pc,
                    {groups, 1, 1});
    stream.barrier();
}

PRS_REGISTER_KERNEL(Op::Unary, DeviceType::Vulkan, DType::F32, unary);
```

Each op file is short: pick a variant, fill push constants, compute the grid,
dispatch. All Vulkan boilerplate stays in `core/`.

---

## 7. Debugging & profiling

- `VK_EXT_debug_utils`: name every buffer and pipeline
  (`vkSetDebugUtilsObjectNameEXT`) so validation messages say
  `"tensor#42 (matmul out)"` instead of a raw hex handle.
- **Timestamp queries** around each dispatch when `PRS_PROFILE=1`; print a
  per-op table at exit (name, count, total µs, GB/s or GFLOP/s).
- Vendor tools: Nsight Graphics / Nsight Systems (NVIDIA), Radeon GPU Profiler
  (AMD), Intel GPA, Xcode Metal capture (works through MoltenVK).
- `spirv-dis` / `spirv-cross` to inspect what the compiler produced.
- GPU-assisted validation (`VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT`)
  catches out-of-bounds buffer access in shaders — run it in nightly tests.

---

## 8. Testing the Vulkan backend

- Every op test runs on every available device and compares to the F64 CPU
  reference with per-dtype tolerances (F32: rtol 1e-5 × √K for GEMM;
  F16: 1e-2).
- Shape fuzzing: 0, 1, primes (17, 23, 19 as in the current tests), powers of
  two ± 1, and large sizes that exceed one dispatch.
- Run the whole suite with validation layers on; any validation message fails the test.
- A leak test: create and destroy the Vulkan device 100× in a loop and check
  that VMA reports zero live allocations and validation reports no leaked objects.
- CI without a GPU: Mesa **lavapipe** (`VK_ICD_FILENAMES` → lvp) runs Vulkan
  compute on the CPU — slow but checks correctness and validation.

---

## 9. Dependencies

| Library | Purpose | How |
|---|---|---|
| Vulkan headers | API | Vulkan SDK |
| volk | Runtime loading of the Vulkan loader | FetchContent, header + 1 .c |
| VMA | Memory allocation | FetchContent, single header |
| glslc / shaderc | GLSL → SPIR-V | Vulkan SDK (build time only) |
| spirv-val | Validate SPIR-V | Vulkan SDK (build time only) |
| Validation layers | Debugging | Vulkan SDK (debug/test only) |
