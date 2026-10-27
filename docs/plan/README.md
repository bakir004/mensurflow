# PRS ML/BLAS Library — Design Plan

A C++ library for tensors, BLAS and ML ops that detects the user's hardware
and runs on the best backend: cuBLAS on NVIDIA, rocBLAS on AMD,
MPS/MPSGraph on Apple, our own kernels on CPU — with **Vulkan compute as the
portable GPU fallback** (Intel, ARM, Qualcomm, or any vendor without its
runtime installed).

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers, public `Tensor`/`Device` API, device autodetection, dispatcher and per-op fallback, folder structure, phases |
| [KERNELS_ROADMAP.md](KERNELS_ROADMAP.md) | Every kernel from the vendor BLAS/DNN libraries, ranked by necessity (P0–P3) and difficulty (★1–5), plus build order by milestone |
| [VULKAN_DESIGN.md](VULKAN_DESIGN.md) | Vulkan backend: RAII handles and destruction order, device/feature selection, VMA memory and buffer pooling, command batching and barriers, descriptors/push constants/specialization constants, shader-writing rules, debugging and testing |

## Decisions in one paragraph

Device-resident, asynchronous `Tensor`s replace the current
`std::vector` API (kept as a convenience wrapper). Ops dispatch through an
`(op, device, dtype) → kernel` table, so each backend registers only what
it has. Vendor backends call vendor libraries for BLAS/conv and use our own
kernels for everything else. The Vulkan and CPU backends must cover every op
and are checked by a coverage test. In Vulkan, every handle is owned by a
move-only RAII wrapper, memory comes from VMA with a caching pool, many
dispatches are batched per submit with timeline semaphores, and shaders are
GLSL compiled to SPIR-V at build time, embedded in the binary, and tuned
through specialization constants.

## First steps

1. Move `vulkan/` into `src/backends/vulkan/` and split its `Context` into the
   RAII classes from VULKAN_DESIGN.md §1.
2. Add `Tensor`, `Device`, `Dispatcher`; reimplement the existing
   `axpy`/`matmul` on top of them.
3. Work through Milestone 1 in KERNELS_ROADMAP.md on CPU and Vulkan until an
   MLP trains on MNIST.
