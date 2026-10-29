# Technology Overview

## The big picture

We’re building a small C++ ML engine with these layers:

```text
Our API: model.addNode("matmul", A, B)
                  ↓
Graph runtime: dependencies, optimization, scheduling       P26
                  ↓
Kernel library: operations and tuned implementations       P27
                  ↓
           CPU or GPU execution
```

## Existing technologies

### CPU

| Technology | What it provides |
|---|---|
| **C++** | Our basic CPU implementations |
| **OpenMP** | Splits CPU work across multiple cores |
| **OpenBLAS / MKL / Apple Accelerate** | Ready-made, optimized CPU linear algebra |

OpenMP helps **parallelize our code**; BLAS libraries provide **operations already implemented and optimized**.

### GPU

| Hardware | Programming platform | Ready-made math libraries |
|---|---|---|
| NVIDIA | **CUDA** | **cuBLAS**, cuDNN |
| AMD | **ROCm / HIP** | **rocBLAS**, MIOpen |
| Apple | **Metal** | **MPS**, MPSGraph |
| Multiple GPU vendors | **Vulkan Compute** | Less standardized ML/math ecosystem |

- **CUDA, HIP, Metal, Vulkan Compute:** let us write and execute GPU kernels.
- **cuBLAS, rocBLAS, MPS:** save us from writing certain kernels ourselves.
- **Vulkan Compute:** cross-vendor GPU programming, not a universal CPU/GPU/accelerator runtime. On Mac, it typically goes through MoltenVK.

**BLAS** is a family of standard linear algebra operations, not one particular library.

## What we actually build

### P27: How to calculate an operation efficiently

For something like matrix multiplication:

1. Implement a correct CPU version.
2. Parallelize/optimize it.
3. Implement a GPU version.
4. **Autotune:** measure different implementations and configurations, then cache the best choice.

Existing math libraries can be comparison baselines or optional implementations. Clearly distinguish their work from our own kernels.

### P26: Where and when to calculate it

For a graph like:

```text
add → multiply → negate
```

Our runtime:

1. Validates shapes and dependencies.
2. Removes unnecessary work.
3. Chooses CPU or GPU.
4. Moves data when needed.
5. Runs operations in dependency order.
6. Eventually executes independent branches concurrently.

**GPU execution is not automatically faster:** transfers, launches, synchronization, and queued work all matter. Keep intermediate tensors on the GPU when that saves transfers.

**We’re not rebuilding CUDA or Metal. We’re building a numerical library and a smart graph executor on top of existing execution platforms.**
