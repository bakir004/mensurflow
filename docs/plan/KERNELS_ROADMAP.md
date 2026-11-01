# Kernel Roadmap

The kernels below are the union of what the vendor libraries offer, grouped and
ranked so we build what matters first:

- **cuBLAS / cuBLASLt** (NVIDIA): BLAS L1–L3, batched/strided GEMM, GEMM with fused epilogues
- **cuDNN** (NVIDIA): conv, pooling, normalization, softmax, activations, attention, RNN
- **rocBLAS / hipBLASLt / MIOpen** (AMD): the same list as cuBLAS / cuBLASLt / cuDNN
- **MPS / MPSGraph** (Apple): matrix multiply, conv, pooling, norms, reductions, gather/scatter, random, plus a graph compiler that fuses ops
- **oneMKL / oneDNN** (Intel / CPU), **OpenBLAS / BLIS / Accelerate** (CPU): BLAS + LAPACK + DNN primitives
- **cuSOLVER / rocSOLVER / LAPACK**: factorizations and solvers
- **cuRAND / rocRAND**: random number generation

### How to read the tables

**Necessity**

| Level | Meaning |
|---|---|
| **P0** | Needed to train an MLP. Build first, on every backend. |
| **P1** | Needed for CNNs and standard training (Adam, norms, conv). |
| **P2** | Needed for transformers at good speed, plus fast paths. |
| **P3** | Full BLAS/LAPACK coverage, niche ML ops. Do last, or never. |

**Difficulty** (for a hand-written Vulkan/CUDA kernel)

| Score | Meaning |
|---|---|
| ★☆☆☆☆ | One thread per element, no communication. An afternoon. |
| ★★☆☆☆ | Workgroup reduction or shared memory, one pass. |
| ★★★☆☆ | Multi-pass or tricky indexing (strides, broadcasting, scans). |
| ★★★★☆ | Tiling + register blocking + tuning to get within 2× of the vendor. |
| ★★★★★ | State of the art (FlashAttention, Winograd, tensor/coop-matrix GEMM, SVD). |

**Vendor column**: which library provides it, i.e. where the vendor backends
just call a library instead of shipping our own kernel. "—" means everyone
writes it by hand (including PyTorch).

---

## 1. BLAS Level 1 (vector–vector)

| Kernel | Op | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|---|
| `axpy` | y = αx + y | P0 | ★☆☆☆☆ | cuBLAS, rocBLAS, MPS(graph), BLAS | Already done. |
| `scal` | x = αx | P0 | ★☆☆☆☆ | all BLAS | Same shader as elementwise mul-by-scalar. |
| `copy` | y = x | P0 | ★☆☆☆☆ | all BLAS | Use `vkCmdCopyBuffer` when contiguous; strided copy kernel otherwise. |
| `dot` | xᵀy | P0 | ★★☆☆☆ | all BLAS | First reduction kernel. Subgroup reduce → shared → atomics or second pass. |
| `asum` | Σ\|xᵢ\| | P1 | ★★☆☆☆ | all BLAS | Reuses the reduction template. |
| `nrm2` | ‖x‖₂ | P1 | ★★☆☆☆ | all BLAS | Scale to avoid overflow (LAPACK-style) or accumulate in F64 where supported. Needed for grad clipping. |
| `iamax` / `iamin` | argmax \|xᵢ\| | P2 | ★★☆☆☆ | all BLAS | Reduce (value, index) pairs; ties → lowest index. |
| `swap` | x ↔ y | P3 | ★☆☆☆☆ | all BLAS | |
| `rot`, `rotg`, `rotm`, `rotmg` | Givens rotations | P3 | ★☆☆☆☆ | all BLAS | Only for LAPACK-style algorithms. |

## 2. BLAS Level 2 (matrix–vector)

| Kernel | Op | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|---|
| `gemv` | y = αAx + βy (N and T) | P0 | ★★☆☆☆ | all BLAS | Transpose case needs a different access pattern (one workgroup per row vs. column-tiles). Memory-bound: aim for bandwidth. |
| `ger` | A += αxyᵀ | P1 | ★☆☆☆☆ | all BLAS | Weight gradient of a batch-1 linear layer. |
| `symv`, `hemv` | symmetric mat-vec | P3 | ★★☆☆☆ | all BLAS | |
| `trmv` | triangular mat-vec | P3 | ★★☆☆☆ | all BLAS | |
| `trsv` | triangular solve | P3 | ★★★☆☆ | all BLAS | Sequential dependency; block it. |
| `gbmv`, `sbmv`, `tbmv`, `tbsv`, `spmv`, `tpmv`, `tpsv`, `syr`, `syr2`, `spr` | banded / packed variants | P3 | ★★☆☆☆ | all BLAS | Completeness only. |

## 3. BLAS Level 3 (matrix–matrix) — the most important section

| Kernel | Op | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|---|
| `gemm` naive/tiled | C = αop(A)op(B) + βC | P0 | ★★★☆☆ | all BLAS, MPS | Have a 16×16 shared tile version. Add α, β, transposes (NN, NT, TN, TT). |
| `gemm` register-blocked | same | P1 | ★★★★☆ | — | Each thread computes a 4×4/8×8 micro-tile, vec4 loads, double-buffered shared memory. Target: ≥50% of cuBLAS on the same GPU. |
| `gemm_batched` / `gemm_strided_batched` | C[i] = A[i]B[i] | P1 | ★★☆☆☆ | cuBLAS, rocBLAS, MPS | Use `gl_WorkGroupID.z` for batch. Needed for attention and batched matmul with broadcasting. |
| `gemm` + epilogue | C = act(AB + bias) | P2 | ★★★☆☆ | cuBLASLt, hipBLASLt, MPSGraph | Fuse bias, ReLU/GELU, residual add into the store. Big win for Linear layers. |
| `gemm` cooperative-matrix | same, F16/BF16 → F32 | P2 | ★★★★★ | — | `VK_KHR_cooperative_matrix` (tensor cores / WMMA / Intel XMX). Fallback to register-blocked when unsupported. |
| `gemm` split-K | same, small M,N large K | P2 | ★★★☆☆ | cuBLAS (internal) | Partial sums + reduction. Needed for weight-gradient shapes. |
| `syrk`, `herk`, `syr2k` | C = αAAᵀ + βC | P3 | ★★★☆☆ | all BLAS | GEMM that writes only one triangle. |
| `symm`, `hemm` | symmetric × general | P3 | ★★★☆☆ | all BLAS | |
| `trmm` | triangular × general | P3 | ★★★☆☆ | all BLAS | |
| `trsm` | solve op(A)X = αB | P3 | ★★★★☆ | all BLAS | Blocked: small triangle solves + GEMM updates. Needed by Cholesky/LU. |

## 4. Elementwise (pointwise)

One templated shader per category, generated for each dtype. Must support
**strides and broadcasting** (rank ≤ 8): pass shapes/strides in a small
storage buffer or push constants, and have a fast path for contiguous same-shape inputs.

| Kernel | Ops | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|---|
| Binary arithmetic | add, sub, mul, div, pow, max, min | P0 | ★★☆☆☆ (★★★ with broadcasting) | MPSGraph, cuDNN (partially) | Broadcasting index math is the hard part — write it once in `common/indexing.glsl`. |
| Unary math | neg, abs, exp, log, sqrt, rsqrt, sin, cos, tanh, sigmoid, reciprocal | P0 | ★☆☆☆☆ | MPSGraph | |
| Activations | relu, leaky_relu, gelu (erf + tanh approx), silu/swish, elu, softplus | P0 (relu) / P1 | ★☆☆☆☆ | cuDNN, MIOpen, MPSGraph | |
| Activation backward | relu', gelu', silu', sigmoid', tanh' | P0 / P1 | ★☆☆☆☆ | cuDNN, MIOpen | |
| Comparison | eq, ne, lt, le, gt, ge → Bool | P1 | ★☆☆☆☆ | MPSGraph | |
| `where` / select | out = c ? a : b | P1 | ★☆☆☆☆ | MPSGraph | Masking in attention. |
| `clamp` | | P1 | ★☆☆☆☆ | MPSGraph | |
| `fill` | constant | P0 | ★☆☆☆☆ | `vkCmdFillBuffer` (32-bit patterns) | zeros/ones. |
| `cast` | F32↔F16↔BF16↔I32 | P1 | ★☆☆☆☆ | MPSGraph | Rounding: round-to-nearest-even for BF16. |
| `arange`, `linspace` | | P2 | ★☆☆☆☆ | MPSGraph | |

## 5. Reductions & scans

| Kernel | Ops | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|---|
| Full reduction | sum, mean, max, min, prod | P0 | ★★☆☆☆ | MPSGraph, cuDNN `reduceTensor` | Generalize the `dot` reduction. |
| Axis reduction | same, along `dim`, keepdim | P0 | ★★★☆☆ | MPSGraph, cuDNN | Two strategies: innermost dim (one workgroup per row) vs. outer dim (one thread per column, coalesced). Needed for bias gradients. |
| `argmax` / `argmin` | | P1 | ★★☆☆☆ | MPSGraph | Accuracy metric. |
| `var` / `std` | | P2 | ★★★☆☆ | MPSGraph | Welford algorithm for stability. |
| `cumsum` / `cumprod` (scan) | | P2 | ★★★★☆ | MPSGraph | Decoupled look-back or 3-phase block scan. Subgroup scan helps. |
| `topk` | | P2 | ★★★★☆ | MPSGraph | Sampling in LLMs. Start with a bitonic sort for small k. |
| `sort` / `argsort` | | P3 | ★★★★☆ | MPSGraph, CUB/rocPRIM | Radix sort is the serious version. |

## 6. Neural-network ops

| Kernel | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|
| `softmax` / `log_softmax` (fwd + bwd) | P0 | ★★☆☆☆ | cuDNN, MIOpen, MPS | Numerically stable (subtract max). One workgroup per row; online softmax (single pass for max+sum) as an upgrade. |
| `cross_entropy` (fused log-softmax + NLL, fwd + bwd) | P0 | ★★☆☆☆ | — | Fusing avoids materializing probabilities. Gradient is `softmax − one_hot`. |
| `mse_loss` (fwd + bwd) | P0 | ★☆☆☆☆ | — | Elementwise + reduction. |
| `bias_add` / bias grad | P0 | ★☆☆☆☆ | cuDNN | Gradient is an axis reduction. |
| `layer_norm` (fwd + bwd) | P1 | ★★★☆☆ | cuDNN (v9), MIOpen, MPSGraph | Welford mean/var per row; backward needs saved mean/rstd and two reductions. |
| `rms_norm` (fwd + bwd) | P1 | ★★☆☆☆ | — | Simpler LayerNorm; used by modern LLMs. |
| `batch_norm` (train + eval, fwd + bwd) | P1 | ★★★☆☆ | cuDNN, MIOpen, MPS | Reduction over N,H,W per channel; running statistics update. |
| `group_norm` | P2 | ★★★☆☆ | MIOpen, MPSGraph | |
| `dropout` (fwd + bwd) | P1 | ★★☆☆☆ | cuDNN | Needs the RNG below; store mask or regenerate from seed+offset. |
| `conv2d` via im2col + GEMM (fwd) | P1 | ★★★☆☆ | cuDNN, MIOpen, MPS | Easiest correct conv. Memory hungry. |
| `conv2d` backward (data + weight) | P1 | ★★★★☆ | cuDNN, MIOpen, MPS | col2im with atomics or transposed-conv formulation. |
| `conv2d` implicit GEMM (no im2col buffer) | P2 | ★★★★☆ | cuDNN | Index on the fly inside the GEMM tile loader. |
| `conv2d` direct / depthwise | P2 | ★★★☆☆ | cuDNN, MIOpen | Depthwise is memory-bound and does badly as GEMM. |
| `conv2d` Winograd F(2×2,3×3) | P3 | ★★★★★ | cuDNN, MIOpen | 3×3 fast path. |
| `conv1d`, `conv3d`, `conv_transpose2d` | P2 / P3 | ★★★☆☆ | cuDNN, MIOpen, MPS | Reuse the im2col machinery. |
| `max_pool2d` / `avg_pool2d` (fwd + bwd) | P1 | ★★☆☆☆ | cuDNN, MIOpen, MPS | Max pool backward needs saved indices. |
| `adaptive_avg_pool2d` | P2 | ★★☆☆☆ | MPSGraph | Global pooling in CNN heads. |
| `upsample` (nearest, bilinear) | P2 | ★★☆☆☆ | MPSGraph | |
| `embedding` (fwd + bwd) | P1 | ★★☆☆☆ | — | Forward is gather; backward is scatter-add (atomics on F32; F16 needs CAS loop or sort). |
| Scaled dot-product attention (naive: GEMM → softmax → GEMM) | P1 | ★★☆☆☆ | — | Composed from batched GEMM + masked softmax. Correctness reference. |
| FlashAttention (fused, tiled, online softmax) fwd | P2 | ★★★★★ | cuDNN SDPA, MPSGraph SDPA | Main transformer speed-up; O(N) memory. |
| FlashAttention bwd | P3 | ★★★★★ | cuDNN SDPA | |
| Rotary embeddings (RoPE) | P2 | ★★☆☆☆ | — | Elementwise with index-dependent angles. |
| RNN / LSTM / GRU cells | P3 | ★★★☆☆ | cuDNN, MIOpen, MPS | Compose from GEMM + pointwise first. |

## 7. Optimizers (fused, in-place)

| Kernel | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|
| SGD (+ momentum, weight decay) | P0 | ★☆☆☆☆ | — | One pass over params, grads, momentum. |
| Adam / AdamW | P1 | ★☆☆☆☆ | — | Fused: one kernel updates param, m, v. Bias correction via push constants. |
| Multi-tensor apply | P2 | ★★★☆☆ | — | One dispatch for all parameters (array of buffer device addresses) instead of one per tensor. Large real-world speed-up. |
| Gradient clipping by global norm | P2 | ★★☆☆☆ | — | nrm2 over many tensors + scal. |

## 8. Data movement & indexing

| Kernel | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|
| Strided copy / `contiguous` | P0 | ★★☆☆☆ | — | Materializes views (transpose, slice). Every other kernel can then assume contiguous input at first. |
| `transpose` 2D (tiled, coalesced) | P0 | ★★☆☆☆ | cuBLAS `geam`, MPSGraph | Shared-memory tile with +1 padding against bank conflicts. |
| `permute` N-D | P1 | ★★★☆☆ | cuTENSOR, MPSGraph | General case of the strided copy. |
| `concat`, `split`, `slice`, `pad` | P1 | ★★☆☆☆ | MPSGraph | |
| `gather` / `index_select` | P1 | ★★☆☆☆ | MPSGraph | |
| `scatter` / `scatter_add` / `index_add` | P2 | ★★★☆☆ | MPSGraph | Atomics; F16 atomics need `VK_EXT_shader_atomic_float2`. |
| `one_hot` | P1 | ★☆☆☆☆ | MPSGraph | |
| `tril`/`triu` masks | P2 | ★☆☆☆☆ | MPSGraph | Causal mask. |

## 9. Random number generation

| Kernel | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|
| Philox4x32-10 counter-based RNG | P1 | ★★☆☆☆ | cuRAND, rocRAND, MPSGraph | Counter-based → stateless, reproducible across backends. Implement once in a `.glsl` include. Needs 32×32→64 multiply (`umulExtended`). |
| `uniform`, `normal` (Box–Muller), `bernoulli` | P1 | ★☆☆☆☆ | same | Weight init (Kaiming/Xavier) and dropout. |

Aim for **bit-identical** random streams across CPU/Vulkan/CUDA for the same
seed — makes cross-backend testing much easier.

## 10. Quantization & mixed precision

| Kernel | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|
| F16/BF16 storage + F32 accumulate for all P0/P1 ops | P2 | ★★☆☆☆ | all | Requires `storageBuffer16BitAccess`; arithmetic in F16 needs `shaderFloat16`. |
| Loss scaling (check inf/nan, scale/unscale) | P2 | ★★☆☆☆ | — | |
| INT8 GEMM (dp4a / integer dot product) | P3 | ★★★★☆ | cuBLASLt, hipBLASLt | `VK_KHR_shader_integer_dot_product`. |
| 4-bit weight dequant-GEMM (Q4 blocks) | P3 | ★★★★☆ | — | llama.cpp-style inference. |

## 11. Dense linear algebra (LAPACK subset)

| Kernel | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|
| `potrf` Cholesky | P3 | ★★★★☆ | cuSOLVER, rocSOLVER, MPS | Blocked: potrf on diagonal block + trsm + syrk. |
| `getrf` LU with partial pivoting | P3 | ★★★★☆ | cuSOLVER, rocSOLVER, MPS | |
| `getrs`, `potrs` solves | P3 | ★★★☆☆ | same | trsm-based. |
| `geqrf` QR (Householder) | P3 | ★★★★☆ | cuSOLVER, rocSOLVER | |
| `gesvd`, `syevd` | P3 | ★★★★★ | cuSOLVER, rocSOLVER | Do on CPU (LAPACK) and copy, unless the course demands it. |
| matrix inverse | P3 | ★★★☆☆ | same | getrf + getri. |

## 12. Sparse (optional)

| Kernel | Nec. | Diff. | Vendor | Notes |
|---|---|---|---|---|
| CSR SpMV | P3 | ★★★☆☆ | cuSPARSE, rocSPARSE | Graph neural nets. |
| CSR SpMM | P3 | ★★★★☆ | cuSPARSE, rocSPARSE | |

---

## Build order (combined necessity × difficulty)

Ordered so every step unlocks something you can actually run.

### Milestone 1 — "train an MLP on MNIST" (all P0)

The CPU reference implementation now covers every operation in the P0 rows
above: BLAS-1 copy/scal/axpy/dot, GEMV and GEMM variants, P0 unary and binary
math, full and axis reductions, stable softmax/log-softmax and losses, bias
gradients, SGD, contiguous copies, and transpose views. GPU backend
registration and backend-specific performance work remain separate.
1. `fill`, `copy`, strided copy/`contiguous`
2. Unary + binary elementwise (contiguous, then broadcasting)
3. `scal`, `axpy` (done)
4. Full and axis reductions (`sum`, `mean`, `max`) — start from `dot`
5. `gemm` with transposes + α/β (upgrade existing tiled kernel), `gemv`
6. `transpose` 2D
7. `relu` + backward, `bias_add` + backward
8. `softmax`, `log_softmax`, fused `cross_entropy` fwd/bwd, `mse_loss`
9. SGD update

### Milestone 2 — "train a small CNN" (P1)
10. Philox RNG, `uniform`/`normal`, `dropout`
11. `gemm_strided_batched`
12. `im2col`/`col2im` + conv2d fwd/bwd
13. `max_pool2d` / `avg_pool2d` fwd/bwd
14. `batch_norm`, `layer_norm`, `rms_norm` fwd/bwd
15. `argmax`, comparisons, `where`, `clamp`, `cast`
16. `gather`/`index_select`, `embedding`, `concat`/`split`/`slice`, `permute`, `one_hot`
17. Adam/AdamW, `nrm2`, `asum`, `ger`
18. Register-blocked GEMM (performance pass)

### Milestone 3 — "tiny transformer, fast" (P2)
19. Naive attention from batched GEMM + masked softmax (P1 really — use as reference)
20. GEMM epilogue fusion (bias + GELU)
21. F16/BF16 storage for all ops
22. Cooperative-matrix GEMM
23. FlashAttention forward
24. RoPE, `tril`/`triu`, `topk`, `cumsum`, `var`/`std`
25. Split-K GEMM, implicit-GEMM conv, depthwise conv
26. Multi-tensor optimizer, grad clipping

### Milestone 4 — completeness (P3)
27. Remaining BLAS L1–L3 (`trsm` first)
28. LAPACK subset (Cholesky → LU → QR)
29. INT8 / 4-bit quantized GEMM
30. Sparse, sort, FlashAttention backward, Winograd

---

## Per-backend coverage strategy

| Category | CPU | CUDA | ROCm | Metal | Vulkan |
|---|---|---|---|---|---|
| BLAS L1–L3 | own kernels, optional OpenBLAS/Accelerate | cuBLAS/cuBLASLt | rocBLAS/hipBLASLt | MPS / MPSGraph | **own** |
| Conv / pool / norms | own | cuDNN (optional) or own | MIOpen (optional) or own | MPS / MPSGraph | **own** |
| Elementwise / reductions | own (OpenMP + SIMD) | own `.cu` | hipified `.cu` | MPSGraph or `.metal` | **own** |
| Softmax / losses / optimizers | own | own `.cu` | hipified | `.metal` | **own** |
| Attention | own (naive) | cuDNN SDPA or own | own | MPSGraph SDPA | **own** |
| RNG | own Philox | own Philox (not cuRAND, for bit-identical results) | same | own | **own** |
| LAPACK | LAPACK | cuSOLVER | rocSOLVER | MPS / CPU | CPU fallback |

The CPU column is also the **reference implementation** for tests: keep a
naive, obviously-correct F64 version of every op in `tests/reference/`,
separate from the optimized CPU backend.
