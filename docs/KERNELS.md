# ML Kernel Implementation Roadmap

## Goal and scope

Provide the operations needed to express useful ML algorithms as tensor graphs.
Start with dense inference and classical ML, then add training, convolutional
networks, and transformers if time permits. Full PyTorch compatibility is not a
goal.

**Priority order:** P0 → P1 → P2 → P3 → P4 → P5. Complete and test a smaller
set before expanding. CPU reference implementations come first; GPU
implementations follow for the selected backend. Not every operation needs a
standalone GPU kernel: some are metadata-only, compositions, or fused
implementations.

## Shared tensor contract

Before adding many operations, agree on:

* Float32 first; integer indices and boolean masks when indexing is introduced.
* Contiguous row-major storage first; explicit copies for unsupported layouts.
* Exact shape checks and documented scalar broadcasting initially; general
NumPy-style trailing-dimension broadcasting later.
* Explicit reduction axes and `keepdim`; full-tensor reductions can come first.
* Defined behavior for empty tensors, invalid axes, division by zero, and NaN/Inf.
* Out-of-place graph operations initially. In-place updates require alias and
lifetime rules, especially for training.
* Stable numerical formulas, operation-specific tolerances, and no promise of
bit-identical CPU/GPU reductions.

## Current implementation status

What exists today. Operations take `Tensor`s (float32 only, `mensurflow::` namespace) and run on
the device the tensors live on, through the dispatcher. Every CPU kernel also has a plain
sequential version, used as the benchmark baseline (`src/backends/cpu/sequential.cpp`).
Measured results are in [BENCHMARKS.md](BENCHMARKS.md).

|Operation|CPU|CUDA|ROCm|Metal|
|-|-|-|-|-|
|`add`, `sub`, `mul`|OpenMP, with broadcasting|cuBLAS (copy + axpy, `sdgmm`)|same, untested|MPSGraph, untested|
|`div`|OpenMP, with broadcasting|—|—|MPSGraph, untested|
|`neg`, `abs`, `square`, `relu`|OpenMP, float32; tensor views supported|—|—|—|
|`sqrt`, `exp`, `sigmoid`|OpenMP, float32; tensor views supported|—|—|—|
|`axpy`, `bias_add`|OpenMP, float32; tensor views supported|—|—|—|
|`dot`|OpenMP, double accumulation|`cublasSdot`|`rocblas_sdot`, untested|MPSGraph, untested|
|`sum`, `mean`, `max`|OpenMP, float32; full-tensor reductions|—|—|—|
|`matmul`|blocked, rows split across threads|—|—|—|

CUDA was tested on an RTX 3060; ROCm and Metal have never been built on real hardware. The
older `axpy`/`matmul` code on the `prs::` API (CUDA, Metal, OpenCL) was removed in the restart and
exists only in git history.

## P0 — Minimum useful dense inference library

This is the first implementation target: linear regression, logistic regression,
and small fully connected networks with externally supplied weights.

### Elementwise arithmetic

* \[x] `add(a, b)` — addition, including scalar broadcasting.
* \[x] `sub(a, b)` — subtraction.
* \[x] `mul(a, b)` — elementwise multiplication, not matrix multiplication.
* \[x] `div(a, b)` — elementwise division.
* \[x] `neg(x)` — negation (CPU, float32).
* \[x] `axpy(alpha, x, y)` — fused float32 `alpha\*x + y`, with broadcasting (CPU).
* \[x] `bias\_add(x, b)` — add a compact `\[features]` bias to `\[batch, features]` (CPU, float32).

### Linear algebra

* \[x] `dot(x, y)` — vector dot product.
* \[ ] `matvec(A, x)` — matrix–vector multiplication; may initially use GEMM.
* \[x] `matmul(A, B)` / GEMM — `\[M,K] × \[K,N] → \[M,N]`. Basic form done (CPU, float32); the
optional `alpha`, `beta`, and transpose flags are not.
* \[ ] Optional GEMM parameters: `alpha`, `beta`, and transpose flags, introduced
after basic multiplication works.

GEMM is the most important performance kernel for dense ML. Start with a simple
CPU reference, then blocked/parallel CPU and tiled GPU implementations. Test
rectangular matrices and dimensions not divisible by tile sizes.

### Reductions and activations

* \[x] `sum(x)` — full-tensor summation; an empty tensor returns zero (CPU, float32).
* \[x] `mean(x)` — full-tensor average; empty tensors are invalid (CPU, float32).
* \[x] `max(x)` — full-tensor maximum; empty tensors are invalid and NaNs propagate (CPU, float32).
* \[x] `relu(x)` — negative values become +0; NaNs and signed zeros are preserved (CPU, float32).
* \[x] `sigmoid(x)` — stable logistic function (CPU, float32).
* \[ ] `softmax(x, axis)` — normalized exponentials; subtract the axis maximum
before exponentiation. Implement last-axis softmax first.

### Tensor plumbing

These are necessary API/runtime operations, even when they do not launch kernels.

* \[ ] `constant`, `empty`, `zeros`, `ones`, `full`, `eye`, and `\*\_like`
variants — construct tensors.
* \[ ] `shape`, `size`, `rank`, and `identity` — metadata queries and
pass-through.
* \[ ] Host/device upload, download, and device-to-device copy where applicable.
* \[ ] `reshape` and `flatten` — metadata-only when layout permits.
* \[ ] `transpose` / `permute` — initially materialize contiguous output; views can
follow once strides are supported.
* \[ ] `contiguous` — copy a strided view into a packed row-major buffer.
* \[ ] `argmax(x, axis)` — integer predictions for classification; specify tie rules.

**Acceptance demo:** run `XW + b`, then sigmoid for binary classification or
softmax for multiclass classification. Run a two-layer ReLU network and compare
with a reference. This supports inference, not training by itself.

## P1 — Broader numerical and classical ML support

Implement these after P0 works end to end.

### Elementwise math and comparisons

* \[x] `abs` and `square` — absolute value and squaring (CPU, float32).
* \[x] `exp` and `sqrt` — exponential and square root (CPU, float32).
* \[ ] `log` and `rsqrt`.
* \[ ] `minimum`, `maximum`, and `clamp`.
* \[ ] `equal`, `less`, `greater`, and related comparisons producing boolean masks.
* \[ ] `where(mask, a, b)` — conditional element selection.
* \[ ] `tanh` — activation used by older networks and recurrent models.
* \[ ] `pow` and `reciprocal`.
* \[ ] `sin` and `cos` — rotary and sinusoidal position encodings.
* \[ ] `erf` — exact GELU.
* \[ ] `floor`, `ceil`, `round`, `trunc`, and `sign`.
* \[ ] `log1p` and `expm1` — accurate near zero.
* \[ ] `log2` and `log10`.
* \[ ] `remainder`, `fmod`, and `floor\_div` — document the sign convention for
negative inputs.
* \[ ] `tan`, `sinh`, `cosh`, and inverse trig (`asin`, `acos`, `atan`, `atan2`,
`asinh`, `acosh`, `atanh`) — rare in models, but in PyTorch's core set.
* \[ ] `isnan` and `isinf`.
* \[ ] `logical\_and`, `logical\_or`, `logical\_xor`, and `logical\_not` on boolean
masks.
* \[ ] `bitwise\_and`, `bitwise\_or`, `bitwise\_xor`, `bitwise\_not`, and shifts —
integer masks and unpacking int4 weights.
* \[ ] `cast` — dtype conversion; needed once integer, boolean, or half types exist.
`bitcast` reinterprets the bits without converting values.
* \[ ] `add\_n` — sum of many same-shaped tensors in one pass.
* \[ ] `bucketize` / `searchsorted` — map values to bins given sorted boundaries.
* \[ ] General broadcasting for elementwise operations.

### Additional reductions and linear algebra

* \[ ] `min`, `argmin`, and `sum\_of\_squares`.
* \[ ] `prod`, `any`, and `all`.
* \[ ] `cumsum` along an axis — top-p sampling and state-space models.
* \[ ] `logsumexp` as a standalone reduction.
* \[ ] `sort` and `argsort`, with defined tie and stability rules.
* \[ ] L1/L2 norms, with explicit axes.
* \[ ] Lp normalization, e.g. L2 `x / ||x||` — embedding and retrieval models.
* \[ ] Variance and standard deviation; specify population/sample convention and
use a numerically stable algorithm where practical.
* \[ ] Batched `matmul`, initially for matching batch dimensions.
* \[ ] `outer(x, y)` — outer product; also appears in gradient formulas.
* \[ ] `einsum` — lowered to permute, reshape, and batched GEMM.
* \[ ] Pairwise squared Euclidean distances, initially composed from reductions
and GEMM; clamp small negative roundoff artifacts to zero.

### Indexing and shape operations

* \[ ] `slice` with a restricted, documented indexing contract; steps (strided
slice) can follow.
* \[ ] `select` and `diagonal` — single-index and diagonal views. Also building a
diagonal matrix from a vector and overwriting a diagonal.
* \[ ] `as\_strided` — general strided view, once views are supported.
* \[ ] `concat` and `split` along a selected axis.
* \[ ] `stack` / `unstack` — join along, or split into, a new axis.
* \[ ] `gather` / row selection with integer indices and bounds validation.
* \[ ] `gather\_nd` / `scatter\_nd` — multi-dimensional index tuples.
* \[ ] Boolean-mask selection (`masked\_select`, ONNX `Compress`).
* \[ ] `one\_hot` and `hardmax` (one-hot of the argmax).
* \[ ] `unique` — output shape depends on data, like `nonzero`.
* \[ ] `broadcast\_to` / `expand`, initially materialized if zero-stride views are
unsupported.
* \[ ] `squeeze` / `unsqueeze` — metadata-only.
* \[ ] `repeat` / tile.
* \[ ] `pad` — constant, reflection, and replication modes.
* \[ ] `flip` and `roll`; `reverse\_sequence` for variable-length sequences.
* \[ ] `depth\_to\_space` / `space\_to\_depth` (pixel shuffle) — super-resolution.
`batch\_to\_space` / `space\_to\_batch` are the batch-axis equivalents.
* \[ ] `tril` / `triu` — build causal masks.
* \[ ] `arange` — index and position tensors.
* \[ ] `index\_select`, `scatter`, and `scatter\_reduce` beyond scatter-add.
* \[ ] `slice\_scatter`, `select\_scatter`, and `index\_put` — write values into part
of a tensor, e.g. KV-cache updates.
* \[ ] `masked\_scatter` — fill masked positions from a source tensor.
* \[ ] `nonzero` — output shape depends on data; needs a dynamic-shape rule.

### Losses and evaluation

* \[ ] Mean squared error (MSE).
* \[ ] Binary cross-entropy **with logits**, using a stable formula.
* \[ ] `log\_softmax`, using stable log-sum-exp.
* \[ ] Multiclass cross-entropy **with logits**, with integer class labels.
* \[ ] Negative log-likelihood loss on log-probabilities.
* \[ ] Huber / smooth-L1 loss and KL divergence.
* \[ ] Label smoothing for cross-entropy.
* \[ ] Accuracy, composed from `argmax`, comparisons, and reductions.

Loss APIs should expose `none`, `sum`, and `mean` reduction modes. The initial
version can choose one mode if it is explicit and tested.

## P2 — Training small dense models

Training needs more than forward kernels: gradients, parameter state, graph
execution rules, and an update loop. Start with manually constructed backward
graphs if a full automatic differentiation system would exceed scope.

### Backward operations

* \[ ] Elementwise gradients for arithmetic, ReLU, sigmoid, and tanh.
* \[ ] GEMM gradients using transposed GEMMs.
* \[ ] Bias gradients and reduction gradients using sum/broadcast.
* \[ ] MSE, binary cross-entropy-with-logits, and cross-entropy-with-logits gradients.
* \[ ] Softmax/log-softmax backward when required independently of fused losses.
* \[ ] `sum\_to\_shape` — reduce broadcasted gradients back to an input's shape.
* \[ ] Gradient accumulation for parameters used by multiple graph branches.
* \[ ] Scatter-add or equivalent accumulation for gather/indexing gradients;
repeated indices must accumulate correctly.

A mathematical backward formula may use existing primitives rather than a new
kernel. For example, GEMM backward can dispatch two existing GEMM operations.

### Parameter initialization and optimizer operations

* \[ ] Seeded uniform and normal random initialization.
* \[ ] `randperm` and Bernoulli sampling — data shuffling and dropout masks.
* \[ ] Xavier/He initialization built from random generation and scaling.
* \[ ] SGD parameter update.
* \[ ] Optional momentum SGD, followed by Adam with optimizer state and bias
correction.
* \[ ] AdamW — Adam with decoupled weight decay; the usual optimizer for
transformer training.
* \[ ] Gradient zeroing and optional global-norm clipping.
* \[ ] Optional dropout forward/backward with saved masks and explicit train/eval
behavior.

### Useful accumulation primitives

* \[ ] Grouped sums/counts or scatter-add for centroid updates and embeddings.
Segment reductions (`segment\_sum`, unsorted segment sum/max/min/prod) are the
same operation keyed by a segment id.
* \[ ] Define determinism guarantees when GPU atomic accumulation is used.

**Runtime requirements:** save or recompute values needed by backward; distinguish
parameters from ordinary inputs; manage gradient lifetimes; sequence stateful
updates after gradient computation. Random state and parameter updates must not be
removed or reordered as if they were pure arithmetic.

**Acceptance demo:** train a small linear/logistic regression model, then an MLP.
Check gradients against finite differences on small inputs and verify decreasing
loss on a controlled dataset.

## P3 — Convolutional networks and normalization

### Convolution and pooling

* \[ ] `conv2d` forward, with one documented layout, e.g. NCHW.
* \[ ] Stride and padding, then dilation and groups if needed.
* \[ ] Simple direct convolution reference implementation.
* \[ ] Optional im2col + GEMM implementation; account for temporary memory costs.
`col2im` is its inverse, used by transposed convolution and backward.
* \[ ] Max pooling (returning indices, for backward and `max\_unpool`) and average
pooling.
* \[ ] Adaptive / global average pooling — classifier heads, e.g. ResNet. Also
global max pooling and Lp pooling.
* \[ ] `conv1d` — audio and sequence models, e.g. Whisper's front end and Mamba's
short causal convolution.
* \[ ] Depthwise convolution kernel (`groups == channels`) — mobile CNNs; the
general grouped-convolution path is slow for this case.
* \[ ] Transposed convolution (`conv\_transpose1d/2d`) — decoders and U-Net
upsampling.
* \[ ] 3D convolution, 3D transposed convolution, and 3D pooling — video and
medical imaging.
* \[ ] Deformable convolution — learned sampling offsets; detection models.
* \[ ] Upsampling: nearest, bilinear, and bicubic interpolation.
* \[ ] `grid\_sample` — bilinear sampling at given coordinates; spatial
transformers and optical flow. `affine\_grid` builds the coordinates.
* \[ ] Non-maximum suppression, ROI Align, and ROI pooling — object detection.
* \[ ] Convolution and pooling backward if training is in scope.

### Normalization

* \[ ] Layer normalization forward, including epsilon and learned scale/bias.
* \[ ] Batch normalization inference using stored running statistics.
* \[ ] Batch normalization training and running-statistic updates, if needed.
* \[ ] Group normalization — diffusion U-Nets and small-batch training. Instance
normalization is the case of one group per channel.
* \[ ] Local response normalization — older CNNs such as AlexNet.
* \[ ] Normalization backward for whichever training operations are supported.

### Additional activations

* \[ ] `silu` (swish), `x \* sigmoid(x)` — LLaMA-family MLPs and diffusion U-Nets.
* \[ ] `leaky\_relu`, `elu`, and `softplus`.
* \[ ] `hardswish`, `hardsigmoid`, and `relu6` / `hardtanh` — mobile CNNs.
* \[ ] `prelu` — leaky ReLU with a learned slope.
* \[ ] Squared ReLU, `relu(x)^2` — used in some recent LLMs.
* \[ ] `selu`, `celu`, `mish`, `softsign`, `shrink`, and `thresholded\_relu`.

### Recurrent networks

* \[ ] Vanilla RNN, LSTM, and GRU cells — composed from GEMM, sigmoid, and tanh;
a fused cell kernel is optional.

**Acceptance demo:** a small CNN inference graph. Training it requires the matching
backward operations; forward-only support is not sufficient.

## P4 — Transformer inference and advanced extensions

This is a stretch tier. Reuse earlier primitives before writing specialized fused
kernels.

* \[ ] Embedding lookup, built on gather.
* \[ ] Batched GEMM with transpose support for attention scores and weighted sums.
* \[ ] Scaled dot-product attention: `softmax(QK^T / sqrt(d) + mask) V`.
* \[ ] Causal, padding, and sliding-window masks, with defined all-masked-row
behavior.
* \[ ] Relative position bias and ALiBi — biases added to attention scores
(T5, SAM, BLOOM).
* \[ ] Window partition / unpartition for windowed attention (Swin, SAM).
* \[ ] GELU activation; specify exact or approximate formula.
* \[ ] LayerNorm or RMSNorm as required by the chosen model. Variants: Gemma's
`(1 + w)` scale and gated RMSNorm.
* \[ ] Positional encodings or rotary positional embeddings, model-dependent.
* \[ ] Gated MLP activations: SwiGLU and GeGLU, `act(xW) \* (xV)` — standard in
LLaMA-family models.
* \[ ] Grouped-query and multi-query attention — several query heads share one
K/V head, shrinking the KV cache.
* \[ ] Multi-head latent attention (DeepSeek) — K/V compressed into a low-rank
latent vector to shrink the KV cache further.
* \[ ] Optional KV cache for incremental decoding, with explicit state/lifetimes.
* \[ ] Paged KV cache — fixed-size blocks plus a block table, as in vLLM's
PagedAttention; needs a row-scatter write kernel.
* \[ ] Optional top-k selection for prediction or sampling.
* \[ ] Sampling: temperature, top-p (sort + cumsum), and multinomial draw.
* \[ ] Logit processing before sampling: repetition / presence penalties and
logit soft-capping (`cap \* tanh(x / cap)`).

Advanced performance work, only after a correct baseline:

* Register tiling, vectorized loads, and double-buffered tiles for the 16×16 GEMM.
* Subgroup-reduction GEMV and cooperative-matrix (tensor core) GEMM on Vulkan, as
in the group's NT-2026 paper; neither is in the repository yet.
* Fused attention to reduce intermediate memory traffic: FlashAttention-style
tiling with an online softmax, so the full score matrix is never stored.
* Float16/BFloat16 storage and mixed-precision accumulation on supported hardware.
* Quantized matrix multiplication with documented scale/zero-point semantics.
* Weight-only int4/int8 GEMM with per-group scales (AWQ, GPTQ, llama.cpp block
formats), plus matching quantize/dequantize kernels; FP8 where hardware has it.
* Integer convolution and dynamic (runtime-computed scale) quantization, as in
ONNX `ConvInteger` and `DynamicQuantizeLinear`.
* Fake quantization — simulated rounding for quantization-aware training.
* Fused LLM blocks: residual add + RMSNorm, and SiLU-and-multiply for SwiGLU.
* Sparse operations, only if a target workload actually needs them, e.g.
sparse–dense matmul (SpMM) for graph neural networks, and sparse↔dense
conversion.

**Acceptance demo:** a small transformer block with fixed weights and reference
outputs. A full language-model serving system is outside the initial scope.

## P5 — Newer model families (research stretch)

Operations behind architectures common in current models but outside a classic
transformer. Each is a substantial project; pick at most one.

### Mixture of experts

* \[ ] Router — softmax or sigmoid scores, then top-k experts per token. Grouped
top-k (DeepSeek-V2/V3) first picks a few groups of experts — in DeepSeek-V2,
the experts on each device — then the top experts within those groups.
* \[ ] Token dispatch and combine — gather tokens per expert, then weighted
scatter-add back.
* \[ ] Grouped / indexed GEMM — one call over many small expert matrices with
different token counts.

### State-space and linear-attention models

* \[ ] Selective scan — Mamba's input-dependent recurrence, computed as a parallel
scan.
* \[ ] Short causal `conv1d` with carried state for incremental decoding.
* \[ ] Linear-attention recurrences: RWKV, gated linear attention, gated delta net.

### Diffusion models

* \[ ] Sinusoidal timestep embedding.
* \[ ] Cross-attention — scaled dot-product attention with K/V from a second input.
* \[ ] U-Net blocks: GroupNorm + SiLU + conv, upsampling, skip-connection concat.

### Signal processing and recommendation

* \[ ] Real FFT (1D and 2D), inverse FFT, and STFT — audio front ends. Needs a
complex dtype with `real`, `imag`, and magnitude.
* \[ ] Hann, Hamming, and Blackman windows, and a mel filterbank matrix — log-mel
spectrograms, e.g. Whisper's input.
* \[ ] `embedding\_bag` — summed or averaged lookups for recommendation models.

**Acceptance demo:** one MoE layer or one Mamba block, checked against a reference
implementation.

## Which kernels unlock which algorithms?

|Workload|Required operations|
|-|-|
|Linear regression inference|GEMM/matvec, bias addition|
|Logistic regression inference|GEMM/matvec, bias addition, sigmoid|
|Multiclass linear classifier|GEMM, bias addition, softmax, argmax|
|Dense neural network inference|GEMM, bias addition, ReLU/sigmoid/tanh|
|Regression/classification training|Forward operations, losses, gradients, SGD|
|K-means|Distances, argmin, grouped sums/counts, division|
|CNN inference|Convolution, activation, pooling, dense layers|
|Transformer block inference|Embeddings, batched GEMM, masking, softmax, normalization, activation|
|LLaMA-style decoder|RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache, sampling|
|Mixture-of-experts LLM|Router top-k, token dispatch/combine, grouped GEMM|
|Mamba / state-space model|Selective scan, causal conv1d, SiLU, softplus|
|Diffusion U-Net|Convolution, GroupNorm, SiLU, upsampling, timestep embedding, attention|
|Speech, Whisper-style|STFT, conv1d, GELU, transformer encoder–decoder|
|Mobile CNN|Depthwise convolution, hardswish, global average pooling|
|Object detection|Convolution, upsampling, non-maximum suppression, ROI Align|
|RNN / LSTM|GEMM, sigmoid, tanh, elementwise arithmetic|

Other ML algorithms can require additional numerical methods. For example, robust
least-squares solvers may need QR/SVD, PCA needs eigendecomposition or SVD, and
Gaussian processes need Cholesky factorization, triangular solves, and
determinants. Those are substantial extensions, not implied by basic BLAS support.
ONNX's string, sequence, optional, and image-decoding operators and LiteRT's
hash tables are preprocessing rather than numerical kernels and are out of scope.
So are multi-GPU collectives (all-reduce, all-gather) and tensor-parallel layers,
since distributed execution is not a goal.

## Implementation strategy for every operation

1. Specify input/output shapes, dtype, broadcasting, and error cases.
2. Implement a simple CPU reference and small hand-checkable tests.
3. Register shape inference and execution in the graph runtime.
4. Add an optimized CPU implementation where measurements justify it.
5. Add the selected GPU backend implementation for important workloads.
6. Test CPU/GPU agreement using operation-appropriate tolerances.
7. Benchmark a shape sweep, including small inputs and awkward dimensions.
8. Add tuning candidates only when multiple correct configurations exist.

Maintain a support table per operation: CPU reference, optimized CPU, GPU backend,
backward support, and tests. Report missing implementations explicitly. Forced
GPU execution should fail for unsupported operations; automatic placement may
choose CPU and insert the required transfers.

## Optimization order

Optimize the dominant costs rather than every operation equally:

1. Persistent GPU buffers and elimination of unnecessary transfers.
2. GEMM, followed by reductions and softmax.
3. Buffer reuse and avoidance of needless tensor copies.
4. Fused elementwise chains and GEMM + bias + activation.
5. Convolution or attention, only when included in the target workload.

Fusion is an execution optimization, not a requirement for separate public APIs.
For example, `matmul → bias\_add → relu` can remain three graph nodes while an
optimized plan dispatches one supported fused implementation.

**Recommended course-project target:** P0, the P1 operations needed for losses and
benchmarks, and a well-tested CPU/GPU execution path with autotuning. Add P2 only
if training is an agreed goal; P3, P4, and P5 are optional expansions.
