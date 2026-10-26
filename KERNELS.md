# ML Kernel Implementation Roadmap

## Goal and scope

Provide the operations needed to express useful ML algorithms as tensor graphs.
Start with dense inference and classical ML, then add training, convolutional
networks, and transformers if time permits. Full PyTorch compatibility is not a
goal.

**Priority order:** P0 → P1 → P2 → P3 → P4. Complete and test a smaller set before
expanding. CPU reference implementations come first; GPU implementations follow
for the selected backend. Not every operation needs a standalone GPU kernel:
some are metadata-only, compositions, or fused implementations.

## Shared tensor contract

Before adding many operations, agree on:

- Float32 first; integer indices and boolean masks when indexing is introduced.
- Contiguous row-major storage first; explicit copies for unsupported layouts.
- Exact shape checks and documented scalar broadcasting initially; general
  NumPy-style trailing-dimension broadcasting later.
- Explicit reduction axes and `keepdim`; full-tensor reductions can come first.
- Defined behavior for empty tensors, invalid axes, division by zero, and NaN/Inf.
- Out-of-place graph operations initially. In-place updates require alias and
  lifetime rules, especially for training.
- Stable numerical formulas, operation-specific tolerances, and no promise of
  bit-identical CPU/GPU reductions.

## P0 — Minimum useful dense inference library

This is the first implementation target: linear regression, logistic regression,
and small fully connected networks with externally supplied weights.

### Elementwise arithmetic

- [ ] `add(a, b)` — addition, including scalar broadcasting.
- [ ] `sub(a, b)` — subtraction.
- [ ] `mul(a, b)` — elementwise multiplication, not matrix multiplication.
- [ ] `div(a, b)` — elementwise division.
- [ ] `neg(x)` — negation.
- [ ] `axpy(alpha, x, y)` — `alpha*x + y`; adapt the existing implementation.
- [ ] `bias_add(x, b)` — add `[features]` bias to `[batch, features]`.

### Linear algebra

- [ ] `dot(x, y)` — vector dot product.
- [ ] `matvec(A, x)` — matrix–vector multiplication; may initially use GEMM.
- [ ] `matmul(A, B)` / GEMM — `[M,K] × [K,N] → [M,N]`.
- [ ] Optional GEMM parameters: `alpha`, `beta`, and transpose flags, introduced
  after basic multiplication works.

GEMM is the most important performance kernel for dense ML. Start with a simple
CPU reference, then blocked/parallel CPU and tiled GPU implementations. Test
rectangular matrices and dimensions not divisible by tile sizes.

### Reductions and activations

- [ ] `sum(x, axis, keepdim)` — summation.
- [ ] `mean(x, axis, keepdim)` — average.
- [ ] `max(x, axis, keepdim)` — maximum values.
- [ ] `relu(x)` — `max(x, 0)`.
- [ ] `sigmoid(x)` — logistic function, using a stable implementation.
- [ ] `softmax(x, axis)` — normalized exponentials; subtract the axis maximum
  before exponentiation. Implement last-axis softmax first.

### Tensor plumbing

These are necessary API/runtime operations, even when they do not launch kernels.

- [ ] `constant`, `zeros`, `ones`, `full` — construct tensors.
- [ ] Host/device upload, download, and device-to-device copy where applicable.
- [ ] `reshape` and `flatten` — metadata-only when layout permits.
- [ ] `transpose` / `permute` — initially materialize contiguous output; views can
  follow once strides are supported.
- [ ] `argmax(x, axis)` — integer predictions for classification; specify tie rules.

**Acceptance demo:** run `XW + b`, then sigmoid for binary classification or
softmax for multiclass classification. Run a two-layer ReLU network and compare
with a reference. This supports inference, not training by itself.

## P1 — Broader numerical and classical ML support

Implement these after P0 works end to end.

### Elementwise math and comparisons

- [ ] `exp`, `log`, `sqrt`, `rsqrt`, `abs`, and `square`.
- [ ] `minimum`, `maximum`, and `clamp`.
- [ ] `equal`, `less`, `greater`, and related comparisons producing boolean masks.
- [ ] `where(mask, a, b)` — conditional element selection.
- [ ] `tanh` — activation used by older networks and recurrent models.
- [ ] General broadcasting for elementwise operations.

### Additional reductions and linear algebra

- [ ] `min`, `argmin`, and `sum_of_squares`.
- [ ] L1/L2 norms, with explicit axes.
- [ ] Variance and standard deviation; specify population/sample convention and
  use a numerically stable algorithm where practical.
- [ ] Batched `matmul`, initially for matching batch dimensions.
- [ ] Pairwise squared Euclidean distances, initially composed from reductions
  and GEMM; clamp small negative roundoff artifacts to zero.

### Indexing and shape operations

- [ ] `slice` with a restricted, documented indexing contract.
- [ ] `concat` and `split` along a selected axis.
- [ ] `gather` / row selection with integer indices and bounds validation.
- [ ] `broadcast_to`, initially materialized if zero-stride views are unsupported.

### Losses and evaluation

- [ ] Mean squared error (MSE).
- [ ] Binary cross-entropy **with logits**, using a stable formula.
- [ ] `log_softmax`, using stable log-sum-exp.
- [ ] Multiclass cross-entropy **with logits**, with integer class labels.
- [ ] Accuracy, composed from `argmax`, comparisons, and reductions.

Loss APIs should expose `none`, `sum`, and `mean` reduction modes. The initial
version can choose one mode if it is explicit and tested.

**Acceptance demos:** evaluate regression/classification losses; standardize input
features; assign samples to nearest centroids for a k-means iteration. Centroid
updates additionally need grouped accumulation, which can follow in P2.

## P2 — Training small dense models

Training needs more than forward kernels: gradients, parameter state, graph
execution rules, and an update loop. Start with manually constructed backward
graphs if a full automatic differentiation system would exceed scope.

### Backward operations

- [ ] Elementwise gradients for arithmetic, ReLU, sigmoid, and tanh.
- [ ] GEMM gradients using transposed GEMMs.
- [ ] Bias gradients and reduction gradients using sum/broadcast.
- [ ] MSE, binary cross-entropy-with-logits, and cross-entropy-with-logits gradients.
- [ ] Softmax/log-softmax backward when required independently of fused losses.
- [ ] `sum_to_shape` — reduce broadcasted gradients back to an input's shape.
- [ ] Gradient accumulation for parameters used by multiple graph branches.
- [ ] Scatter-add or equivalent accumulation for gather/indexing gradients;
  repeated indices must accumulate correctly.

A mathematical backward formula may use existing primitives rather than a new
kernel. For example, GEMM backward can dispatch two existing GEMM operations.

### Parameter initialization and optimizer operations

- [ ] Seeded uniform and normal random initialization.
- [ ] Xavier/He initialization built from random generation and scaling.
- [ ] SGD parameter update.
- [ ] Optional momentum SGD, followed by Adam with optimizer state and bias
  correction.
- [ ] Gradient zeroing and optional global-norm clipping.
- [ ] Optional dropout forward/backward with saved masks and explicit train/eval
  behavior.

### Useful accumulation primitives

- [ ] Grouped sums/counts or scatter-add for centroid updates and embeddings.
- [ ] Define determinism guarantees when GPU atomic accumulation is used.

**Runtime requirements:** save or recompute values needed by backward; distinguish
parameters from ordinary inputs; manage gradient lifetimes; sequence stateful
updates after gradient computation. Random state and parameter updates must not be
removed or reordered as if they were pure arithmetic.

**Acceptance demo:** train a small linear/logistic regression model, then an MLP.
Check gradients against finite differences on small inputs and verify decreasing
loss on a controlled dataset.

## P3 — Convolutional networks and normalization

### Convolution and pooling

- [ ] `conv2d` forward, with one documented layout, e.g. NCHW.
- [ ] Stride and padding, then dilation and groups if needed.
- [ ] Simple direct convolution reference implementation.
- [ ] Optional im2col + GEMM implementation; account for temporary memory costs.
- [ ] Max pooling and average pooling.
- [ ] Convolution and pooling backward if training is in scope.

### Normalization

- [ ] Layer normalization forward, including epsilon and learned scale/bias.
- [ ] Batch normalization inference using stored running statistics.
- [ ] Batch normalization training and running-statistic updates, if needed.
- [ ] Normalization backward for whichever training operations are supported.

**Acceptance demo:** a small CNN inference graph. Training it requires the matching
backward operations; forward-only support is not sufficient.

## P4 — Transformer inference and advanced extensions

This is a stretch tier. Reuse earlier primitives before writing specialized fused
kernels.

- [ ] Embedding lookup, built on gather.
- [ ] Batched GEMM with transpose support for attention scores and weighted sums.
- [ ] Scaled dot-product attention: `softmax(QK^T / sqrt(d) + mask) V`.
- [ ] Causal and padding masks, with defined all-masked-row behavior.
- [ ] GELU activation; specify exact or approximate formula.
- [ ] LayerNorm or RMSNorm as required by the chosen model.
- [ ] Positional encodings or rotary positional embeddings, model-dependent.
- [ ] Optional KV cache for incremental decoding, with explicit state/lifetimes.
- [ ] Optional top-k selection for prediction or sampling.

Advanced performance work, only after a correct baseline:

- Fused attention to reduce intermediate memory traffic.
- Float16/BFloat16 storage and mixed-precision accumulation on supported hardware.
- Quantized matrix multiplication with documented scale/zero-point semantics.
- Sparse operations, only if a target workload actually needs them.

**Acceptance demo:** a small transformer block with fixed weights and reference
outputs. A full language-model serving system is outside the initial scope.

## Which kernels unlock which algorithms?

| Workload | Required operations |
|---|---|
| Linear regression inference | GEMM/matvec, bias addition |
| Logistic regression inference | GEMM/matvec, bias addition, sigmoid |
| Multiclass linear classifier | GEMM, bias addition, softmax, argmax |
| Dense neural network inference | GEMM, bias addition, ReLU/sigmoid/tanh |
| Regression/classification training | Forward operations, losses, gradients, SGD |
| K-means | Distances, argmin, grouped sums/counts, division |
| CNN inference | Convolution, activation, pooling, dense layers |
| Transformer block inference | Embeddings, batched GEMM, masking, softmax, normalization, activation |

Other ML algorithms can require additional numerical methods. For example, robust
least-squares solvers may need QR/SVD, and PCA needs eigendecomposition or SVD.
Those are substantial extensions, not implied by basic BLAS support.

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
For example, `matmul → bias_add → relu` can remain three graph nodes while an
optimized plan dispatches one supported fused implementation.

**Recommended course-project target:** P0, the P1 operations needed for losses and
benchmarks, and a well-tested CPU/GPU execution path with autotuning. Add P2 only
if training is an agreed goal; P3 and P4 are optional expansions.
