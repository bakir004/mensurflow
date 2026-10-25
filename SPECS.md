# PRS: Heterogeneous ML Compute Library and Task Graph Runtime

## 1. Goal

Build a small C++17, inference-oriented compute system inspired by PyTorch and
TensorFlow: define a graph of tensor operations, optimize it, then execute it on
CPU and/or GPU. The runtime should eventually choose devices using measured
performance, data location, and current device load—not assume that GPU is always
faster.

This combines two course topics:

- **P27 — CPU/GPU BLAS and ML Kernel Library with Autotuning:** implement correct,
  fast numerical operations and select good kernel implementations/configurations.
- **P26 — Heterogeneous CPU–GPU Task Graph Runtime:** represent dependencies,
  optimize graphs, schedule work, and manage data across devices.

These are separate layers with separate tests and benchmarks. P27 must be usable
without the graph runtime; P26 calls P27 rather than embedding numerical kernels.

**Scope:** a learning/research prototype, not a replacement for PyTorch. Automatic
differentiation, training frameworks, distributed execution, and full BLAS/API
compatibility are not initial requirements.

## 2. Target user experience

Illustrative target API, to be implemented in stages:

```cpp
prs::Model model;

// Declare named, float32 scalar inputs. Tensor shapes work the same way.
auto i1 = model.input("i1", prs::Shape{});
auto i2 = model.input("i2", prs::Shape{});
auto i3 = model.input("i3", prs::Shape{});

auto x1 = model.addNode("plus", i1, i2);
auto x2 = model.addNode("mult", x1, i3);
auto o1 = model.addNode("neg", x2);

model.optimize(o1);  // Validate and compile the subgraph needed for this output.
auto result = model.run({prs::Tensor::scalar(1.0f),
                         prs::Tensor::scalar(2.0f),
                         prs::Tensor::scalar(3.0f)});
// result[0].item() == -9.0f
```

API rules:

- `run` binds inputs in declaration order; input count, dtype, and shape must match.
- `optimize` selects outputs, validates the graph, and creates a reusable plan.
  It need not perform sophisticated rewrites in the first version.
- Support multiple selected outputs later. Return outputs in requested order.
- `plus` and `mult` are elementwise operations; matrix multiplication is `matmul`.
- Nodes belong to one model; reject foreign handles, unknown operations, invalid
  shapes, and cycles before execution.
- Initially shapes are fixed at compilation. Dynamic shapes are optional later.
- Graph mutation invalidates the compiled plan; require another `optimize` call.
- First version: synchronous `run`, outputs ready on CPU when it returns.
- Provide forced CPU, forced available GPU, and automatic execution modes.
  Forced unsupported execution must fail clearly rather than silently fall back.

## 3. Current starting point

The repository already contains:

- Public float32 AXPY (`y = alpha*x + y`) on CPU, CUDA, and Metal.
- Optional OpenMP on CPU, correctness tests, and an AXPY benchmark.
- A standalone CPU/Metal matrix multiplication benchmark, not a public kernel API.
- CMake backend detection and CPU-only builds.

It does **not** yet contain a tensor abstraction, graph engine, persistent GPU
storage, adaptive scheduling, or autotuning. Existing GPU AXPY calls allocate and
copy on each invocation. Preserve the existing API/tests while adding new layers.

Use Metal on the available Apple Silicon machine, or CUDA on an NVIDIA machine.
**One working GPU backend is enough for the first integrated version.** Keep a
backend boundary so the other can follow. Do not require both GPUs in one system.

## 4. Implementation order and stopping points

Implement stages in this order. Each stage adds work to both topics, with a
working demo at the end. Later stages are optional extensions, not prerequisites
for a useful first submission; confirm assessment requirements with the instructor.

| Stage | P27: kernel library | P26: graph runtime | Deliverable |
|---|---|---|---|
| 0 | Preserve and measure existing kernels | Define interfaces | Reproducible baseline |
| 1 | CPU tensor operations | Sequential CPU DAG execution | User's scalar graph works |
| 2 | CPU GEMM and basic ML ops | Validation and safe graph optimizations | Small CPU inference graph |
| 3 | Persistent tensors on one GPU | CPU/GPU dispatch and transfer planning | Same graph on CPU or GPU |
| 4 | Measured kernel autotuning | Profile-based automatic device placement | **Recommended combined project target** |
| 5 | Concurrent-safe kernel execution | Parallel DAG scheduling and load awareness | Advanced heterogeneous runtime |
| 6 | Fusion and advanced kernels | Joint graph/device optimization | Stretch research features |

### Stage 0 — Baseline and shared contracts

**P27**

- Build/test existing AXPY in Release mode, including CPU-only configuration.
- Record machine, compiler, backend, thread count, input sizes, and timings.
- Preserve the standalone matrix multiplication benchmark as a reference.

**P26 / shared design**

- Define backend capability queries, error handling, operation identifiers, and
  the initial tensor/shape interfaces.
- Choose the first GPU backend based on actual test hardware.
- Separate reference computation, optimized execution, and benchmarking.

**Done when:** clean builds and existing tests pass; unavailable backend tests
are reported as skipped, not presented as GPU correctness evidence.

### Stage 1 — Smallest working graph, CPU only

**P27**

- Implement a float32, contiguous, row-major `Tensor` with owned CPU storage.
- Support scalars (rank zero), vectors, and matrices; check dimension products and
  allocation sizes for overflow. Define zero-element tensor behavior explicitly.
- Implement serial reference kernels: elementwise add, multiply, and negate.
- Initially require equal shapes, except scalar-to-tensor broadcasting.
- Graph operations are out-of-place; do not introduce arbitrary views or aliasing.

**P26**

- Implement input nodes, operation nodes, node handles, and selected output nodes.
- Build an operation registry: name, input count, shape inference, implementation.
- Find output-reachable nodes, validate them, and produce a topological order.
- Execute sequentially; store intermediate tensors until the run finishes.
- Reuse the compiled graph across different values with the same input shapes.

**Done when:** the target example returns `-9`; vector equivalents work; malformed
inputs fail clearly. Test branching, shared intermediates, and repeated runs.

**Stop here if necessary:** working graph proof of concept, but not yet a
heterogeneous runtime or autotuned library.

### Stage 2 — Useful CPU BLAS/ML subset and graph optimization

**P27**

Implement in increasing difficulty:

1. AXPY through the tensor API, preserving the existing public function.
2. Dot product and sum reduction (initially full-tensor reduction).
3. Dense matrix multiplication: `C[M,N] = A[M,K] * B[K,N]`.
4. ReLU and explicit row-bias addition for `[batch, features]` tensors.
5. Optional numerically stable row-wise softmax, subtracting the row maximum.

For performance, add OpenMP for suitable sizes, then blocked CPU GEMM. Keep serial
reference kernels as correctness oracles. Specify reduction precision/tolerances;
parallel reduction order need not produce bit-identical floats.

**P26**

- Add constants and shape inference for the supported operations.
- Eliminate unreachable nodes; evaluate constant-only subgraphs once.
- Track consumer counts and release intermediates after their last use, retaining
  selected outputs. Buffer pooling can wait until later.
- Expose a plan dump containing operations, shapes, and chosen execution order.

**Done when:** a small forward-only network runs:

```text
X -> matmul(W1) -> bias_add(b1) -> relu -> matmul(W2) -> bias_add(b2)
```

Compare every output with a direct reference implementation. No training needed.

### Stage 3 — Real CPU/GPU execution and memory residency

**P27**

- Add a backend storage abstraction and explicit host/device synchronization.
- Implement GPU add, multiply, negate, ReLU, AXPY, and GEMM on the chosen backend.
  Add reductions/bias operations as needed by the demo.
- Promote/adapt the existing Metal matmul example into the library if using Metal.
- Retain buffers between dependent GPU operations; avoid per-node host round trips.
- Establish a simple buffer pool and clear ownership/lifetime rules.
- Track the authoritative tensor location and validity of cached copies.
  Unified memory does not remove synchronization requirements.

**P26**

- Add force-CPU and force-GPU plans, with backend capability validation.
- Add a simple automatic fallback policy: unsupported GPU operations use CPU;
  insert transfers only when a consumer needs data on another device.
- Initially use sequential submission and conservative synchronization.
- Copy final results to CPU at the API boundary; keep GPU intermediates resident.
- Record transfers, allocations, and device execution in a trace.

**Done when:** identical graphs pass CPU/GPU correctness checks; a GPU operation
chain avoids intermediate readbacks; a mixed supported/unsupported graph transfers
correctly; unavailable devices produce useful errors.

**Stop here:** a functional heterogeneous prototype, but no real autotuning yet.

### Stage 4 — Autotuning and measured automatic placement

This is the recommended target demonstrating substantial work in **both** topics.

**P27: tune implementations**

- Maintain candidate configurations, starting with a small, bounded search space:
  - CPU serial vs. parallel kernels; optional thread counts and GEMM tile sizes.
  - GPU workgroup/block sizes and supported GEMM tile variants.
- Reject configurations exceeding device/resource limits before benchmarking.
- Tune explicitly or on first use, using warmups and repeated measurements.
  Validate candidate outputs against the reference outside timed regions.
- Cache the best configuration by operation, shape/layout, dtype, device identity,
  and implementation version. Invalidate stale entries.
- Offer tuning disabled, deterministic fixed configuration, and cache reset modes.
- Bound tuning time and exclude it from steady-state performance claims; report
  cold-start/tuning costs separately.

**P26: select devices**

- Compare estimated completion costs, not just isolated kernel speed:

```text
candidate_finish = max(dependencies_ready + required_transfer_time,
                       device_available_time)
                   + launch_overhead + measured_compute_time
```

- Account for current tensor residency, final output readback, and CPU thread use.
- Start with measured shape-specific profiles and greedy topological placement.
  Do not claim this finds a globally optimal graph assignment.
- Unknown shapes use bounded profiling or a documented conservative fallback.
- Provide explainable decisions: selected backend and estimated cost components.
- Keep placement separate from kernel tuning: tuning chooses *how* an operation
  executes on a device; scheduling chooses *where and when* it executes.

**Done when:** tests show small operations favor CPU when measured overhead makes
GPU slower; a sufficiently compute-heavy graph can favor GPU on suitable hardware;
repeated executions reuse tuning results. Compare against CPU-only, GPU-only, and
an untuned automatic baseline, including cases where automatic placement loses.

### Stage 5 — Parallel graph runtime and “who is faster right now”

**P27**

- Make kernel dispatch, caches, and buffer management safe for concurrent use.
- Provide asynchronous submissions/completion events for the chosen GPU backend.
- Coordinate CPU kernel thread counts with graph workers to avoid oversubscription.

**P26**

- Use dependency counters and ready queues instead of one serial execution loop.
- Add a bounded CPU worker pool and GPU submission/completion handling.
- Run independent graph branches concurrently where resources allow.
- Express transfers as dependencies; do not reuse buffers until all relevant work
  completes. Handle errors without leaving dependents waiting forever.
- Update execution-time estimates with recent measurements and queued work.
- Add hysteresis/minimum sample counts so noise does not cause device oscillation.
- Respect memory limits; do not dispatch work that cannot allocate its outputs.

“Faster right now” initially means **our observed task timings and our queues**.
External OS/GPU contention may affect observations, but predicting all system load
is not guaranteed. CPU and GPU may also contend for shared memory bandwidth.
Placement is decided before dispatch; preemption or migration of a running kernel
is not required.

**Done when:** a fork/join graph demonstrably overlaps independent work; outputs
remain correct under repeated stress runs; controlled CPU/GPU contention changes
placement when beneficial. Include a trace proving overlap, not only a speedup.

### Stage 6 — Stretch goals, ordered roughly by difficulty

1. Additional reductions, transpose support, and broader broadcasting.
2. More optimized GEMM variants and comparison with vendor libraries.
3. Pattern fusion: `matmul + bias + ReLU`, or elementwise chains.
4. Critical-path-aware scheduling and transfer-aware placement lookahead.
5. Multiple execution plans specialized by input shape or residency.
6. More advanced buffer reuse, asynchronous transfer overlap, and batching.
7. Second GPU backend with equivalent conformance tests.
8. Optional convolution, mixed precision, or automatic differentiation—each a
   substantial separate extension, not part of the core acceptance criteria.

Rewrites must preserve documented numerical semantics. In particular, do not
assume arbitrary floating-point reassociation is exact.

## 5. Suggested architecture

```text
User Model API
    -> Graph IR and shape validation
    -> Graph rewrites and execution plan                     [P26]
    -> Scheduler + profiling + transfer/memory management     [P26]
    -> Kernel registry + autotuner                           [P27]
    -> CPU / Metal / CUDA kernels                            [P27]
```

Suggested components (names can change):

- `Tensor`, `Shape`, `Storage`: metadata, checked sizes, ownership, residency.
- `OpRegistry`, `KernelRegistry`: semantics and backend implementations.
- `Graph`, `Node`, `Model`: graph construction and public interface.
- `ExecutionPlan`: selected outputs, dependencies, placements, buffer lifetimes.
- `Executor`, `Scheduler`: sequential first, asynchronous later.
- `Device`, `Buffer`, `Event`: backend-specific resource abstractions.
- `Profiler`, `Autotuner`, `TuningCache`: measurements and reusable configurations.

Keep public headers under `include/prs/`; organize implementations under `src/`
into core, graph, runtime, and backend/kernel areas as the project grows. Keep
correctness tests under `tests/`, performance programs under `benchmarks/`, and
add small runnable graph demos under `examples/`.

## 6. Correctness and testing requirements

- Compare all optimized kernels with simple reference implementations.
- Use absolute + relative float tolerances, chosen per operation and reduction
  size; explicitly test NaN/Inf behavior if supported. Do not require bit equality
  between different reduction orders.
- Cover scalars, empty inputs where allowed, rectangular GEMM, non-tile-aligned
  dimensions, zero/negative values, and invalid dimensions/shapes.
- Graph tests: chains, diamonds, disconnected nodes, multiple outputs, constants,
  shared inputs, invalid handles, missing inputs, and repeated execution.
- Optimization tests compare optimized and unoptimized graph outputs.
- Placement tests use fake deterministic costs rather than requiring a specific
  physical CPU/GPU speed ranking.
- Memory tests exercise repeated runs, pooling, transfers, and early release.
- Run CPU tests everywhere; GPU tests require real hardware and explicit skip
  reporting. Test forced-backend failures separately.
- Add sanitizer-enabled CPU builds where practical; GPU correctness needs its own
  backend checks and synchronization/lifetime tests.

## 7. Benchmark and demonstration plan

Measure **end-to-end graph latency** separately from kernel-only timings. GPU
wall-clock timing must wait for completion; use device timestamps for kernel-only
measurements where supported. Never compare asynchronous launch time against CPU
completion time.

Workloads, in increasing complexity:

1. Scalar example: `(i1 + i2) * i3`, then negate.
2. AXPY/elementwise size sweep: demonstrate GPU launch/transfer overhead.
3. GEMM size sweep, including rectangular cases: demonstrate compute-heavy work.
4. Small two-layer network: demonstrate persistent GPU intermediates.
5. Fork/join graph: demonstrate CPU/GPU scheduling and, in Stage 5, concurrency.
6. Controlled load experiment: demonstrate adaptive decisions, if implemented.

For each relevant workload report:

- CPU serial, CPU parallel, GPU-only, and automatic modes where implemented.
- First-use time separately from warmed steady-state time.
- Median and variability across repeated trials; warmups and trial count.
- Input sizes, hardware/software details, CPU thread counts, and tuning settings.
- Allocations, transfer bytes/time, kernel time, scheduling time, total latency.
- Correctness result and numerical tolerance.
- Speedup `baseline_time / candidate_time`, including slowdowns.

Compare custom GEMM against a vendor baseline (Accelerate/OpenBLAS/MKL or
cuBLAS/MPS, as available) when possible. Vendor libraries may be optional backend
candidates, but distinguish them from kernels we implemented ourselves. No fixed
speedup is promised: performance claims must follow actual measurements.

## 8. Work split and final checklist

Suggested ownership:

- **P27 lead:** tensor-level kernel interface, CPU/GPU implementations, kernel
  correctness, tuning search/cache, and kernel benchmarks.
- **P26 lead:** graph API/IR, validation, graph optimization, execution plans,
  scheduling, and graph benchmarks.
- **Shared:** tensor/device/event contracts, transfer ownership, integration tests,
  profiling format, and final demo. Agree on these interfaces before splitting.

Recommended final deliverables through Stage 4:

- [ ] C++ graph-building API with the scalar example and an ML inference example.
- [ ] Tested CPU kernels and one tested GPU backend.
- [ ] Persistent device buffers and explicit transfer/residency handling.
- [ ] Output-reachable graph compilation and basic safe optimizations.
- [ ] Measured kernel autotuning with reusable configuration cache.
- [ ] Automatic placement using compute and transfer costs.
- [ ] Forced CPU/GPU modes and an explainable execution-plan report.
- [ ] Reproducible benchmarks and honest comparisons, including failures to win.
- [ ] Documentation clearly distinguishing completed features from stretch goals.

If time is short, finish and validate the current stage rather than partially
implementing every advanced feature. Stage 4 demonstrates measured heterogeneous
placement; Stage 5 is what adds genuine concurrent, load-aware task scheduling.
