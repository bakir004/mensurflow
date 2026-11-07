# CPU activations

All five activations use the existing Tensor API, dispatcher and CPU F32 backend. Their low-level
kernels share one templated loop, `activation_f32`, including the existing OpenMP cutoff of 32,768
elements and support for exact input/output aliasing. Public calls allocate a fresh output and accept
strided or expanded views. Autograd uses the same CPU F32 restrictions as the other unary operations.

```cpp
leaky_relu(x);                         // negative slope 0.01
leaky_relu(x, 0.2f, Execution::CPU);
gelu(x);                               // Gaussian CDF / erf form
gelu(x, true, Execution::CPU);          // tanh approximation
silu(x);
elu(x, 1.5f);                          // alpha
softplus(x, 2.0f, 15.0f);              // beta, linear threshold
```

GELU implements both forms documented by [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html).
SiLU is `x * sigmoid(x)`; ELU's negative branch uses `alpha * expm1(x)`. Softplus implements
`log(1+exp(beta*x))/beta` and the [documented linear shortcut](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html)
when `beta*x > threshold`. It rejects nonpositive/nonfinite beta and nonfinite thresholds.

The transcendental implementations retain intermediate values in double where cancellation or
premature underflow would destroy negative tails. Softplus uses `log1p` with a nonpositive exponential
argument. GELU and SiLU explicitly return their limiting value `-0` at negative infinity and positive
infinity at positive infinity. These infinity conventions need not match a vendor implementation's
literal `infinity * 0` behavior. NaNs propagate. At zero, leaky ReLU and ELU backward use the supplied
slope/alpha; GELU and SiLU have derivative 0.5, while softplus uses its selected linear shortcut.

`activation_test` checks float64 references, custom parameters, tails/special values, views, scalars,
empty inputs, dtype errors, registration, one/many threads, in-place kernels, weighted finite-difference
gradients and a composed backward path. The existing test suite checks regression of the shared unary
helper. These scalar/libm implementations are not claimed to provide an explicit AVX2 transcendental
kernel, a tuned per-operation cutoff, or measured SIMD speedup.

## Existing autograd engine limits

The activation rules have been checked for simple chains. The core autograd engine currently has
two separate limitations, reproduced with the pre-existing `square` operation during review:

- Reusing an intermediate loses contributions to its parents: for scalar `x=2`, `z=square(x)` and
  `add(z,z).backward()`, `x.grad()` is 4 instead of 8. `propagate` consumes a node on its first visit
  before the other incoming gradients have been collected.
- A transposed view shares the original tensor's autograd state without an inverse view-backward
  mapping. For input shape `[2,3]`, `sum(square(x.transpose(0,1))).backward()` stores a gradient with
  shape `[3,2]` on the original tensor instead of `[2,3]`.

Forward execution on views remains covered. The engine issues require separate graph-accumulation
and view-gradient work; the activation tests do not establish support for arbitrary branched graphs
or differentiation through views.

## Build and verify

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DMENSURFLOW_CUDA=OFF \
  -DMENSURFLOW_METAL=OFF -DMENSURFLOW_NATIVE=OFF -DMENSURFLOW_OPENMP=ON
cmake --build build-cpu -j 4
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ctest --test-dir build-cpu --output-on-failure
```

## Benchmark and reference comparison

The existing CPU benchmark now accepts the five names plus `gelu_tanh`. It checks each new kernel
against a separate float64 reference before timing and still requires bitwise agreement between
one-thread and multi-thread results. It measures kernels into preallocated buffers, not allocation,
dispatch, packing views or backward. The optional size list enables a short verification run:

```bash
mkdir -p build-cpu/results
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark \
  leaky_relu gelu gelu_tanh silu elu softplus --sizes=1000,32769,100003 \
  > build-cpu/results/activations.txt
```

Omit `--sizes` for the full existing size sweep. Repeat official measurements in a separate quiet
session, on AC power, after cooling; preserve the machine/build header. `GB/s` counts one F32 read and
one write (8 bytes per element), not a measurement of physical DRAM traffic. One thread may use SIMD;
the one-thread/many-thread ratio does not isolate SIMD speedup. Below the cutoff, requested parallel
rows execute the serial branch.

### Native C++ reference (preferred for timing)

`activation_torch_benchmark` calls both implementations from C++ using the same input and output
addresses, observation/checksum, timer and requested intra-op thread count. Its preallocated ATen
`out` calls include ATen dispatch but exclude Python. MensurFlow is called at its direct CPU-kernel
boundary, so this is a direct-kernel vs C++ operator comparison, not whole Tensor-API latency.

The target is optional and OFF by default. A CPU PyTorch wheel supplies the LibTorch headers,
libraries and CMake configuration; NumPy is not required for this executable. Configure a separate
build directory so required ABI flags cannot mix with older build objects:

```bash
mensurflow_torch_prefix="$(build-cpu/torch-reference/bin/python -c \
  'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build-torch-bench -DCMAKE_BUILD_TYPE=Release \
  -DMENSURFLOW_CUDA=OFF -DMENSURFLOW_METAL=OFF -DMENSURFLOW_NATIVE=OFF \
  -DMENSURFLOW_OPENMP=ON -DMENSURFLOW_OPENBLAS=OFF \
  -DMENSURFLOW_TORCH_BENCHMARK=ON -DCMAKE_PREFIX_PATH="$mensurflow_torch_prefix"
cmake --build build-torch-bench -j 4
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ctest --test-dir build-torch-bench --output-on-failure
```

The installed Torch target supplies the C++ standard its headers require. With the tested 2.15 CPU
wheel, only the reference executable needs C++20; the core and harness remain C++17/ABI1. Any required
`TORCH_CXX_FLAGS` are applied consistently across targets that exchange C++ standard-library values.
Other builds retain their current C++17 requirements and have no Torch dependency.

For an official comparison, plug in, close heavy programs, let the laptop cool for about three
minutes after building/testing, then run both orders with another cooldown between them:

```bash
mkdir -p build-torch-bench/results
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-torch-bench/activation_torch_benchmark --sizes=1000,32769,1000000 \
  --threads=1,2,6,12 > build-torch-bench/results/activations-run1.txt
# Let the laptop cool again before the second command.
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-torch-bench/activation_torch_benchmark --sizes=1000,32769,1000000 \
  --threads=1,2,6,12 --reverse > build-torch-bench/results/activations-run2.txt
```

Each side is checked against float64 before timing, with nine samples and a 500 ms worker-settling
pause before each library. `vs torch` is ATen median / MensurFlow median, so above one favors
MensurFlow. Each library retains its own small-input parallelization policy. Build/parallel metadata
are included in the output. Use `--help`, operation names and smaller size/thread lists for a smoke
check; a smoke run is not a controlled performance result.

Compare both logs with the standard-library-only parser; it checks matching paired rows and flags
timing drift above 10% rather than hiding unstable measurements:

```bash
python3 benchmarks/reference/compare_native_activations.py \
  build-torch-bench/results/activations-run1.txt \
  build-torch-bench/results/activations-run2.txt \
  > build-torch-bench/results/activations-paired.md
```

The local build/test validation and native reference measurements are recorded in the
[2026-11-07 report](benchmarks/aldin-ryzen5-6600h-2026-11-07.md).
