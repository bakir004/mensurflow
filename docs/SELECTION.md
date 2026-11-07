# Comparisons and where

The CPU F32 operations `eq`, `ne`, `lt`, `le`, `gt` and `ge` compare two tensors and return a fresh
contiguous Bool tensor. `where(condition, x, y)` selects an F32 value from `x` for a nonzero Bool
condition and from `y` for zero. All inputs must share a device; comparison operands and the two
`where` value operands must have matching dtypes. Shapes broadcast using the existing Tensor rules,
including scalars, empty dimensions and expanded or strided views.

```cpp
Tensor x = Tensor::from_vector<float>({-2, 0, 3}, {3});
Tensor zero = Tensor::full({}, 0.0);
Tensor positive = gt(x, zero);            // Bool: {0, 0, 1}
Tensor selected = where(positive, x, zero); // F32: {0, 0, 3}

// Bool is already a one-byte dtype. No vector<bool> or additional dtype conversion is needed.
std::vector<std::uint8_t> mask(static_cast<std::size_t>(positive.numel()));
positive.copy_to_host(mask.data());
```

An externally supplied mask can use `Tensor::from_host(bytes.data(), shape, DType::Bool)`.
`Tensor::from_vector<std::uint8_t>` creates a U8 tensor, which is a different dtype and is rejected
as a condition. Comparisons produce canonical bytes 0/1; `where` accepts any nonzero Bool byte as
true. Bool conversion APIs and additional value dtypes are left to the later cast task.

NaNs follow ordinary F32 comparison rules: `ne` is true when either operand is NaN; the other
comparisons are false. Positive and negative zero compare equal. `where` selects a value without
arithmetic masking, preserving the selected signed zero or quiet NaN payload.

These operations record no autograd graph. Comparisons are nondifferentiable; `where` is currently
forward-only, including when a value input has `requires_grad` set. Its output does not track
gradients. Backward support remains outside this forward implementation.

## Architecture and support

The operations follow the existing path: public API → shape/device validation and contiguous
packing → dispatcher → CPU adapter → plain C++ loop with `MENSURFLOW_PARALLEL_FOR`. The six CPU
comparison wrappers share a templated loop. `where` uses the same OpenMP cutoff of 32,768 elements.
There are no hand-written intrinsics, new thread helpers or per-operation cutoff changes.

The dispatcher still keys on the first input's dtype. Comparisons use F32 inputs and a Bool output.
Internally `where` supplies `{x, y, condition}` to `OpArgs`, so its key remains F32; its adapter reads
the mask from input 2. The public argument order remains `where(condition, x, y)`. No dispatcher
policy or Bool-specific registry mechanism is introduced.

Currently only CPU F32 value kernels are registered. Unsupported value dtypes or explicit GPU
execution throw `NotImplemented`, including for empty results; unavailable GPUs use the existing
runtime error. Undefined tensors, dtype/device mismatches, non-Bool conditions and incompatible
broadcast shapes throw `invalid_argument`. Explicit CPU and current Auto execution retain the
existing cross-device transfer behavior and return results on the input device.

## Build and verify

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DMENSURFLOW_CUDA=OFF \
  -DMENSURFLOW_METAL=OFF -DMENSURFLOW_NATIVE=OFF -DMENSURFLOW_OPENMP=ON
cmake --build build-cpu -j4
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ctest --test-dir build-cpu --output-on-failure
```

`selection_test` uses independent logical-index broadcasting and byte/bit references. It checks
IEEE truth tables, API metadata, scalars/empty shapes, views and storage offsets, three-way
broadcasting, registration/errors, thread invariance and in-place low-level `where` calls. The
existing arithmetic and activation tests continue to check regression of the old operations.

## Benchmark

`selection_benchmark` measures direct CPU calls into preallocated buffers using the shared harness.
Its normal build reports one-thread/many-thread medians; the existing optional
`MENSURFLOW_TORCH_BENCHMARK=ON` also enables native C++ ATen `out` references for this executable.
LibTorch remains optional and is not linked as a production backend. Use the separate build setup
from [ACTIVATIONS.md](ACTIVATIONS.md#native-c-reference-preferred-for-timing).

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/selection_benchmark \
  --sizes=1000,32769,4000003 --threads=1,2,4

OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-torch-bench/selection_benchmark --sizes=1000,32769,4000003 --threads=1,2,6,12 \
  > build-torch-bench/results/selection-run1.txt
# Cool down before repeating, and add --reverse for the second native run.
```

Inputs are deterministic finite values from the shared harness. The mask repeats two true and
three false bytes. Before timing, each implementation is checked against an independent scalar
reference, with output sentinels preventing stale buffers from passing. Native calls use the same
addresses, thread counts and observer on both sides; Python, wrapping, allocation and checksums are
excluded, while ATen operator dispatch is included. This compares direct kernels with ATen operators,
rather than both libraries' full Tensor API latency. Each timed library follows a 500 ms pause.

Comparison rates count 9 logical bytes per element (two F32 reads and a Bool write). The `where`
rate counts an upper bound of 13 bytes (mask, both value branches and output); the implementation
may load only the selected branch. Neither is a physical DRAM bandwidth measurement. The largest
listed case exceeds this machine's L3 cache. Repeat cooled sessions and preserve both medians and
drift; the copy-bandwidth ceiling remains dependent on Vedad's separate benchmark.

The existing standard-library-only `benchmarks/reference/compare_native_activations.py` parser can
compare the native selection logs too, since both use the same paired table format.
