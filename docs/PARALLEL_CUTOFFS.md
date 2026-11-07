# CPU elementwise parallel cutoffs

Each CPU elementwise operation has an independent cutoff in
`src/backends/cpu/ops/elementwise/parallel_cutoffs.hpp`. The OpenMP loop uses multiple requested
threads when `n > cutoff`; at or below the cutoff it uses the serial branch. Actual worker usage
also depends on OpenMP settings. Without OpenMP the same loops always execute serially.

All initial defaults are **32,768 elements**, preserving the previous scheduling behavior. They
are independent entries, ready for measurement; they are not measured optimal values. GELU exact
and tanh approximation have separate entries because they execute different arithmetic.

The Tensor API, dispatcher keys, CPU adapter signatures and kernel arithmetic stay the same.
The shared activation and comparison templates select their cutoff by an internal `ElementwiseOp`.
Other elementwise loops select their own entry directly. BLAS and reduction kernels retain their
existing scheduling constants in `threads.hpp`.

## Measurement controls

The internal functions `elementwise_parallel_cutoff(op)` and
`set_elementwise_parallel_cutoff(op, cutoff)` access one process-local entry. Overrides are atomic
and each OpenMP region snapshots its cutoff once. Configure before timing so concurrent changes
cannot mix policies during a measurement. Negative cutoffs and invalid IDs throw `invalid_argument`.
Overrides last until changed or the process exits; there is no persistent cache or automatic tuning.

The existing `cpu_parallel_benchmark`, `selection_benchmark`, `cast_benchmark` and optional
`activation_torch_benchmark` accept `--cutoff=n`, applying it to selected MensurFlow elementwise
operations. Headers print the effective values. `dot` in the arithmetic benchmark retains its
existing cutoff and is unaffected by this option.

| Option | Effect on selected MensurFlow kernels |
|---|---|
| omitted | Use the operation's default cutoff |
| `--cutoff=0` | Enable the parallel branch for every nonempty input |
| `--cutoff=9223372036854775807` | Keep every input on the serial branch |
| another nonnegative int64 | Use that candidate cutoff |

All four benchmarks use nine samples, preallocated buffers and a 500 ms worker-settling pause
before each measurement. Selection and activation native references use unchanged ATen scheduling.
Their headers read `at::internal::GRAIN_SIZE` from the installed `ATen/TensorIterator.h` instead of
hard-coding a PyTorch value. In the locally installed CPU Torch 2.15.0 headers it is 32,768 elements.
TensorIterator's grain controls work splitting as well as small-input decisions; matching that
number does not imply identical worker scheduling or an optimal MensurFlow cutoff.

## Final measurement procedure

Compilation and correctness results are recorded in the daily worklog. Per-operation crossover
measurements remain pending. These commands describe the tuning procedure; they have not been run.
Use a portable Release/OpenMP build, AC power, a quiet machine and repeated cooled sessions.

First compare forced serial and forced parallel execution at the **same requested thread count**.
For example, for `where` (repeat the process for each operation):

```bash
OMP_NUM_THREADS=6 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-cpu/selection_benchmark where --threads=6 \
  --sizes=1000,4000,8000,16000,32768,32769,64000,128000,1000000,4000003 \
  --cutoff=9223372036854775807 > build-cpu/results/where-serial-run1.txt
# Cool down before the next measurement.
OMP_NUM_THREADS=6 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-cpu/selection_benchmark where --threads=6 \
  --sizes=1000,4000,8000,16000,32768,32769,64000,128000,1000000,4000003 \
  --cutoff=0 > build-cpu/results/where-parallel-run1.txt
```

Create the results folder first. Repeat both after cooling in reversed order with different output
filenames, preserving both logs. Compare the rows with the same size and thread count; the normal
selection benchmark also prints its one-thread baseline. Native builds instead print paired ATen
rows. `cpu_parallel_benchmark` covers arithmetic/activation operations; `selection_benchmark` covers
the six comparisons and `where`; `cast_benchmark` covers the dtype conversion pairs.

Find a size interval where the parallel median repeatedly beats the serial median, refine the size
sweep around it, and repeat at intermediate, physical-core and hardware-thread counts. If timings
drift by more than 10%, retain the noisy result and repeat before choosing a default. A single
per-operation cutoff is a compromise across thread counts and machines, so document that tradeoff.

Update only the relevant table entry, verify just below/at/above the chosen cutoff, and rerun the
native activation/selection comparisons with the default policy in both library orders. Record
median, minimum, ratios, drift, machine, thread placement and the ATen grain in the result report.
Keep previous reports attached to the scheduling policy they actually measured.

`parallel_cutoff_test` is registered for the final suite. It checks operation isolation, invalid
configuration and representative arithmetic/activation/comparison/selection results with cutoffs
0, 17 and INT64_MAX, including empty inputs and sizes around 17. Run the suite with and without
OpenMP; then run the existing operation tests around the final measured defaults too.
