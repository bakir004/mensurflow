# CPU benchmarks: sequential vs parallel

Results from the float32 CPU kernels (`add`, `sub`, `mul`, `div`, `neg`, `abs`,
`square`, `relu`, `dot`), measured on each machine we have so the numbers can be compared. Keep the build and run conditions below the same on every machine,
otherwise the comparison is not fair.

Code:
- parallel: `src/backends/cpu/parallel.hpp` (`parallel_for`, `parallel_sum`),
  `src/backends/cpu/elementwise.cpp`, `src/backends/cpu/blas.cpp`.
- sequential: `src/backends/cpu/sequential.cpp`, plain loops with no OpenMP. This is the baseline.

Benchmarks: `benchmarks/cpu_parallel_benchmark.cpp` (sequential vs parallel, kernels called
directly), `benchmarks/blas/dot_benchmark.cpp` (through the public `mensurflow::dot`).
Tests cover CPU kernels, sequential agreement, and the public API in
`tests/ops/elementwise_test.cpp`, `tests/ops/sequential_test.cpp`, and
`tests/ops/elementwise_ops_test.cpp`.

## How to run

Use a Release build (the default when no build type is given). Do not enable
`MENSURFLOW_NATIVE` for these runs, so every machine uses the same instruction set.

```
cmake -S . -B build-bench -DMENSURFLOW_CUDA=OFF
cmake --build build-bench --config Release
```

Add `-G "MinGW Makefiles"` (MinGW in PATH) or `-G "Visual Studio 18 2026" -A x64` (MSVC) as
needed. `-DMENSURFLOW_CUDA=OFF` keeps the GPU out of the `dot_benchmark` tables.

Run (the executable is in `build-bench` for Makefiles, `build-bench\Release` for Visual Studio):

```
cpu_parallel_benchmark > cpu_parallel_<machine>.txt
dot_benchmark > dot_<machine>.txt
```

Each size prints a `sequential` row (the plain loop) and `parallel` rows for 1, 2, 4, 8, 16, half
the hardware threads (the physical core count on SMT CPUs) and all hardware threads. The speedup
column is always sequential time / that row's time. The `parallel` row with 1 thread shows what the
parallel machinery costs when there is nothing to gain.

Conditions to keep equal and to write down for every run:

- Plugged in, power plan set to High performance (or equivalent).
- Other programs closed. Run twice and compare: if two runs differ by more than ~10 %, the machine
  was busy and the numbers are noise.
- Compiler and version (GCC, MSVC and Clang can produce different speeds for the same code).
- Commit hash of the code.

## Ryzen 5 6600H: elementwise cutoff checks (2026-10-31)

| Setting | Value |
|---|---|
| CPU | AMD Ryzen 5 6600H, 6 cores / 12 hardware threads |
| RAM | Approximately 14.8 GiB reported usable memory |
| OS | Linux 6.8.0-110-generic |
| Compiler | GCC 13.3.0 |
| Build | Release, `-O3 -DNDEBUG -fopenmp`, `MENSURFLOW_NATIVE=OFF`, GPU backends off |
| Code | `08fe3ae` plus the local benchmark changes adding sub/div and cutoff probes |
| Thread limit | `OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE`; rows measured with 1, 2, and 4 threads |
| Method | Preallocated staggered arrays, warmup, median of 9 samples, two complete runs |
| Power/background conditions | Not recorded |

The table preserves the two cutoff runs before deleting the temporary `.txt` files.
Values separated by `/` are run 1 / run 2. Times are milliseconds at the benchmark's
printed precision. Speedups use the unrounded internal timings. Every measured result
passed the benchmark's comparison against its sequential implementation.

| Op | n | Sequential (ms) | Parallel, 4 threads (ms) | Speedup, 2 threads | Speedup, 4 threads |
|---|---:|---|---|---|---|
| add | 32,769 | 0.004 / 0.004 | 0.001 / 0.002 | 1.712x / 1.883x | 2.620x / 2.473x |
| add | 40,000 | 0.005 / 0.005 | 0.002 / 0.002 | 1.843x / 1.885x | 2.652x / 2.802x |
| add | 50,000 | 0.009 / 0.006 | 0.002 / 0.002 | 2.437x / 2.069x | 3.604x / 3.216x |
| add | 75,000 | 0.011 / 0.010 | 0.003 / 0.003 | 1.856x / 2.339x | 3.597x / 3.520x |
| add | 100,000 | 0.015 / 0.013 | 0.004 / 0.004 | 2.348x / 1.957x | 3.827x / 3.489x |
| add | 200,000 | 0.035 / 0.035 | 0.008 / 0.008 | 2.516x / 2.550x | 4.220x / 4.482x |
| add | 500,000 | 0.082 / 0.079 | 0.031 / 0.024 | 2.359x / 2.147x | 2.662x / 3.262x |
| add | 1,000,000 | 0.622 / 0.378 | 0.216 / 0.182 | 1.713x / 1.461x | 2.883x / 2.078x |
| add | 2,000,000 | 1.721 / 1.687 | 1.013 / 1.315 | 1.240x / 1.254x | 1.700x / 1.283x |
| add | 5,000,000 | 5.457 / 4.427 | 4.046 / 4.057 | 1.219x / 1.106x | 1.349x / 1.091x |
| add | 10,000,000 | 7.728 / 10.208 | 6.930 / 7.967 | 1.147x / 1.205x | 1.115x / 1.281x |
| add | 33,554,432 | 27.383 / 25.411 | 23.583 / 31.126 | 1.155x / 1.057x | 1.161x / 0.816x |
| sub | 32,769 | 0.004 / 0.005 | 0.002 / 0.002 | 1.907x / 1.889x | 2.579x / 2.480x |
| sub | 40,000 | 0.005 / 0.008 | 0.002 / 0.002 | 1.747x / 1.782x | 2.503x / 3.432x |
| sub | 50,000 | 0.007 / 0.013 | 0.002 / 0.003 | 2.241x / 2.555x | 3.407x / 3.661x |
| sub | 75,000 | 0.011 / 0.014 | 0.003 / 0.005 | 2.019x / 2.198x | 3.881x / 2.794x |
| sub | 100,000 | 0.012 / 0.017 | 0.003 / 0.004 | 1.836x / 2.214x | 3.564x / 4.566x |
| sub | 200,000 | 0.024 / 0.042 | 0.008 / 0.009 | 1.894x / 2.275x | 2.990x / 4.556x |
| sub | 500,000 | 0.066 / 0.133 | 0.020 / 0.025 | 1.779x / 2.550x | 3.255x / 5.283x |
| sub | 1,000,000 | 0.271 / 0.470 | 0.090 / 0.107 | 2.191x / 2.159x | 3.015x / 4.389x |
| sub | 2,000,000 | 1.275 / 1.399 | 0.926 / 1.277 | 1.410x / 1.215x | 1.377x / 1.096x |
| sub | 5,000,000 | 3.363 / 3.797 | 3.172 / 3.389 | 1.128x / 1.150x | 1.060x / 1.120x |
| sub | 10,000,000 | 7.819 / 7.759 | 6.757 / 7.251 | 1.203x / 1.155x | 1.157x / 1.070x |
| sub | 33,554,432 | 25.363 / 27.859 | 22.131 / 23.441 | 1.158x / 1.187x | 1.146x / 1.188x |
| mul | 32,769 | 0.003 / 0.003 | 0.001 / 0.002 | 1.431x / 1.427x | 2.042x / 1.674x |
| mul | 40,000 | 0.004 / 0.004 | 0.002 / 0.002 | 1.520x / 1.545x | 2.165x / 2.249x |
| mul | 50,000 | 0.005 / 0.005 | 0.002 / 0.002 | 1.532x / 1.613x | 2.429x / 2.308x |
| mul | 75,000 | 0.008 / 0.008 | 0.003 / 0.003 | 1.653x / 1.697x | 2.952x / 2.772x |
| mul | 100,000 | 0.010 / 0.010 | 0.003 / 0.003 | 1.600x / 1.593x | 2.773x / 2.916x |
| mul | 200,000 | 0.021 / 0.024 | 0.008 / 0.008 | 1.636x / 1.952x | 2.674x / 3.215x |
| mul | 500,000 | 0.061 / 0.060 | 0.023 / 0.021 | 1.428x / 1.770x | 2.598x / 2.856x |
| mul | 1,000,000 | 0.311 / 0.205 | 0.059 / 0.085 | 2.042x / 1.930x | 5.275x / 2.412x |
| mul | 2,000,000 | 1.111 / 1.331 | 1.049 / 1.043 | 1.106x / 1.263x | 1.059x / 1.276x |
| mul | 5,000,000 | 3.199 / 4.302 | 3.579 / 4.021 | 0.935x / 1.049x | 0.894x / 1.070x |
| mul | 10,000,000 | 7.679 / 7.743 | 6.570 / 6.476 | 1.166x / 1.266x | 1.169x / 1.196x |
| mul | 33,554,432 | 24.352 / 24.310 | 22.999 / 22.575 | 1.091x / 1.148x | 1.059x / 1.077x |
| div | 32,769 | 0.006 / 0.006 | 0.002 / 0.002 | 1.648x / 1.700x | 2.731x / 2.815x |
| div | 40,000 | 0.008 / 0.008 | 0.003 / 0.003 | 1.698x / 1.728x | 2.871x / 2.988x |
| div | 50,000 | 0.010 / 0.010 | 0.003 / 0.003 | 1.724x / 1.759x | 2.964x / 3.040x |
| div | 75,000 | 0.014 / 0.015 | 0.004 / 0.004 | 1.810x / 1.896x | 3.310x / 3.417x |
| div | 100,000 | 0.019 / 0.020 | 0.006 / 0.006 | 1.861x / 1.903x | 3.293x / 3.359x |
| div | 200,000 | 0.039 / 0.040 | 0.012 / 0.012 | 1.930x / 1.927x | 3.285x / 3.288x |
| div | 500,000 | 0.101 / 0.102 | 0.030 / 0.030 | 2.007x / 2.040x | 3.370x / 3.468x |
| div | 1,000,000 | 0.250 / 0.220 | 0.064 / 0.120 | 2.072x / 1.222x | 3.897x / 1.837x |
| div | 2,000,000 | 1.168 / 1.603 | 0.853 / 1.025 | 1.405x / 1.540x | 1.369x / 1.564x |
| div | 5,000,000 | 3.186 / 3.753 | 3.124 / 3.703 | 0.936x / 1.070x | 1.020x / 1.014x |
| div | 10,000,000 | 7.270 / 8.190 | 9.358 / 7.242 | 0.985x / 1.087x | 0.777x / 1.131x |
| div | 33,554,432 | 28.137 / 23.885 | 22.222 / 22.177 | 1.264x / 1.054x | 1.266x / 1.077x |

Observations and cutoff recommendation:

- All four operations benefit from 2 and 4 threads at 32,769 elements in both runs.
  Keep the existing 32,768-element cutoff as a conservative initial setting for this
  machine with a four-thread limit. It is not a measured optimum: parallel execution
  below that threshold was not tested.
- Rows at or below 32,768 elements use the serial fallback, even when labelled
  `parallel`. They cannot establish the cost of actual multithreaded execution.
- Benefits often shrink for large arrays; add/sub/mul are primarily limited by data
  movement. Division does more work per element but also approaches the memory limit
  on large inputs. A single lower cutoff does not guarantee a gain at every larger size.
- Some timings differ by more than 10% across runs. The consistent wins near the cutoff
  support the recommendation, while individual large-input speedups remain provisional.
- These results do not justify applying a four-thread policy or this cutoff to every
  machine, or changing the shared threshold for unrelated operations such as dot.

Reproduce with the current benchmark in a Release CPU build:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark
```

### Negation (`neg`): shared elementwise implementation

Negation is implemented in `src/backends/cpu/elementwise.cpp` and declared in
`elementwise.hpp`, alongside add/sub/mul/div. It uses the shared unary loop and
`parallel_for`; its plain sequential baseline is in `sequential.cpp`.

The following two runs use `cpu_parallel_benchmark`, which now measures negation
alongside the other operations. The machine, Release build, and four-thread limit
are the same as above. These values describe the grouped implementation.
Negation reads one float and writes one float per element (8 logical bytes).
Tensor allocation and packing strided inputs are outside the measured kernel calls.

As with the binary kernels, `parallel_for` uses its serial fallback at or below
32,768 elements or with one thread. Every size and thread count passed the
benchmark's exact comparison against the sequential baseline.

| n | Sequential (ms) | Parallel, 4 threads (ms) | Speedup, 2 threads | Speedup, 4 threads |
|---:|---|---|---|---|
| 1,000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.530x / 0.540x | 0.531x / 0.538x |
| 10,000 | 0.001 / 0.001 | 0.001 / 0.001 | 0.535x / 0.584x | 0.534x / 0.567x |
| 32,768 | 0.002 / 0.002 | 0.004 / 0.004 | 0.519x / 0.224x | 0.516x / 0.501x |
| 32,769 | 0.002 / 0.002 | 0.001 / 0.001 | 1.181x / 1.105x | 1.474x / 1.627x |
| 40,000 | 0.002 / 0.002 | 0.001 / 0.002 | 1.352x / 1.194x | 1.846x / 1.454x |
| 50,000 | 0.003 / 0.003 | 0.002 / 0.002 | 1.548x / 1.473x | 2.023x / 2.023x |
| 75,000 | 0.005 / 0.006 | 0.004 / 0.002 | 1.556x / 1.534x | 1.527x / 2.406x |
| 100,000 | 0.008 / 0.008 | 0.002 / 0.003 | 1.717x / 1.452x | 3.233x / 3.020x |
| 200,000 | 0.013 / 0.020 | 0.005 / 0.009 | 1.525x / 1.733x | 2.714x / 2.363x |
| 500,000 | 0.034 / 0.063 | 0.014 / 0.015 | 1.499x / 1.802x | 2.452x / 4.120x |
| 1,000,000 | 0.103 / 0.163 | 0.042 / 0.057 | 2.116x / 1.621x | 2.432x / 2.884x |
| 2,000,000 | 0.553 / 0.778 | 0.308 / 0.415 | 1.621x / 1.356x | 1.794x / 1.876x |
| 5,000,000 | 2.223 / 3.260 | 1.921 / 2.681 | 1.017x / 0.958x | 1.157x / 1.216x |
| 10,000,000 | 4.985 / 8.239 | 4.441 / 4.609 | 1.139x / 1.678x | 1.122x / 1.787x |
| 33,554,432 | 17.769 / 17.715 | 15.720 / 15.566 | 1.114x / 1.167x | 1.130x / 1.138x |

Classification: little arithmetic per element, so large negation workloads are
primarily memory-bound. Small workloads also depend on thread and call overhead.

Recommended initial cutoff: retain 32,768 elements for this machine with up to
four threads, since the first tested parallel size (32,769) improves in both runs.
This does not establish the optimal threshold below 32,768 or the best thread
count for every size. Large-input gains vary and should not be generalized.

Some timings vary by more than 10% between runs. Keep both results rather than
selecting the faster run. A displayed time of 0.000 ms is below print precision,
not a zero-duration operation. Exact ratios are provisional.

Run the common benchmark without creating output files in the repository:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark
```

Negation uses the existing test groups: `elementwise_test.cpp` for CPU kernels,
`sequential_test.cpp` for sequential/parallel agreement and special floating-point
values, and `elementwise_ops_test.cpp` for the public API, views, and errors.
Its float64 reference is in `tests/reference/elementwise.hpp`.

### Absolute value, square, and ReLU: shared unary operations

These CPU float32 operations use the same elementwise files, unary loop, dispatcher,
reference file, test groups, and benchmark as negation. Sequential versions are in
`src/backends/cpu/sequential.cpp`; parallel versions use `parallel_for` in
`src/backends/cpu/elementwise.cpp`.

- `abs(x)` clears the sign bit, including for negative zero and NaNs.
- `square(x)` computes `x * x` with float32 rounding, overflow, and underflow.
- `relu(x)` returns +0 for negative values and otherwise preserves the input,
  including signed zeros and NaNs.

Two runs were measured on the Ryzen 5 6600H with the Release configuration above,
GCC 13.3.0, no `-march=native`, and `OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE`.
Code: `d016986` plus the local abs/square/relu changes. Power/background conditions
were not recorded. Each operation reads and writes one float per element (8 logical
bytes). Values separated by `/` are run 1 / run 2; timings are kernel-only milliseconds.
Every measured output matched its sequential baseline bit for bit.

| Op | n | Sequential (ms) | Parallel, 4 threads (ms) | Speedup, 2 threads | Speedup, 4 threads |
|---|---:|---|---|---|---|
| abs | 1,000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.527x / 0.516x | 0.526x / 0.513x |
| abs | 10,000 | 0.001 / 0.001 | 0.001 / 0.001 | 0.513x / 0.520x | 0.524x / 0.520x |
| abs | 32,768 | 0.002 / 0.002 | 0.004 / 0.004 | 0.522x / 0.520x | 0.499x / 0.500x |
| abs | 32,769 | 0.002 / 0.002 | 0.001 / 0.001 | 1.215x / 1.255x | 1.606x / 1.680x |
| abs | 40,000 | 0.002 / 0.002 | 0.001 / 0.001 | 1.256x / 1.366x | 1.715x / 1.851x |
| abs | 50,000 | 0.003 / 0.003 | 0.002 / 0.001 | 1.467x / 1.521x | 2.039x / 2.097x |
| abs | 75,000 | 0.006 / 0.005 | 0.002 / 0.002 | 1.925x / 1.671x | 2.903x / 2.461x |
| abs | 100,000 | 0.008 / 0.007 | 0.002 / 0.002 | 1.960x / 1.862x | 3.404x / 3.210x |
| abs | 200,000 | 0.013 / 0.015 | 0.004 / 0.005 | 1.645x / 1.730x | 2.964x / 3.042x |
| abs | 500,000 | 0.047 / 0.037 | 0.014 / 0.017 | 2.116x / 1.520x | 3.373x / 2.239x |
| abs | 1,000,000 | 0.242 / 0.118 | 0.030 / 0.028 | 5.028x / 2.108x | 8.154x / 4.284x |
| abs | 2,000,000 | 0.806 / 0.651 | 0.363 / 0.256 | 1.690x / 1.342x | 2.222x / 2.542x |
| abs | 5,000,000 | 3.043 / 2.225 | 2.460 / 2.079 | 1.335x / 1.134x | 1.237x / 1.070x |
| abs | 10,000,000 | 6.302 / 5.114 | 4.557 / 4.882 | 1.352x / 1.145x | 1.383x / 1.047x |
| abs | 33,554,432 | 16.823 / 18.359 | 16.073 / 16.149 | 1.078x / 1.224x | 1.047x / 1.137x |
| square | 1,000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.526x / 0.525x | 0.528x / 0.522x |
| square | 10,000 | 0.001 / 0.001 | 0.001 / 0.001 | 0.523x / 0.540x | 0.548x / 0.543x |
| square | 32,768 | 0.002 / 0.002 | 0.004 / 0.004 | 0.515x / 0.552x | 0.490x / 0.527x |
| square | 32,769 | 0.002 / 0.002 | 0.001 / 0.001 | 1.138x / 1.380x | 1.572x / 1.640x |
| square | 40,000 | 0.002 / 0.003 | 0.002 / 0.001 | 1.280x / 1.532x | 1.651x / 1.973x |
| square | 50,000 | 0.003 / 0.003 | 0.002 / 0.002 | 1.559x / 1.250x | 2.132x / 2.146x |
| square | 75,000 | 0.006 / 0.005 | 0.002 / 0.002 | 1.919x / 1.608x | 2.866x / 2.441x |
| square | 100,000 | 0.007 / 0.008 | 0.002 / 0.003 | 1.742x / 1.853x | 2.900x / 2.976x |
| square | 200,000 | 0.014 / 0.013 | 0.005 / 0.005 | 1.533x / 1.345x | 2.496x / 2.587x |
| square | 500,000 | 0.039 / 0.036 | 0.014 / 0.015 | 1.801x / 1.666x | 2.758x / 2.506x |
| square | 1,000,000 | 0.111 / 0.089 | 0.026 / 0.030 | 2.765x / 1.695x | 4.242x / 3.020x |
| square | 2,000,000 | 0.561 / 0.624 | 0.239 / 0.341 | 1.575x / 1.384x | 2.343x / 1.832x |
| square | 5,000,000 | 2.443 / 2.425 | 2.117 / 2.185 | 1.162x / 1.171x | 1.154x / 1.110x |
| square | 10,000,000 | 4.813 / 5.486 | 4.974 / 4.611 | 0.950x / 1.180x | 0.968x / 1.190x |
| square | 33,554,432 | 18.664 / 16.933 | 15.630 / 14.878 | 0.903x / 1.052x | 1.194x / 1.138x |
| relu | 1,000 | 0.000 / 0.000 | 0.000 / 0.000 | 1.620x / 1.632x | 1.625x / 1.633x |
| relu | 10,000 | 0.001 / 0.001 | 0.001 / 0.001 | 1.466x / 1.299x | 1.447x / 1.298x |
| relu | 32,768 | 0.004 / 0.004 | 0.003 / 0.002 | 1.453x / 1.510x | 1.466x / 1.510x |
| relu | 32,769 | 0.004 / 0.004 | 0.002 / 0.002 | 1.508x / 1.516x | 2.255x / 2.174x |
| relu | 40,000 | 0.005 / 0.004 | 0.002 / 0.002 | 1.640x / 1.553x | 2.365x / 2.268x |
| relu | 50,000 | 0.006 / 0.006 | 0.003 / 0.002 | 1.551x / 1.704x | 2.218x / 2.668x |
| relu | 75,000 | 0.010 / 0.009 | 0.004 / 0.004 | 1.800x / 1.730x | 2.674x / 2.494x |
| relu | 100,000 | 0.011 / 0.011 | 0.004 / 0.004 | 1.606x / 1.744x | 2.736x / 2.873x |
| relu | 200,000 | 0.048 / 0.023 | 0.007 / 0.008 | 3.430x / 1.772x | 6.465x / 2.887x |
| relu | 500,000 | 0.059 / 0.058 | 0.021 / 0.019 | 1.555x / 1.777x | 2.784x / 3.082x |
| relu | 1,000,000 | 0.146 / 0.124 | 0.052 / 0.047 | 2.275x / 1.414x | 2.822x / 2.629x |
| relu | 2,000,000 | 0.976 / 0.520 | 0.418 / 0.230 | 2.014x / 1.605x | 2.337x / 2.256x |
| relu | 5,000,000 | 2.570 / 2.340 | 2.334 / 2.015 | 1.183x / 1.087x | 1.101x / 1.161x |
| relu | 10,000,000 | 5.182 / 5.081 | 4.518 / 4.868 | 1.003x / 1.017x | 1.147x / 1.044x |
| relu | 33,554,432 | 17.433 / 18.829 | 16.332 / 15.299 | 0.995x / 1.180x | 1.067x / 1.231x |

Cutoff recommendation: retain the existing 32,768-element cutoff as an initial
setting for these operations on this machine with a four-thread limit. All three
benefit from four threads at 32,769 elements in both runs. Four threads also beat
two at that size. This does not identify an optimum below the cutoff or a universal
thread policy. Rows at or below 32,768 elements use the serial fallback.

All three do little computation per element, so large workloads are primarily
memory-bound; small workloads also depend on thread and call overhead. Gains shrink
or vary for larger arrays. Several timings vary by more than 10% across runs, so the
exact ratios remain provisional. Both runs are retained, including outliers; displayed
0.000 ms values are below print precision, not zero-duration calls.

The existing shared tests cover float64 references, exact sequential/parallel
agreement with 1/2/3/maximum threads, sizes around the cutoff, in-place use, tensor
views, empty and unsupported inputs, signed zeros, NaNs, infinities, and square
rounding/overflow/underflow. Public API examples are also checked against known values.

Reproduce through the common benchmark:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark
```

## Machine 1: laptop

| | |
|---|---|
| Model | 20S6001YMZ (Lenovo machine type) |
| CPU | Intel Core i7-10510U (Comet Lake, 14 nm) |
| Cores / threads | 4 / 8 |
| Clock | 1.8 GHz base, boosts higher (idle reading was 801 MHz) |
| Cache | L2 1 MB (256 KB per core), L3 8 MB |
| RAM | 2 × 16 GB DDR4 = 31.8 GB, one stick at 3200 MHz and one at 2667 MHz (mismatched, so the memory probably runs at the slower speed) |
| OS | Windows 11 Pro, build 26200 |
| Compiler | GCC 16.1.0 (MSYS2 MinGW-w64), `-O3 -DNDEBUG -fopenmp`, no `-march=native` |
| OpenMP | 5.2 (libgomp) |
| Code | branch `feat/dispatcher-dot-elementwise`, commit `61187fc` plus the sequential implementation and the updated benchmark (uncommitted when measured) |
| Power state | **On battery, "Balanced" plan**, with Chrome running in the background (checked: battery 65 %, CPU idling at 801 MHz). The CPU may have been running below its full speed, and the runs vary a lot (see Results) |

### Results (two runs, sequential implementation as the baseline)

The baseline is the plain sequential kernel (`src/backends/cpu/sequential.cpp`: one loop, no OpenMP).
Cells show run 3 / run 4 (the two runs with the final benchmark). "8 threads" is the parallel
kernel with all 8 hardware threads; "best" is the fastest thread count, in brackets.

| Op | n | Sequential (ms) | Speedup, 8 threads | Speedup, best (threads) |
|---|---|---|---|---|
| add | 100,000 | 0.032 / 0.020 | 0.30x / 0.34x | 1.14x (1) / 0.87x (1) |
| add | 1,000,000 | 0.573 / 0.465 | 1.07x / 1.22x | 1.07x (8) / 1.32x (2) |
| add | 10,000,000 | 7.81 / 6.36 | 1.37x / 1.09x | 1.37x (8) / 1.10x (4) |
| add | 33,554,432 | 23.4 / 21.0 | 1.10x / 1.05x | 1.10x (8) / 1.09x (4) |
| mul | 100,000 | 0.024 / 0.022 | 0.27x / 0.33x | 0.96x (1) / 1.05x (1) |
| mul | 1,000,000 | 0.561 / 0.433 | 1.01x / 0.79x | 1.26x (4) / 1.01x (2) |
| mul | 10,000,000 | 7.69 / 6.87 | 1.21x / 1.08x | 1.21x (8) / 1.09x (4) |
| mul | 33,554,432 | 28.6 / 25.9 | 1.12x / 1.05x | 1.38x (4) / 1.16x (4) |
| dot | 100,000 | 0.187 / 0.149 | 1.87x / 0.97x | 2.15x (4) / 1.01x (4) |
| dot | 1,000,000 | 1.41 / 1.90 | 3.96x / 2.48x | 3.96x (8) / 4.27x (4) |
| dot | 10,000,000 | 16.5 / 16.1 | 4.57x / 3.52x | 4.57x (8) / 3.52x (8) |
| dot | 33,554,432 | 51.2 / 54.6 | 3.91x / 5.37x | 3.91x (8) / 5.37x (8) |

What it shows:

- `dot` scales with threads: 3.5-5.4x with 8 threads at 10 million elements and above. The sequential
  `dot` is slow (about 5 GB/s) because every product is accumulated in double precision, so it is
  limited by computation, not memory.
- `add` and `mul` gain little: about 1.0-1.4x at best. The sequential loop already reaches about
  19 GB/s at 10 million elements, which is this laptop's memory bandwidth, so extra threads have
  nothing left to speed up.
- At 100,000 elements the parallel `add` and `mul` with 8 threads are 3-4x slower than the sequential
  loop (0.27-0.34x). The data fits in cache, so starting threads costs more than it saves. The
  cutoff (32,768 elements) is too low for the elementwise ops.
- The parallel kernel with 1 thread stays within about +-20 % of the sequential loop (0.76-1.21x
  across both runs), which is the noise level here, so the parallel machinery shows no consistent
  cost when it has nothing to gain.
- Elementwise results are bit-identical between the two implementations (checked after every timing,
  and in `tests/ops/sequential_test.cpp`). `dot` agrees within float rounding, and exactly with 1 thread.

Reliability of these numbers: the laptop was on battery (Balanced plan, CPU idling at 800 MHz) with
Chrome running, and the same row can differ by 30-40 % between the two runs (for example `dot` at
1 million elements, 2.5x vs 4.0x). Read the conclusions above (dot scales, add/mul do not, cutoff too
low), not the individual ratios. For comparable numbers, plug in, use High performance and close
other programs.

Benchmark method note: an earlier version of the benchmark allocated its four arrays at the same
offset within a page, and the sequential `mul` came out about 2x slower than `add`
(0.85 ms vs 0.44 ms at 1 million elements). That is the usual signature of "4K aliasing" (loads and
stores whose addresses agree in the low 12 bits falsely depend on each other). The benchmark now
starts the arrays 1 KB apart, and `mul` matches `add`. Results taken before this change are not
comparable.

<details>
<summary>Full output: cpu_parallel_benchmark, run 3</summary>

```
Sequential vs parallel on the CPU (float32)
compiler: gcc 16.1.0, OpenMP: yes, max threads: 8, hardware threads: 8, parallel cutoff: 32768 elements
sequential = plain loop, no OpenMP. parallel = parallel kernel with N threads.
Speedup = sequential time / this time. Median of 9 samples.

== add ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    68.151     1.000x
        1000    parallel        1       0.000    71.460     1.049x
        1000    parallel        2       0.000    77.729     1.141x
        1000    parallel        4       0.000    80.610     1.183x
        1000    parallel        8       0.000    81.621     1.198x

       10000  sequential        -       0.003    47.650     1.000x
       10000    parallel        1       0.002    50.589     1.062x
       10000    parallel        2       0.002    51.421     1.079x
       10000    parallel        4       0.002    50.012     1.050x
       10000    parallel        8       0.002    52.463     1.101x

      100000  sequential        -       0.032    37.836     1.000x
      100000    parallel        1       0.028    43.408     1.147x
      100000    parallel        2       0.069    17.485     0.462x
      100000    parallel        4       0.077    15.604     0.412x
      100000    parallel        8       0.106    11.352     0.300x

     1000000  sequential        -       0.573    20.947     1.000x
     1000000    parallel        1       0.755    15.897     0.759x
     1000000    parallel        2       0.734    16.345     0.780x
     1000000    parallel        4       0.662    18.138     0.866x
     1000000    parallel        8       0.537    22.329     1.066x

    10000000  sequential        -       7.806    15.373     1.000x
    10000000    parallel        1       6.650    18.046     1.174x
    10000000    parallel        2       6.591    18.207     1.184x
    10000000    parallel        4       6.178    19.422     1.263x
    10000000    parallel        8       5.680    21.128     1.374x

    33554432  sequential        -      23.403    17.205     1.000x
    33554432    parallel        1      22.763    17.689     1.028x
    33554432    parallel        2      21.565    18.672     1.085x
    33554432    parallel        4      23.588    17.070     0.992x
    33554432    parallel        8      21.339    18.869     1.097x


== mul ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    80.072     1.000x
        1000    parallel        1       0.000   105.149     1.313x
        1000    parallel        2       0.000    95.855     1.197x
        1000    parallel        4       0.000    96.141     1.201x
        1000    parallel        8       0.000    96.362     1.203x

       10000  sequential        -       0.002    57.338     1.000x
       10000    parallel        1       0.002    55.713     0.972x
       10000    parallel        2       0.002    57.260     0.999x
       10000    parallel        4       0.002    55.210     0.963x
       10000    parallel        8       0.002    55.959     0.976x

      100000  sequential        -       0.024    49.580     1.000x
      100000    parallel        1       0.025    47.779     0.964x
      100000    parallel        2       0.064    18.852     0.380x
      100000    parallel        4       0.076    15.704     0.317x
      100000    parallel        8       0.088    13.561     0.274x

     1000000  sequential        -       0.561    21.406     1.000x
     1000000    parallel        1       0.599    20.040     0.936x
     1000000    parallel        2       0.560    21.414     1.000x
     1000000    parallel        4       0.445    26.960     1.259x
     1000000    parallel        8       0.555    21.618     1.010x

    10000000  sequential        -       7.686    15.612     1.000x
    10000000    parallel        1       7.699    15.585     0.998x
    10000000    parallel        2       7.327    16.378     1.049x
    10000000    parallel        4       6.712    17.877     1.145x
    10000000    parallel        8       6.333    18.947     1.214x

    33554432  sequential        -      28.640    14.059     1.000x
    33554432    parallel        1      29.267    13.758     0.979x
    33554432    parallel        2      23.930    16.826     1.197x
    33554432    parallel        4      20.793    19.365     1.377x
    33554432    parallel        8      25.614    15.720     1.118x


== dot ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001     5.966     1.000x
        1000    parallel        1       0.002     5.226     0.876x
        1000    parallel        2       0.002     5.021     0.842x
        1000    parallel        4       0.002     5.136     0.861x
        1000    parallel        8       0.002     4.536     0.760x

       10000  sequential        -       0.013     5.977     1.000x
       10000    parallel        1       0.014     5.919     0.990x
       10000    parallel        2       0.013     6.171     1.032x
       10000    parallel        4       0.013     6.087     1.018x
       10000    parallel        8       0.015     5.371     0.899x

      100000  sequential        -       0.187     4.269     1.000x
      100000    parallel        1       0.214     3.733     0.874x
      100000    parallel        2       0.143     5.610     1.314x
      100000    parallel        4       0.087     9.154     2.144x
      100000    parallel        8       0.100     7.980     1.869x

     1000000  sequential        -       1.407     5.684     1.000x
     1000000    parallel        1       1.477     5.418     0.953x
     1000000    parallel        2       0.796    10.050     1.768x
     1000000    parallel        4       0.475    16.853     2.965x
     1000000    parallel        8       0.355    22.541     3.966x

    10000000  sequential        -      16.508     4.846     1.000x
    10000000    parallel        1      15.534     5.150     1.063x
    10000000    parallel        2       7.914    10.109     2.086x
    10000000    parallel        4       4.495    17.797     3.672x
    10000000    parallel        8       3.616    22.125     4.565x

    33554432  sequential        -      51.190     5.244     1.000x
    33554432    parallel        1      51.167     5.246     1.000x
    33554432    parallel        2      27.212     9.865     1.881x
    33554432    parallel        4      14.184    18.925     3.609x
    33554432    parallel        8      13.107    20.480     3.906x

```

</details>

<details>
<summary>Full output: cpu_parallel_benchmark, run 4</summary>

```
Sequential vs parallel on the CPU (float32)
compiler: gcc 16.1.0, OpenMP: yes, max threads: 8, hardware threads: 8, parallel cutoff: 32768 elements
sequential = plain loop, no OpenMP. parallel = parallel kernel with N threads.
Speedup = sequential time / this time. Median of 9 samples.

== add ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   117.112     1.000x
        1000    parallel        1       0.000   116.391     0.994x
        1000    parallel        2       0.000   112.251     0.958x
        1000    parallel        4       0.000   111.871     0.955x
        1000    parallel        8       0.000   111.598     0.953x

       10000  sequential        -       0.002    68.964     1.000x
       10000    parallel        1       0.002    66.316     0.962x
       10000    parallel        2       0.002    65.456     0.949x
       10000    parallel        4       0.002    61.143     0.887x
       10000    parallel        8       0.002    66.692     0.967x

      100000  sequential        -       0.020    60.073     1.000x
      100000    parallel        1       0.023    51.632     0.859x
      100000    parallel        2       0.035    34.290     0.571x
      100000    parallel        4       0.039    30.841     0.513x
      100000    parallel        8       0.058    20.539     0.342x

     1000000  sequential        -       0.465    25.820     1.000x
     1000000    parallel        1       0.435    27.572     1.068x
     1000000    parallel        2       0.352    34.103     1.321x
     1000000    parallel        4       0.385    31.175     1.207x
     1000000    parallel        8       0.380    31.571     1.223x

    10000000  sequential        -       6.363    18.858     1.000x
    10000000    parallel        1       6.549    18.325     0.972x
    10000000    parallel        2       5.908    20.312     1.077x
    10000000    parallel        4       5.801    20.686     1.097x
    10000000    parallel        8       5.826    20.597     1.092x

    33554432  sequential        -      21.049    19.129     1.000x
    33554432    parallel        1      21.247    18.951     0.991x
    33554432    parallel        2      20.577    19.568     1.023x
    33554432    parallel        4      19.343    20.816     1.088x
    33554432    parallel        8      20.117    20.016     1.046x


== mul ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   101.089     1.000x
        1000    parallel        1       0.000    90.469     0.895x
        1000    parallel        2       0.000    90.453     0.895x
        1000    parallel        4       0.000    88.630     0.877x
        1000    parallel        8       0.000    87.049     0.861x

       10000  sequential        -       0.002    63.910     1.000x
       10000    parallel        1       0.002    62.114     0.972x
       10000    parallel        2       0.002    63.514     0.994x
       10000    parallel        4       0.002    63.177     0.989x
       10000    parallel        8       0.002    61.563     0.963x

      100000  sequential        -       0.022    54.576     1.000x
      100000    parallel        1       0.021    57.703     1.057x
      100000    parallel        2       0.034    35.068     0.643x
      100000    parallel        4       0.042    28.817     0.528x
      100000    parallel        8       0.066    18.262     0.335x

     1000000  sequential        -       0.433    27.696     1.000x
     1000000    parallel        1       0.488    24.599     0.888x
     1000000    parallel        2       0.427    28.077     1.014x
     1000000    parallel        4       0.638    18.824     0.680x
     1000000    parallel        8       0.549    21.873     0.790x

    10000000  sequential        -       6.874    17.456     1.000x
    10000000    parallel        1       8.069    14.871     0.852x
    10000000    parallel        2       6.785    17.685     1.013x
    10000000    parallel        4       6.289    19.081     1.093x
    10000000    parallel        8       6.376    18.820     1.078x

    33554432  sequential        -      25.918    15.536     1.000x
    33554432    parallel        1      26.429    15.235     0.981x
    33554432    parallel        2      24.552    16.400     1.056x
    33554432    parallel        4      22.318    18.041     1.161x
    33554432    parallel        8      24.663    16.326     1.051x


== dot ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001     6.428     1.000x
        1000    parallel        1       0.001     7.290     1.134x
        1000    parallel        2       0.001     6.092     0.948x
        1000    parallel        4       0.001     6.072     0.945x
        1000    parallel        8       0.001     6.561     1.021x

       10000  sequential        -       0.012     6.788     1.000x
       10000    parallel        1       0.014     5.676     0.836x
       10000    parallel        2       0.016     5.114     0.753x
       10000    parallel        4       0.014     5.683     0.837x
       10000    parallel        8       0.015     5.412     0.797x

      100000  sequential        -       0.149     5.383     1.000x
      100000    parallel        1       0.174     4.609     0.856x
      100000    parallel        2       0.150     5.324     0.989x
      100000    parallel        4       0.148     5.421     1.007x
      100000    parallel        8       0.154     5.204     0.967x

     1000000  sequential        -       1.896     4.220     1.000x
     1000000    parallel        1       1.568     5.103     1.209x
     1000000    parallel        2       0.824     9.707     2.300x
     1000000    parallel        4       0.444    18.016     4.270x
     1000000    parallel        8       0.763    10.492     2.486x

    10000000  sequential        -      16.058     4.982     1.000x
    10000000    parallel        1      19.638     4.074     0.818x
    10000000    parallel        2       8.764     9.128     1.832x
    10000000    parallel        4       6.163    12.980     2.605x
    10000000    parallel        8       4.566    17.520     3.517x

    33554432  sequential        -      54.628     4.914     1.000x
    33554432    parallel        1      50.326     5.334     1.085x
    33554432    parallel        2      25.308    10.607     2.159x
    33554432    parallel        4      13.486    19.905     4.051x
    33554432    parallel        8      10.170    26.394     5.371x

```

</details>

<details>
<summary>Full output: dot_benchmark (through the public API)</summary>

```
dot benchmark (float32), through mensurflow::dot
compiler: gcc 16.1.0, OpenMP: yes, CPU threads: 8, hardware threads: 8
Devices: cpu:0
Median of 7 samples; each call includes reading the result back.
GB/s counts 8 bytes per element (two float reads); GFLOP/s counts 2.

== 1. Throughput by device ==
    device           n          ms      GB/s   GFLOP/s    vs cpu:0
     cpu:0           1       0.002     0.005     0.001      1.000x
     cpu:0        1000       0.003     2.545     0.636      1.000x
     cpu:0      100000       0.097     8.267     2.067      1.000x
     cpu:0     1000000       0.296    27.054     6.764      1.000x
     cpu:0    10000000       3.433    23.300     5.825      1.000x
     cpu:0    33554432      11.364    23.622     5.906      1.000x

== 2. Cost of one call on 1 element (pure overhead) ==
    device  microseconds
     cpu:0         1.267

== 3. CPU: API vs direct kernel vs serial loop ==
           n      api ms   kernel ms   serial ms    api/kernel
           1       0.001       0.000       0.000      357.171x
        1000       0.002       0.001       0.001        2.106x
      100000       0.084       0.081       0.103        1.032x
     1000000       0.288       0.268       1.225        1.074x
    10000000       3.470       2.939      13.270        1.181x
    33554432      11.443      10.445      44.024        1.096x
```

</details>

## Machine 2: desktop

| | |
|---|---|
| CPU | AMD Ryzen 9 5900X (Zen 3, two core complexes) |
| Cores / threads | 12 / 24 |
| Clock | 3.7 GHz base, up to 4.8 GHz boost (published figures). Probably **no boost** during these runs, see Power state |
| Cache | L2 6 MB (512 KB per core), L3 64 MB (2 × 32 MB, one per core complex) |
| RAM | 4 × 32 GB DDR4 = 127.9 GB (Transcend JM3200HLE-32G, rated 3200), running at **3000 MT/s** (Windows reports "3000 MHz"). All four slots filled (A1, A2, B1, B2), so dual channel with two sticks per channel |
| OS | Windows 11 Pro 26H2, build 26300 |
| Compiler | MSVC 19.51.36260 (Visual Studio Build Tools 2026, x64), `/O2 /Ob2 /DNDEBUG -openmp`, generator `Visual Studio 18 2026`. **Differs from the laptop's GCC 16.1.0** |
| OpenMP | 2.0 (MSVC `-openmp`) |
| Code | branch `feat/dispatcher-dot-elementwise`, commit `62d734b` (clean working tree) |
| Power state | Desktop on mains power, custom power plan **"Dedera", not High performance**: minimum processor state 80 %, maximum 95 %. A maximum below 100 % usually disables boost on Ryzen, so the CPU probably ran near its 3.7 GHz base clock |

> **Needs a rerun.** These desktop results were measured with the earlier benchmark, whose
> "sequential" baseline was the parallel kernel limited to 1 thread, not the separate sequential
> implementation, and whose arrays were not staggered (see the method note under Machine 1). Rerun
> `cpu_parallel_benchmark` from the current branch and replace this section. The comparison table
> at the end is not valid until then.

### Results (two runs, earlier benchmark)

Best parallel time at each size, compared with the sequential (1 thread) run. Values are from
run 1. Where run 2 differed by more than 10 %, its value is given in brackets.

| Op | n | Sequential | Best parallel | Speedup |
|---|---|---|---|---|
| add | 100,000 | 0.012 ms | 0.022 ms (4 threads) | 0.58x (slower) |
| add | 1,000,000 | 0.130 ms | 0.087 ms (12 threads) | 1.51x |
| add | 10,000,000 | 4.06 ms | 3.15 ms (8 threads) | 1.29x [run 2: 2.96 ms with 12 threads, 1.38x] |
| add | 33,554,432 | 15.1 ms | 14.0 ms (12 threads) | 1.08x |
| mul | 100,000 | 0.013 ms | 0.024 ms (4 threads) | 0.55x (slower) |
| mul | 1,000,000 | 0.130 ms | 0.087 ms (12 threads) | 1.49x |
| mul | 10,000,000 | 3.97 ms | 2.66 ms (8 threads) | 1.49x [run 2: 3.36 ms with 12 threads, 1.19x] |
| mul | 33,554,432 | 15.1 ms | 13.7 ms (8 threads) | 1.11x |
| dot | 100,000 | 0.086 ms | 0.032 ms (8 threads) | 2.68x |
| dot | 1,000,000 | 0.860 ms | 0.079 ms (24 threads) | 10.9x |
| dot | 10,000,000 | 8.95 ms | 1.05 ms (12 threads) | 8.6x [run 2: 1.26 ms with 8 threads, 7.1x] |
| dot | 33,554,432 | 29.9 ms | 5.94 ms (8 threads) | 5.0x |

Run-to-run agreement: the median row differs by 0.9 % and every sequential row is within 9 %,
but 9 of the 126 rows differ by more than 10 %. All of them are multithreaded runs at 100,000 to
10,000,000 elements, including the 10 million rows above. The machine was idle (about 8 % CPU
load), so treat the parallel speedups for `add`/`mul` at 10 million as roughly ±0.2x.

What it shows:

- One thread is faster than on the laptop: about 30 GB/s for `add`/`mul` at 10 million elements
  (laptop 18.5) and 8.9 GB/s for `dot` (laptop 6.3), even without boost.
- `add` and `mul` are still limited by memory bandwidth once the data leaves the cache. At 10 million
  elements (120 MB) one thread reaches about 30 GB/s and the best thread count about 38-45 GB/s.
  Dual-channel DDR4-3000 peaks at 48 GB/s in theory, so there is little left for threads to gain.
  At 33.5 million elements every thread count is about 27-29 GB/s (1.1x).
- At 1 million elements (12 MB for the three arrays) the data fits in the 64 MB L3, so `add`/`mul`
  scale to about 1.5x with 8-12 threads and reach about 135 GB/s. 2 threads are slower than 1
  (0.62-0.67x), and 24 threads are slower than 12 in both runs.
- `dot` keeps scaling past 8 threads: 10.9x with 24 threads at 1 million elements (fits in cache, so
  the extra SMT threads still help), 8.6x with 12 threads at 10 million. Past 12 threads it drops
  at 10 million (5.8-6.9x with 16-24 threads). At 33.5 million it reaches the memory limit
  (about 43-45 GB/s) and stays at about 5x from 8 threads on.
- At 100,000 elements the threaded `add` and `mul` are 1.7-5x slower than sequential at every thread
  count (0.2-0.58x). The cutoff (32,768 elements) is still too low for them. For `dot` it is fine:
  2.5-2.7x at 100,000.

<details>
<summary>Full output: cpu_parallel_benchmark, run 1</summary>

```
Sequential vs parallel on the CPU (float32)
compiler: msvc, OpenMP: yes, max threads: 24, hardware threads: 24, parallel cutoff: 32768 elements
threads = 1 is the sequential run. Median of 9 samples.

== add ==
           n  threads          ms      GB/s    speedup
        1000        1       0.000   136.117     1.000x
        1000        2       0.000   136.842     1.005x
        1000        4       0.000   136.714     1.004x
        1000        8       0.000   136.671     1.004x
        1000       12       0.000   136.604     1.004x
        1000       16       0.000   136.108     1.000x
        1000       24       0.000   136.780     1.005x

       10000        1       0.001   109.911     1.000x
       10000        2       0.001   109.704     0.998x
       10000        4       0.001   109.817     0.999x
       10000        8       0.001   109.753     0.999x
       10000       12       0.001   109.768     0.999x
       10000       16       0.001   109.925     1.000x
       10000       24       0.001   109.749     0.999x

      100000        1       0.012    96.111     1.000x
      100000        2       0.029    41.414     0.431x
      100000        4       0.022    55.453     0.577x
      100000        8       0.027    43.731     0.455x
      100000       12       0.034    35.648     0.371x
      100000       16       0.034    35.297     0.367x
      100000       24       0.061    19.676     0.205x

     1000000        1       0.130    91.972     1.000x
     1000000        2       0.195    61.487     0.669x
     1000000        4       0.107   112.002     1.218x
     1000000        8       0.092   130.262     1.416x
     1000000       12       0.087   138.673     1.508x
     1000000       16       0.098   122.231     1.329x
     1000000       24       0.108   110.695     1.204x

    10000000        1       4.059    29.562     1.000x
    10000000        2       3.902    30.753     1.040x
    10000000        4       3.999    30.007     1.015x
    10000000        8       3.150    38.090     1.288x
    10000000       12       3.495    34.336     1.161x
    10000000       16       3.898    30.783     1.041x
    10000000       24       3.703    32.411     1.096x

    33554432        1      15.068    26.723     1.000x
    33554432        2      14.210    28.336     1.060x
    33554432        4      15.870    25.371     0.949x
    33554432        8      14.082    28.593     1.070x
    33554432       12      13.961    28.841     1.079x
    33554432       16      14.577    27.622     1.034x
    33554432       24      14.409    27.945     1.046x


== mul ==
           n  threads          ms      GB/s    speedup
        1000        1       0.000   137.143     1.000x
        1000        2       0.000   137.287     1.001x
        1000        4       0.000   137.282     1.001x
        1000        8       0.000   137.100     1.000x
        1000       12       0.000   137.234     1.001x
        1000       16       0.000   137.210     1.000x
        1000       24       0.000   136.990     0.999x

       10000        1       0.001   109.837     1.000x
       10000        2       0.001   109.822     1.000x
       10000        4       0.001   109.807     1.000x
       10000        8       0.001   109.709     0.999x
       10000       12       0.001   109.690     0.999x
       10000       16       0.001   109.832     1.000x
       10000       24       0.001   109.827     1.000x

      100000        1       0.013    93.428     1.000x
      100000        2       0.031    38.742     0.415x
      100000        4       0.024    50.900     0.545x
      100000        8       0.027    44.173     0.473x
      100000       12       0.034    35.270     0.378x
      100000       16       0.034    35.258     0.377x
      100000       24       0.041    29.035     0.311x

     1000000        1       0.130    92.544     1.000x
     1000000        2       0.203    59.172     0.639x
     1000000        4       0.106   113.512     1.227x
     1000000        8       0.093   128.980     1.394x
     1000000       12       0.087   137.654     1.487x
     1000000       16       0.105   114.764     1.240x
     1000000       24       0.116   103.348     1.117x

    10000000        1       3.969    30.236     1.000x
    10000000        2       3.738    32.105     1.062x
    10000000        4       3.692    32.498     1.075x
    10000000        8       2.661    45.104     1.492x
    10000000       12       3.001    39.993     1.323x
    10000000       16       3.815    31.456     1.040x
    10000000       24       3.768    31.849     1.053x

    33554432        1      15.106    26.655     1.000x
    33554432        2      15.154    26.570     0.997x
    33554432        4      14.968    26.901     1.009x
    33554432        8      13.658    29.482     1.106x
    33554432       12      14.038    28.683     1.076x
    33554432       16      15.069    26.720     1.002x
    33554432       24      14.650    27.485     1.031x


== dot ==
           n  threads          ms      GB/s    speedup
        1000        1       0.001     9.327     1.000x
        1000        2       0.001     9.329     1.000x
        1000        4       0.001     9.328     1.000x
        1000        8       0.001     9.332     1.000x
        1000       12       0.001     9.328     1.000x
        1000       16       0.001     9.331     1.000x
        1000       24       0.001     9.330     1.000x

       10000        1       0.009     9.307     1.000x
       10000        2       0.009     9.308     1.000x
       10000        4       0.009     9.308     1.000x
       10000        8       0.009     9.301     0.999x
       10000       12       0.009     9.308     1.000x
       10000       16       0.009     9.308     1.000x
       10000       24       0.009     9.308     1.000x

      100000        1       0.086     9.294     1.000x
      100000        2       0.055    14.614     1.572x
      100000        4       0.035    22.885     2.462x
      100000        8       0.032    24.929     2.682x
      100000       12       0.034    23.566     2.536x
      100000       16       0.037    21.817     2.347x
      100000       24       0.044    18.334     1.973x

     1000000        1       0.860     9.300     1.000x
     1000000        2       0.444    18.025     1.938x
     1000000        4       0.230    34.854     3.748x
     1000000        8       0.129    61.961     6.662x
     1000000       12       0.101    78.825     8.476x
     1000000       16       0.085    93.649    10.070x
     1000000       24       0.079   101.571    10.921x

    10000000        1       8.954     8.934     1.000x
    10000000        2       4.535    17.640     1.974x
    10000000        4       2.313    34.593     3.872x
    10000000        8       1.169    68.420     7.658x
    10000000       12       1.045    76.548     8.568x
    10000000       16       1.540    51.938     5.813x
    10000000       24       1.299    61.591     6.894x

    33554432        1      29.944     8.964     1.000x
    33554432        2      15.183    17.680     1.972x
    33554432        4       7.864    34.136     3.808x
    33554432        8       5.939    45.197     5.042x
    33554432       12       6.292    42.663     4.759x
    33554432       16       6.433    41.727     4.655x
    33554432       24       6.145    43.684     4.873x
```

</details>

<details>
<summary>Full output: cpu_parallel_benchmark, run 2</summary>

```
Sequential vs parallel on the CPU (float32)
compiler: msvc, OpenMP: yes, max threads: 24, hardware threads: 24, parallel cutoff: 32768 elements
threads = 1 is the sequential run. Median of 9 samples.

== add ==
           n  threads          ms      GB/s    speedup
        1000        1       0.000   136.957     1.000x
        1000        2       0.000   136.533     0.997x
        1000        4       0.000   137.215     1.002x
        1000        8       0.000   137.004     1.000x
        1000       12       0.000   137.105     1.001x
        1000       16       0.000   136.995     1.000x
        1000       24       0.000   137.176     1.002x

       10000        1       0.001   109.842     1.000x
       10000        2       0.001   109.842     1.000x
       10000        4       0.001   109.773     0.999x
       10000        8       0.001   109.857     1.000x
       10000       12       0.001   109.852     1.000x
       10000       16       0.001   109.812     1.000x
       10000       24       0.001   109.871     1.000x

      100000        1       0.012    96.860     1.000x
      100000        2       0.031    39.140     0.404x
      100000        4       0.023    51.327     0.530x
      100000        8       0.029    41.808     0.432x
      100000       12       0.034    35.330     0.365x
      100000       16       0.039    31.010     0.320x
      100000       24       0.044    27.473     0.284x

     1000000        1       0.128    93.860     1.000x
     1000000        2       0.205    58.460     0.623x
     1000000        4       0.111   108.487     1.156x
     1000000        8       0.094   127.182     1.355x
     1000000       12       0.089   135.159     1.440x
     1000000       16       0.086   139.530     1.487x
     1000000       24       0.117   102.828     1.096x

    10000000        1       4.097    29.288     1.000x
    10000000        2       3.775    31.792     1.086x
    10000000        4       3.866    31.038     1.060x
    10000000        8       3.138    38.245     1.306x
    10000000       12       2.963    40.497     1.383x
    10000000       16       3.794    31.631     1.080x
    10000000       24       3.938    30.473     1.040x

    33554432        1      16.460    24.462     1.000x
    33554432        2      14.634    27.515     1.125x
    33554432        4      15.169    26.544     1.085x
    33554432        8      14.386    27.989     1.144x
    33554432       12      14.790    27.224     1.113x
    33554432       16      14.877    27.065     1.106x
    33554432       24      14.943    26.946     1.102x


== mul ==
           n  threads          ms      GB/s    speedup
        1000        1       0.000   136.609     1.000x
        1000        2       0.000   136.928     1.002x
        1000        4       0.000   136.714     1.001x
        1000        8       0.000   136.895     1.002x
        1000       12       0.000   137.019     1.003x
        1000       16       0.000   136.947     1.002x
        1000       24       0.000   136.604     1.000x

       10000        1       0.001   109.768     1.000x
       10000        2       0.001   109.704     0.999x
       10000        4       0.001   109.803     1.000x
       10000        8       0.001   109.695     0.999x
       10000       12       0.001   109.700     0.999x
       10000       16       0.001   109.704     0.999x
       10000       24       0.001   109.665     0.999x

      100000        1       0.012    96.398     1.000x
      100000        2       0.029    41.331     0.429x
      100000        4       0.022    55.260     0.573x
      100000        8       0.029    41.824     0.434x
      100000       12       0.034    35.598     0.369x
      100000       16       0.037    32.273     0.335x
      100000       24       0.046    25.806     0.268x

     1000000        1       0.143    84.093     1.000x
     1000000        2       0.222    54.018     0.642x
     1000000        4       0.119   100.992     1.201x
     1000000        8       0.098   122.758     1.460x
     1000000       12       0.090   133.980     1.593x
     1000000       16       0.107   112.150     1.334x
     1000000       24       0.119   100.748     1.198x

    10000000        1       4.016    29.884     1.000x
    10000000        2       3.724    32.225     1.078x
    10000000        4       3.698    32.450     1.086x
    10000000        8       3.446    34.822     1.165x
    10000000       12       3.362    35.693     1.194x
    10000000       16       3.792    31.648     1.059x
    10000000       24       3.729    32.181     1.077x

    33554432        1      15.027    26.796     1.000x
    33554432        2      15.332    26.263     0.980x
    33554432        4      15.436    26.085     0.973x
    33554432        8      13.816    29.145     1.088x
    33554432       12      13.935    28.894     1.078x
    33554432       16      14.407    27.948     1.043x
    33554432       24      14.888    27.046     1.009x


== dot ==
           n  threads          ms      GB/s    speedup
        1000        1       0.001     9.319     1.000x
        1000        2       0.001     9.322     1.000x
        1000        4       0.001     9.315     1.000x
        1000        8       0.001     9.331     1.001x
        1000       12       0.001     9.312     0.999x
        1000       16       0.001     9.322     1.000x
        1000       24       0.001     9.329     1.001x

       10000        1       0.009     9.308     1.000x
       10000        2       0.009     9.307     1.000x
       10000        4       0.009     9.307     1.000x
       10000        8       0.009     9.308     1.000x
       10000       12       0.009     9.309     1.000x
       10000       16       0.009     9.298     0.999x
       10000       24       0.009     9.300     0.999x

      100000        1       0.086     9.289     1.000x
      100000        2       0.055    14.580     1.570x
      100000        4       0.035    23.163     2.494x
      100000        8       0.033    24.257     2.611x
      100000       12       0.034    23.374     2.516x
      100000       16       0.034    23.397     2.519x
      100000       24       0.043    18.783     2.022x

     1000000        1       0.860     9.301     1.000x
     1000000        2       0.457    17.507     1.882x
     1000000        4       0.254    31.442     3.380x
     1000000        8       0.129    62.016     6.668x
     1000000       12       0.099    80.744     8.681x
     1000000       16       0.085    94.011    10.108x
     1000000       24       0.078   102.090    10.976x

    10000000        1       8.988     8.901     1.000x
    10000000        2       4.536    17.638     1.982x
    10000000        4       2.335    34.266     3.850x
    10000000        8       1.260    63.515     7.136x
    10000000       12       1.293    61.891     6.954x
    10000000       16       1.509    53.005     5.955x
    10000000       24       1.407    56.867     6.389x

    33554432        1      29.963     8.959     1.000x
    33554432        2      15.130    17.742     1.980x
    33554432        4       7.894    34.004     3.796x
    33554432        8       6.590    40.735     4.547x
    33554432       12       6.597    40.693     4.542x
    33554432       16       6.517    41.191     4.598x
    33554432       24       6.108    43.948     4.906x
```

</details>

<details>
<summary>Full output: dot_benchmark (through the public API)</summary>

```
dot benchmark (float32), through mensurflow::dot
compiler: msvc, OpenMP: yes, CPU threads: 24, hardware threads: 24
Devices: cpu:0
Median of 7 samples; each call includes reading the result back.
GB/s counts 8 bytes per element (two float reads); GFLOP/s counts 2.

== 1. Throughput by device ==
    device           n          ms      GB/s   GFLOP/s    vs cpu:0
     cpu:0           1       0.001     0.007     0.002      1.000x
     cpu:0        1000       0.002     3.733     0.933      1.000x
     cpu:0      100000       0.045    17.767     4.442      1.000x
     cpu:0     1000000       0.082    97.814    24.454      1.000x
     cpu:0    10000000       1.959    40.835    10.209      1.000x
     cpu:0    33554432       6.690    40.123    10.031      1.000x

== 2. Cost of one call on 1 element (pure overhead) ==
    device  microseconds
     cpu:0         1.091

== 3. CPU: API vs direct kernel vs serial loop ==
           n      api ms   kernel ms   serial ms    api/kernel
           1       0.001       0.000       0.000      208.583x
        1000       0.002       0.001       0.001        2.262x
      100000       0.046       0.042       0.086        1.079x
     1000000       0.083       0.080       0.862        1.044x
    10000000       1.689       1.378       8.959        1.225x
    33554432       6.450       6.647      30.075        0.970x
```

</details>

### Comparison

Best speedup over sequential at 10,000,000 elements, and sequential throughput.

**The desktop columns are not comparable with the laptop's yet.** The laptop values are from the
current benchmark (separate sequential baseline, staggered arrays; range over its two runs, best
thread count). The desktop values are from the earlier benchmark (baseline = parallel kernel on 1
thread, no staggering; run 1, run 2 in brackets where it differed by more than 10 %). Replace the
desktop columns after rerunning.

| Op | Laptop speedup | Desktop speedup (old) | Laptop sequential GB/s | Desktop sequential GB/s (old) |
|---|---|---|---|---|
| add | 1.10-1.37x | 1.29x [1.38x] | 15-19 | 29.6 |
| mul | 1.09-1.21x | 1.49x [1.19x] | 16-17 | 30.2 |
| dot | 3.5-4.6x | 8.6x [7.1x] | about 5 | 8.9 |

Caveats for this comparison:

- Compiler: the desktop used MSVC (OpenMP 2.0) and the laptop GCC 16.1.0 (OpenMP 5.2). Part of
  the difference can come from the compiler rather than the hardware. Building the desktop with
  MinGW as well would separate the two.
- Power: neither machine ran on a High performance plan (laptop on battery with Balanced, desktop on
  "Dedera" with a 95 % processor cap). The desktop's sequential numbers would probably be higher
  with boost enabled.

## Matrix multiplication (laptop only so far)

`C = A * B` for square float32 matrices, `benchmarks/blas/matmul_benchmark.cpp`. Same machine,
compiler and conditions as Machine 1 above (on battery, Balanced plan, so expect noise).

- sequential: the plain i-k-j loop (`sequential.cpp`), no blocking, no threads.
- parallel: the cache-blocked kernel (`matmul.cpp`) with the rows of C split between threads. The
  1-thread row shows what blocking alone buys.

Run it like the other benchmarks: `matmul_benchmark > matmul_<machine>.txt`. GFLOP/s counts
2 n^3 operations.

### Results (one run, default build: baseline SSE2 instructions)

| n | Sequential | Blocked, 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| 64 | 10.3 GFLOP/s | 10.2 (0.99x) | 9.4 (0.92x) | 6.7 (0.65x) |
| 128 | 10.7 | 10.5 (0.98x) | 30.8 (2.9x) | 26.8 (2.5x) |
| 256 | 11.0 | 11.1 (1.0x) | 40.4 (3.7x) | 38.1 (3.5x) |
| 512 | 11.1 | 12.5 (1.1x) | 44.4 (4.0x) | 41.9 (3.8x) |
| 1024 | 8.7 | 10.3 (1.2x) | 30.9 (3.6x) | 37.3 (4.3x) |

What it shows:

- The threads give 3.5-4.3x on this 4-core CPU from n = 128 up. At n = 64 the matrices are too small
  to share (the kernel stays serial below 262,144 multiply-adds, but 8 threads are still slower).
- Cache blocking alone helps little (1.0-1.2x), because the plain i-k-j loop already streams
  through B row by row and the compiler vectorizes it.
- Absolute speed is low: 44 GFLOP/s at best. The inner loop does two loads and a store for every
  multiply-add, so it is limited by cache bandwidth, not arithmetic. A kernel that keeps a small
  tile of C in registers while it runs over the depth would be several times faster.
- Each C entry sums its products in the same order as the sequential loop, whatever the thread
  count, so results do not depend on it (and agree with the sequential ones to within rounding;
  tested in `tests/ops/matmul_test.cpp`).

### Same benchmark built with `-DMENSURFLOW_NATIVE=ON` (AVX2 and FMA)

| n | Sequential | Blocked, 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| 128 | 16.7 GFLOP/s | 15.4 (0.92x) | 20.5 (1.2x) | 17.7 (1.1x) |
| 256 | 13.4 | 9.9 (0.74x) | 33.9 (2.5x) | 25.0 (1.9x) |
| 512 | 14.2 | 18.4 (1.3x) | 45.0 (3.2x) | 58.3 (4.1x) |
| 1024 | 16.0 | 12.8 (0.80x) | 33.2 (2.1x) | 43.7 (2.7x) |

The wider instructions raise the sequential loop from about 11 to about 15 GFLOP/s and the best
parallel result from 44 to 58 GFLOP/s, much less than the instructions could give. The same limit
applies: the loop is bound by loads and stores. All 13 tests pass in this build too (fused
multiply-adds change the last bits, so the matmul tests compare against the float64 reference
within a tolerance, not bit for bit). These numbers vary a lot between runs on this laptop.

<details>
<summary>Full output: matmul_benchmark, default build</summary>

```
Matrix multiplication C = A * B, square float32 matrices
compiler: gcc 16.1.0, OpenMP: yes, max threads: 8, hardware threads: 8
sequential = plain i-k-j loop. parallel = blocked kernel with N threads.
Speedup = sequential time / this time.

       n        impl  threads          ms    GFLOP/s    speedup
      64  sequential        -       0.051     10.292     1.000x
      64    parallel        1       0.052     10.177     0.989x
      64    parallel        2       0.057      9.176     0.892x
      64    parallel        4       0.056      9.412     0.915x
      64    parallel        8       0.078      6.715     0.653x

     128  sequential        -       0.390     10.748     1.000x
     128    parallel        1       0.398     10.528     0.979x
     128    parallel        2       0.220     19.024     1.770x
     128    parallel        4       0.136     30.787     2.864x
     128    parallel        8       0.157     26.762     2.490x

     256  sequential        -       3.052     10.995     1.000x
     256    parallel        1       3.025     11.093     1.009x
     256    parallel        2       1.529     21.949     1.996x
     256    parallel        4       0.831     40.364     3.671x
     256    parallel        8       0.881     38.072     3.463x

     512  sequential        -      24.131     11.124     1.000x
     512    parallel        1      21.448     12.515     1.125x
     512    parallel        2      10.387     25.844     2.323x
     512    parallel        4       6.052     44.356     3.987x
     512    parallel        8       6.402     41.930     3.769x

    1024  sequential        -     246.591      8.709     1.000x
    1024    parallel        1     207.691     10.340     1.187x
    1024    parallel        2     109.511     19.610     2.252x
    1024    parallel        4      69.413     30.938     3.553x
    1024    parallel        8      57.590     37.289     4.282x

```

</details>

<details>
<summary>Full output: matmul_benchmark, MENSURFLOW_NATIVE=ON build</summary>

```
Matrix multiplication C = A * B, square float32 matrices
compiler: gcc 16.1.0, OpenMP: yes, max threads: 8, hardware threads: 8
sequential = plain i-k-j loop. parallel = blocked kernel with N threads.
Speedup = sequential time / this time.

       n        impl  threads          ms    GFLOP/s    speedup
      64  sequential        -       0.044     11.943     1.000x
      64    parallel        1       0.037     14.263     1.194x
      64    parallel        2       0.067      7.796     0.653x
      64    parallel        4       0.083      6.349     0.532x
      64    parallel        8       0.116      4.535     0.380x

     128  sequential        -       0.251     16.733     1.000x
     128    parallel        1       0.272     15.400     0.920x
     128    parallel        2       0.217     19.337     1.156x
     128    parallel        4       0.204     20.526     1.227x
     128    parallel        8       0.237     17.717     1.059x

     256  sequential        -       2.497     13.437     1.000x
     256    parallel        1       3.397      9.878     0.735x
     256    parallel        2       1.900     17.663     1.314x
     256    parallel        4       0.990     33.908     2.523x
     256    parallel        8       1.343     24.978     1.859x

     512  sequential        -      18.847     14.243     1.000x
     512    parallel        1      14.609     18.375     1.290x
     512    parallel        2       6.430     41.750     2.931x
     512    parallel        4       5.961     45.033     3.162x
     512    parallel        8       4.603     58.319     4.095x

    1024  sequential        -     134.006     16.025     1.000x
    1024    parallel        1     168.099     12.775     0.797x
    1024    parallel        2     110.223     19.483     1.216x
    1024    parallel        4      64.616     33.235     2.074x
    1024    parallel        8      49.178     43.668     2.725x

```

</details>

Not measured yet: the Ryzen 9 5900X desktop (rerun `matmul_benchmark` there after pulling).
