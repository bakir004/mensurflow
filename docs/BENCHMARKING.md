# How we write benchmarks

Every benchmark in this repository uses the same harness (`benchmarks/bench.hpp`), measures the same
way, and prints the same table, so numbers written by different people on different machines can be
put next to each other. This page is the rulebook. Results live in [BENCHMARKS.md](BENCHMARKS.md).

## One thread vs many

CPU kernels live in `src/backends/cpu/ops/<group>/` (`blas`, `elementwise`, `reduction`). Almost
every kernel is a single loop marked with `MENSURFLOW_PARALLEL_FOR` (`src/backends/cpu/threads.hpp`):
in an OpenMP build it becomes `#pragma omp parallel for`, otherwise nothing. So for these kernels
the benchmarks compare the same function on different thread counts:

| impl | What it is |
|---|---|
| `sequential` | the kernel on one thread (`cpu::set_max_threads(1)`) |
| `parallel` | the same kernel on N threads |

`matmul`, `max` and `min` are the exceptions: their parallel algorithm differs from the plain loop,
so they have two functions, `<op>_f32_sequential` and `<op>_f32_parallel`, and the benchmark times
both.

The table's `vs seq` column is the sequential median time divided by this row's, so above 1x means
the threads made it faster.

## Which benchmark answers what

| Executable | Question | Source |
|---|---|---|
| `cpu_parallel_benchmark` | sequential vs parallel arithmetic/activations and `dot`, thread sweep | `benchmarks/cpu_parallel_benchmark.cpp` |
| `activation_torch_benchmark` (optional) | direct CPU activations vs preallocated C++ ATen `out` calls at matching thread counts | `benchmarks/activation_torch_benchmark.cpp` |
| `activation_movement_benchmark` | public activation-backward and tensor-movement APIs across CPU thread counts | `benchmarks/activation_movement_benchmark.cpp` |
| `random_indexing_benchmark` | seeded RNG, dropout and indexing forward/backward public APIs; optional native ATen comparison | `benchmarks/random_indexing_benchmark.cpp` |
| `selection_benchmark` | F32 comparisons and `where`, one/many CPU threads; optional native ATen reference | `benchmarks/selection_benchmark.cpp` |
| `cast_benchmark` | all 16 F32/I32/F16/BF16 pairs, one/many CPU threads; optional native ATen `copy_` reference | `benchmarks/cast_benchmark.cpp` |
| `reduction_benchmark` | sequential vs parallel `sum`, `mean`, and `max`, across configured OpenMP thread counts | `benchmarks/blas/reduction_benchmark.cpp` |
| `matmul_benchmark` | the same for matrix multiplication, in GFLOP/s | `benchmarks/blas/matmul_benchmark.cpp` |
| `matvec_benchmark` | sequential vs parallel matrix-vector multiplication across thread counts | `benchmarks/blas/matvec_benchmark.cpp` |
| `argmax_benchmark` | sequential vs parallel last-axis argmax across thread counts | `benchmarks/blas/argmax_benchmark.cpp` |
| `dot_benchmark` | `dot` on every device, the cost of one API call, API vs direct kernel | `benchmarks/blas/dot_benchmark.cpp` |
| `elementwise_benchmark` | `add` and `mul` on every device through the public API | `benchmarks/elementwise_benchmark.cpp` |

`cpu_parallel_benchmark` takes operation names to run only some: `cpu_parallel_benchmark relu exp`.

## Writing a benchmark

A benchmark is one `main()`: print the header, then for each size and implementation check, measure,
and print a row. This is a complete small one (the real ones are `cpu_parallel_benchmark.cpp` and
`matmul_benchmark.cpp`):

```cpp
#include "bench.hpp"
#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/threads.hpp"

namespace cpu = mensurflow::cpu;

int main() {
    const bench::Config config;                       // 1 warm-up, median of 9 samples
    const int threads = cpu::max_threads();
    bench::print_header("add on the CPU (float32)", config);
    bench::Table table("GB/s", 1e9, "vs seq", "");    // rate unit, scale, speedup column
    table.section("add");

    for (std::int64_t n : {1000, 100000, 10000000}) {
        const auto len = static_cast<std::size_t>(n);
        bench::Arrays arrays(len, 4);                 // 4 arrays, staggered (see rules)
        float *a = arrays[0], *b = arrays[1], *out = arrays[2], *expected = arrays[3];
        bench::fill_a(a, len);
        bench::fill_b(b, len);
        const double bytes = 12.0 * n;                // 2 reads + 1 write of 4 bytes

        auto run_seq = [&] { cpu::add_f32(a, b, expected, n); };
        auto run_par = [&] { cpu::add_f32(a, b, out, n); };
        cpu::set_max_threads(1);
        run_seq();                                    // reference result, on one thread
        const auto seq = bench::measure(config, run_seq, [&] { return bench::checksum(expected, len); });
        cpu::set_max_threads(threads);
        run_par();
        bench::check_bits(expected, out, len, "parallel add");          // BEFORE timing
        const auto par = bench::measure(config, run_par, [&] { return bench::checksum(out, len); });
        table.row(std::to_string(n), "sequential", "-", seq, bytes, &seq);
        table.row(std::to_string(n), "parallel", std::to_string(threads), par, bytes, &seq);
        table.blank();
    }
}
```

Register it in `CMakeLists.txt` with `mensurflow_add_benchmark(name path)` next to the others. That
links the harness and gives it the include paths for `src/` and `tests/reference/`.

### What the harness does

| Function | Purpose |
|---|---|
| `print_header(title, config, notes)` | CPU model, threads, compiler, OpenMP, AVX2, Release or not, timing settings |
| `measure(config, run, observe)` | warm up, calibrate calls per sample, time only `run()`, return a `Sample` (median, min, max, every run, CPU/wall) |
| `consume(x)`, `checksum(p, n)` | noinline and atomic, so the compiler cannot delete work whose result is unread |
| `check_bits`, `check_close`, `check_value` | throw if the result is wrong, naming the first bad index |
| `Arrays(n, count)` | arrays that start 1 KB apart modulo 4 KB |
| `fill_a`, `fill_b` | the same reproducible inputs in every benchmark |
| `thread_counts(max)` | 1, powers of two, max/2 (physical cores), max |
| `Table` | the standard table: `section`, `row`, `blank` |
| `avx2_supported()` | runtime check an AVX2 variant needs before it is called |

## Rules

1. **Time only the call.** `observe()` (the checksum, or the returned value) runs after the timer
   stops. Never time a correctness check.
2. **Check before timing, for every implementation and size.** A fast wrong answer is worse than no
   benchmark. Pick the check by kind of kernel:

   | Kernel | Check |
   |---|---|
   | elementwise, same arithmetic (`add`, `mul`, `relu`, ...) | `check_bits` (exact) |
   | `exp`, `sqrt`, `sigmoid` against another implementation | `check_close`, `rel_tol` about 1e-4 |
   | reductions (`dot`, `sum`) | `check_value`, `rel_tol` about 1e-4; pass `scale` = sum of absolute terms if the sum can cancel |
   | matmul | `check_close`, `rel_tol` 1e-4 (rounding can differ between compilers) |
   | anything on a GPU | `check_close`, never `check_bits`; vendor libraries round differently |

3. **Warm up, and report the median plus min and max.** The median ignores a stray slow sample; min
   and max show how noisy the run was. A row where max is several times the median means the machine
   was busy.
4. **Allocate arrays with `bench::Arrays`.** Equally sized arrays tend to start at the same offset
   within a page, and addresses that agree in the low 12 bits make loads and stores falsely depend on
   each other ("4K aliasing"). It once made `mul` look 2x slower than `add`.
5. **If the kernel changes its own input** (an accumulating kernel like `axpy` into its own `y`),
   reset the input inside `run()`, or the data drifts between calls.
6. **Pick a unit that lets variants be compared**: GB/s for memory-bound ops (count the bytes the
   op must touch), GFLOP/s for compute-bound ops (`2*n^3` for matmul), `Melems/s` for small per-element
   work. Say in the header notes how you counted. `Table(unit, scale)`: scale 1e9 gives G, 1e6 gives M.
7. **Build in Release.** The header prints a warning if `NDEBUG` is not defined. Do not turn on
   `MENSURFLOW_NATIVE` for numbers you report: it makes the build specific to your CPU.
8. **Choose sizes on purpose.** 2^17 floats is 512 KB, about L2; much larger sizes measure memory
   bandwidth. Include sizes around the operation's parallel cutoff (initially 32,768 elements)
   and one that is not a multiple of 16 or of the SIMD width.

## Reading the table

```
         n         impl  threads   median ms      min ms      max ms       GB/s     vs seq  cpu/wall
   1000000   sequential        -      1.2786      1.2200      1.3931      6.257      1.00x         -
   1000000     parallel        8      0.2834      0.2769      0.2996     28.232      4.51x         -
```

- `median ms`, `min ms`, `max ms`: wall-clock time per call over the samples.
- rate (`GB/s` here): work divided by the median time.
- `vs seq`: the sequential median time divided by this row's. 4.51x means 4.51 times faster.
- `cpu/wall`: process CPU time over wall time across all samples. About 1 for one busy thread, near N
  for N busy threads, lower if threads sit idle. It is `-` when the run was too short for the OS CPU
  clock to give a meaningful ratio (Windows advances it in ticks of about 15 ms; the harness needs
  0.1 s of timed work).

## Adding a kernel to the comparison

1. Add `<op>_f32` to the group's file in `src/backends/cpu/ops/<group>/` (declaration in the
   `.hpp`, the loop in the `.cpp`, with `MENSURFLOW_PARALLEL_FOR(if (...))` on the line before it),
   and an adapter plus registration in `src/backends/cpu/backend.cpp`. Only if the parallel
   algorithm has to differ from the plain loop, write `<op>_f32_sequential` and `<op>_f32_parallel`
   instead, as `matmul`, `max` and `min` do.
2. Add tests (`tests/ops/`), and a row for it in `cpu_parallel_benchmark.cpp` (an entry in the
   `ops` list; unary and `axpy`-like signatures have adapter templates there).
3. Run it twice, plugged in, and add the table to [BENCHMARKS.md](BENCHMARKS.md) with your machine.

## Comparing against OpenBLAS

`matmul_openblas_benchmark` runs our sequential and parallel matmul next to `cblas_sgemm` at the same
thread counts and prints a `vs obl` column (OpenBLAS median time / this row's, so 1.00 = as fast as
OpenBLAS). It is only built when OpenBLAS is found.

1. Get OpenBLAS for the same architecture as the build (x64): the prebuilt `OpenBLAS-<version>-x64.zip`
   from the OpenBLAS GitHub releases, or `vcpkg install openblas:x64-windows`.
2. Configure with `-DOPENBLAS_ROOT=<folder with include/ lib/ bin/>` (or the vcpkg toolchain file). Use
   `-DMENSURFLOW_OPENBLAS=ON` to fail when it is not found. `libopenblas.dll` is copied next to the
   executable.
3. Build and run `matmul_openblas_benchmark` in Release, then follow the rules below as usual.

The header line prints the OpenBLAS build and the core type it selected; report both with the numbers.

## Comparing against MKL and AOCL-BLIS

`gemm_shapes_openblas_benchmark` (the products a neural network makes) also times Intel MKL and AMD's
AOCL-BLIS when they are installed. All three libraries export the same BLAS names, so only OpenBLAS is
linked; the other two are opened at run time from their DLLs (`benchmarks/blas/reference_blas.hpp`). A
library that is not found is left out of the table, and nothing else in the build depends on them.

1. **MKL**: the NuGet packages `intelmkl.redist.win-x64` and `intelopenmp.redist.win` (a `.nupkg` is a
   zip). Put `mkl_rt.3.dll`, `mkl_core.3.dll`, `mkl_intel_thread.3.dll`, `mkl_def.3.dll`,
   `mkl_avx2.3.dll`, `mkl_mc3.3.dll`, `mkl_avx512.3.dll` and `libiomp5md.dll` in one folder `<mkl>/bin`.
   On Linux, any install with `libmkl_rt.so`.
2. **AOCL-BLIS**: the conda-forge package `aocl-blas` (win-64; a `.conda` file is a zip of `.tar.zst`
   archives, which the Windows `tar` unpacks) and `libomp.dll` from the conda-forge package
   `llvm-openmp`. Put `AOCL-LibBlis-Win-MT-dll.dll` and `libomp.dll` in one folder `<aocl>/bin`. AMD's
   own installer has the same library. On Linux, `libblis-mt.so`.
3. Configure with `-DMKL_ROOT=<mkl> -DAOCL_BLIS_ROOT=<aocl>`; CMake prints which ones it found. The
   environment variables `MENSURFLOW_MKL_DLL` and `MENSURFLOW_AOCL_BLIS_DLL` (full path of the DLL)
   override the configured paths at run time.

What to know when reading the numbers:

- The header prints each library's version and code path. On an AMD processor MKL loads `mkl_def`,
  which has its own AVX2 kernel for Zen; that is what MKL gives an AMD user by default.
- MKL and AOCL-BLIS each bring their own OpenMP runtime, and their idle workers spin for 200 ms after a
  call. The benchmark pauses 300 ms before each library (`settle=`); do not lower it below 200.
- By default no library's threads are bound to cores except ours. Run a second time with
  `OMP_PLACES=cores OMP_PROC_BIND=close` in the environment, which binds the threads of MKL and
  AOCL-BLIS, and report both: binding is what a careful user of those libraries would set.
- `threads=1,2,4,6,8,12,24` gives every library the chance to show its best thread count; the summary
  compares each library at its own best (`*` columns).
- `mkl=0` or `aocl=0` leaves a library out, to check that the others measure the same without it.

## Comparing activations against PyTorch

`activation_torch_benchmark` is OFF by default; enable it with
`-DMENSURFLOW_TORCH_BENCHMARK=ON`. Point CMake at an installed LibTorch configuration
(a CPU PyTorch wheel supplies one) and use a separate build directory:

```bash
cmake -S . -B build-torch-bench -DCMAKE_BUILD_TYPE=Release \
  -DMENSURFLOW_CUDA=OFF -DMENSURFLOW_METAL=OFF -DMENSURFLOW_NATIVE=OFF \
  -DMENSURFLOW_OPENMP=ON -DMENSURFLOW_OPENBLAS=OFF \
  -DMENSURFLOW_TORCH_BENCHMARK=ON -DCMAKE_PREFIX_PATH=/path/to/torch/share/cmake
cmake --build build-torch-bench -j 4
ctest --test-dir build-torch-bench --output-on-failure
mkdir -p build-torch-bench/results
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-torch-bench/activation_torch_benchmark --sizes=1000,32769,1000000 \
  --threads=1,2,6,12 > build-torch-bench/results/activations-run1.txt
# Cool down, then repeat with --reverse and save as activations-run2.txt.
```

For a wheel, obtain the prefix with `python -c 'import torch; print(torch.utils.cmake_prefix_path)'`
using the Python environment where Torch is installed.

Both libraries use the same preallocated F32 input/output addresses and benchmark harness. The timed
calls are direct MensurFlow CPU kernels and native C++ ATen `out` operators, including ATen dispatch;
allocation, Tensor wrapping and Python are excluded. `vs torch` is ATen median / MensurFlow median,
so above one favors MensurFlow. After two cooled runs with matching operations, sizes and thread
counts, create the paired Markdown table:

```bash
python3 benchmarks/reference/compare_native_activations.py \
  build-torch-bench/results/activations-run1.txt \
  build-torch-bench/results/activations-run2.txt \
  > build-torch-bench/results/activations-paired.md
```

The parser needs only Python's standard library. It validates matching rows and shows both median
times and ratios. Its drift column takes the larger of the two libraries' `abs(a-b)/min(a,b)*100`
values; drift values above 10% are bold and the affected rows are counted in the summary.

The cast benchmark uses buffers of the actual storage type, with input/output/reference page
offsets 0/1024/2048 modulo 4 KB, matching the shared float-buffer spacing strategy.

The same build option enables native references in `selection_benchmark` and `cast_benchmark`.
Their logs use the same paired format and parser. Selection rates count 9 logical bytes per
comparison and at most 13 for `where`; casts count source plus destination bytes. These are not
physical DRAM bandwidth measurements. Selection uses deterministic finite inputs; cast floating
inputs use quarter increments and I32 inputs use integers from -2046 to 2046. Cast names use
`<source>_to_<destination>` (for example, `f32_to_bf16`); omit names to run all 16 pairs.

## Comparing random generation and indexing against PyTorch

`random_indexing_benchmark` is always built. Without LibTorch it compares the public APIs on one
and many CPU threads. `MENSURFLOW_TORCH_BENCHMARK=ON` enables native allocating ATen references in
the same executable, using the existing Torch CMake setup above. Both sides time allocation,
validation and execution; tensor construction, correctness checks and host readback are outside
the timer. Autograd recording is disabled. This scope differs from the preallocated activation
benchmark and should not be compared with its raw times.

```bash
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-torch-bench/random_indexing_benchmark --sizes=1000,32769,8388608 \
  --threads=1,2,6,12 > build-torch-bench/results/random-indexing-run1.txt
# Cool the machine before the second run.
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./build-torch-bench/random_indexing_benchmark --sizes=1000,32769,8388608 \
  --threads=1,2,6,12 --reverse > build-torch-bench/results/random-indexing-run2.txt
python3 benchmarks/reference/random_indexing.py \
  build-torch-bench/results/random-indexing-run1.txt \
  build-torch-bench/results/random-indexing-run2.txt
```

Select individual operations by name: `uniform normal bernoulli dropout dropout_backward one_hot
embedding embedding_backward`. Sizes are output-element targets; one-hot rounds up to a multiple
of 16 classes, and embedding rounds up to 32 features. Embedding backward reports its upstream
shape and always produces the 512×32 table gradient. The large default exceeds a 16 MiB L3 cache.

MensurFlow uses Philox, while PyTorch's CPU RNG uses a different stream. Bounds and distribution
moments are checked independently rather than requiring equal samples. Dropout forward checks
each implementation's own mask and scaling; backward uses the same predetermined Bool mask.
One-hot and both embedding directions use shared deterministic inputs and independent references,
including repeated indices. Every measured library is preceded by a 500 ms worker-settling pause;
all rows use nine samples. The reporting script retains paired medians/minimums and flags drift
above 10%; raw logs also retain maximums and throughput.

## Elementwise parallel cutoffs

`cpu_parallel_benchmark`, `selection_benchmark`, `cast_benchmark` and optional
`activation_torch_benchmark` accept `--cutoff=n` for selected MensurFlow elementwise operations;
`dot` and ATen scheduling are unaffected. Headers print effective values. Overrides are process-local
and do not persist: omit the option for defaults, use `--cutoff=0` to force the parallel branch for
nonempty inputs, or `--cutoff=9223372036854775807` to force serial execution.

Compare forced serial and parallel medians at matching sizes and requested thread counts, including
sizes around 32,768. Repeat cooled sessions in reversed order, refine the crossover interval, and
check intermediate, physical-core and hardware-thread counts. Retain noisy rows and repeat if drift
exceeds 10%. Change a default only from stable evidence; check sizes below/at/above it and rerun the
suite with and without OpenMP. Current defaults remain untuned.

## Before you report numbers

- Release build, no `MENSURFLOW_NATIVE`, CMake from a clean build folder.
- Plugged in, High performance power plan, other heavy programs closed.
- **Let the machine cool down between runs.** On the laptop (a 15 W CPU), a run started right
  after another was slower in almost every row, including single-thread ones, because the CPU slows
  down once it has been loaded for a few minutes. Wait about 3 minutes before each run.
- Run twice; if the two runs differ by more than about 10 % in a row, the machine was busy.
  Expect the **2- and 4-thread rows to vary more** (up to 2x on the laptop even after cooling):
  unpinned threads can land on two hyperthreads of one core or on separate cores. The 1-thread and
  all-threads rows were stable. To compare thread counts more reliably, pin the threads
  (`OMP_PLACES=cores`, `OMP_PROC_BIND=close`); this reduced run-to-run drift in a quick test but has
  not been checked on other machines, so say in the results if you used it.
- The header block is part of the result: keep it when you paste the output.
- Write the commit hash, compiler and version next to the table.
- Exact build and run commands for each toolchain, and the result form: [benchmarks/README.md](benchmarks/README.md).
- On Windows with MinGW, use the **UCRT64** toolchain (`C:\msys64\ucrt64`), not the older `mingw64`
  one: its `fmaf` is not fused, which breaks the `axpy` tests.

## Not done yet

- Activation kernels use ordinary C++/OpenMP loops; no current activation benchmark establishes
  explicit SIMD gain.
- Per-operation elementwise cutoff tuning and final measurements are pending; see
  [the measurement guidance above](#elementwise-parallel-cutoffs).
- Results in BENCHMARKS.md were taken with the earlier kernel layout (`parallel_for` helpers, and a
  `scalar` no-SIMD baseline that has since been removed); its file paths refer to that layout.
