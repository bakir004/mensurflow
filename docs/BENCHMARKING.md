# How we write benchmarks

Every benchmark in this repository uses the same harness (`benchmarks/bench.hpp`), measures the same
way, and prints the same table, so numbers written by different people on different machines can be
put next to each other. This page is the rulebook. Results live in [BENCHMARKS.md](BENCHMARKS.md).

## The three implementations we compare

For every CPU kernel we compare up to three versions of the same computation:

| impl | What it is | Where it comes from |
|---|---|---|
| `scalar` | the plain loop, **compiled without auto-vectorization** (one element at a time) | `src/backends/cpu/scalar.cpp` |
| `sequential` | the same loop with whatever SIMD the compiler finds, no threads | `src/backends/cpu/sequential.cpp` |
| `parallel` | the parallel kernel (`parallel_for` / `parallel_sum`) with N threads | `elementwise.cpp`, `blas.cpp`, `matmul.cpp`, ... |

Keeping the three apart answers two different questions: how much SIMD helps (`scalar` to
`sequential`) and how much threads help (`sequential` to `parallel`). The table has a speedup column for each:
`vs scalar` and `vs seq` (the baseline's median time divided by this row's, so above 1x is faster).

`scalar.cpp` is not separate code to maintain. It includes `sequential.cpp` a second time, in the
namespace `scalar`, with per-file compiler flags that turn auto-vectorization off (`-fno-tree-vectorize`
on GCC, `-fno-vectorize` on Clang, `/Qvec-` on MSVC; see `CMakeLists.txt`; the MSVC flag has not been
tried yet). Results are identical to the sequential ones, and `tests/ops/scalar_test.cpp` checks that.

## Which benchmark answers what

| Executable | Question | Source |
|---|---|---|
| `cpu_parallel_benchmark` | scalar vs sequential vs parallel for every elementwise op and `dot`, thread sweep | `benchmarks/cpu_parallel_benchmark.cpp` |
| `reduction_benchmark` | sequential vs parallel `sum`, `mean`, and `max`, across configured OpenMP thread counts | `benchmarks/blas/reduction_benchmark.cpp` |
| `matmul_benchmark` | the same for matrix multiplication, in GFLOP/s | `benchmarks/blas/matmul_benchmark.cpp` |
| `matvec_benchmark` | sequential vs parallel matrix-vector multiplication across thread counts | `benchmarks/blas/matvec_benchmark.cpp` |
| `softmax_benchmark` | sequential vs parallel last-axis softmax across thread counts | `benchmarks/blas/softmax_benchmark.cpp` |
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
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/scalar.hpp"
#include "backends/cpu/sequential.hpp"

namespace cpu = mensurflow::cpu;

int main() {
    const bench::Config config;                       // 1 warm-up, median of 9 samples
    bench::print_header("add on the CPU (float32)", config);
    bench::Table table("GB/s");                       // the unit of the rate column
    table.section("add");

    for (std::int64_t n : {1000, 100000, 10000000}) {
        const auto len = static_cast<std::size_t>(n);
        bench::Arrays arrays(len, 3);                 // 3 arrays, staggered (see rules)
        float *a = arrays[0], *b = arrays[1], *out = arrays[2];
        bench::fill_a(a, len);
        bench::fill_b(b, len);
        const double bytes = 12.0 * n;                // 2 reads + 1 write of 4 bytes

        auto run_seq = [&] { cpu::sequential::add_f32(a, b, out, n); };
        auto observe = [&] { return bench::checksum(out, len); };
        run_seq();                                     // reference result, to check the others against
        std::vector<float> expected(out, out + len);

        auto run_scalar = [&] { cpu::scalar::add_f32(a, b, out, n); };
        run_scalar();
        bench::check_bits(expected.data(), out, len, "scalar add");      // BEFORE timing
        const auto scalar = bench::measure(config, run_scalar, observe);
        const auto seq = bench::measure(config, run_seq, observe);
        table.row(std::to_string(n), "scalar", "-", scalar, bytes, &scalar, &seq);
        table.row(std::to_string(n), "sequential", "-", seq, bytes, &scalar, &seq);
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
   bandwidth. Include sizes around the parallel cutoff (32,768 elements) and one that is not a
   multiple of 16 or of the SIMD width.

## Reading the table

```
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
   1000000       scalar        -      1.2969      1.2664      1.3313      6.169      1.00x      0.99x         -
   1000000   sequential        -      1.2786      1.2200      1.3931      6.257      1.01x      1.00x         -
   1000000     parallel        8      0.2834      0.2769      0.2996     28.232      4.58x      4.51x         -
```

- `median ms`, `min ms`, `max ms`: wall-clock time per call over the samples.
- rate (`GB/s` here): work divided by the median time.
- `vs scalar`, `vs seq`: that baseline's median time divided by this row's. 4.58x means 4.58 times faster.
- `cpu/wall`: process CPU time over wall time across all samples. About 1 for one busy thread, near N
  for N busy threads, lower if threads sit idle. It is `-` when the run was too short for the OS CPU
  clock to give a meaningful ratio (Windows advances it in ticks of about 15 ms; the harness needs
  0.1 s of timed work).

## Adding a kernel to the comparison

1. Write the sequential kernel in `src/backends/cpu/sequential.{hpp,cpp}` as usual. It automatically
   gets a scalar twin in namespace `scalar`.
2. Copy its declaration into `src/backends/cpu/scalar.hpp`.
3. Add the parallel kernel, tests (`tests/ops/`), and a row for it in `cpu_parallel_benchmark.cpp`
   (an entry in the `ops` list; unary and `axpy`-like signatures have adapter templates there).
4. Run it twice, plugged in, and add the table to [BENCHMARKS.md](BENCHMARKS.md) with your machine.

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

- An AVX2 variant (intrinsics in their own file, compiled with `-mavx2 -mfma` for that file only,
  called only when `bench::avx2_supported()`) and an `omp simd` variant: the harness is ready for
  them; no kernel has one yet.
- The scalar baseline does not work on MSVC. Tried with MSVC 19.51 (Visual Studio Build Tools
  2026): `cl` ignores `/Qvec-` (`warning D9002: ignoring unknown option '/Qvec-'`), so `scalar.cpp`
  is auto-vectorized and its rows match `sequential` (see Machine 2 in BENCHMARKS.md). MSVC needs
  another way to turn auto-vectorization off for that file.
- Results in BENCHMARKS.md for Machine 1, Machine 3 and the laptop's matrix multiplication were
  taken before the scalar baseline and the min/max/cpu-wall columns existed; they show sequential
  vs parallel only. Machine 2 uses the new format.
