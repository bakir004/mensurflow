# CPU benchmarks: sequential vs parallel

Results from the float32 CPU kernels (`add`, `sub`, `mul`, `div`, `neg`, `abs`,
`square`, `relu`, `sqrt`, `exp`, `sigmoid`, `axpy`, `bias_add`, `dot`), measured on each machine we have
so the numbers can be compared. Keep the build and run conditions below the same on every machine,
otherwise the comparison is not fair.

**How the benchmarks are written, and how to add one: [BENCHMARKING.md](BENCHMARKING.md).** All of
them share one harness (`benchmarks/bench.hpp`), so the tables look the same whoever ran them.

Code:
- parallel: `src/backends/cpu/parallel.hpp` (`parallel_for`, `parallel_sum`),
  `src/backends/cpu/elementwise.cpp`, `src/backends/cpu/blas.cpp`, `src/backends/cpu/matmul.cpp`.
- sequential: `src/backends/cpu/sequential.cpp`, plain loops with no OpenMP and no threads.
- scalar: the same loops compiled without auto-vectorization (`src/backends/cpu/scalar.cpp`), the
  baseline that separates what SIMD adds from what threads add.

Benchmarks (`cpu_parallel_benchmark`, `matmul_benchmark`, `dot_benchmark`, `elementwise_benchmark`):
the first two compare scalar, sequential and parallel on the CPU; the last two compare devices
through the public API. Which one answers what is listed in BENCHMARKING.md.

> **Format change.** Newer runs print a `scalar` row, `min` and `max` columns, a second speedup
> column (`vs scalar` next to `vs seq`) and a `cpu/wall` column. Results below that were measured
> with the earlier format show sequential vs parallel only, with the sequential row as the
> baseline; they are still valid for that comparison but have no scalar numbers. Machine 2
> (desktop) is measured with the new format; Machine 1, Machine 3 and the laptop's matrix
> multiplication tables still use the earlier one.
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
matmul_benchmark > matmul_<machine>.txt
dot_benchmark > dot_<machine>.txt
```

Each size prints a `scalar` row (no SIMD), a `sequential` row (the same loop with the compiler's
SIMD) and `parallel` rows for 1, 2, 4, 8, 16, half the hardware threads (the physical core count on
SMT CPUs) and all hardware threads. The two speedup columns are the scalar or sequential median
divided by that row's, so above 1x is faster. The `parallel` row with 1 thread shows what the
parallel machinery costs when there is nothing to gain. On Windows with MinGW use the UCRT64
toolchain. **The exact commands for every toolchain, the quick start, and where to put your result
are in [benchmarks/README.md](benchmarks/README.md).**

Conditions to keep equal and to write down for every run:

- Plugged in, power plan set to High performance (or equivalent).
- Other programs closed. Run twice and compare: if two runs differ by more than ~10 %, the machine
  was busy and the numbers are noise.
- Compiler and version (GCC, MSVC and Clang can produce different speeds for the same code).
- Commit hash of the code.

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
| Power state | Older results below: **on battery, "Balanced" plan**, with Chrome running in the background (battery 65 %, CPU idling at 801 MHz), so the CPU may have run below its full speed. Newest results (2026-11-01): on AC, still "Balanced"; see their conditions |

### Results with the shared harness (2026-11-01): scalar, sequential, parallel

The first results measured with the shared harness ([BENCHMARKING.md](BENCHMARKING.md)), which adds the
`scalar` baseline (auto-vectorization off) next to `sequential` (the compiler's SIMD) and `parallel`.
The older tables further down have no scalar rows.

Conditions of these runs:

- Code: commit `dcdc3c7` plus a later fix that prints more digits for very fast calls (output only).
  `cpu_parallel_benchmark`, all 14 ops, 15 sizes, run twice; `matmul_benchmark` once.
- Toolchain: **MSYS2 UCRT64 GCC 16.1.0** (the older tables below came from the `mingw64` one),
  OpenMP 5.2, `-O3`, no `-march=native`. The CPU has AVX2 but the build only uses baseline SSE2.
- Power: **on AC** (battery charging), Windows plan reported as "Balanced" (the Best performance
  mode overlay could not be checked from the command line), CPU load about 5 % before the runs.
- Chrome (12 processes), Edge and Avast were running but idle.
- Each run started after a **3-minute pause** (11:37, 11:43, matmul 11:48).
- Cells below read `run 1 / run 2`. Speedup = sequential time / parallel time.

#### How stable the runs were

Share of rows (size x implementation) whose median differs by more than 10 % between the two runs:

| impl | back to back, no pause (earlier pair) | with a 3-minute pause (these runs) |
|---|---|---|
| scalar | 81 of 200 (run 2 slower in 79) | 37 of 199 |
| sequential | 96 of 200 (run 2 slower in 92) | 26 of 199 |
| parallel, 1 thread | 98 of 200 | 28 of 198 |
| parallel, 2 threads | 107 of 200 | **134 of 199** |
| parallel, 4 threads | 126 of 200 | **126 of 199** |
| parallel, 8 threads | 120 of 200 | 39 of 200 |

- **Heat.** Started right after another run, almost every row was slower: the whole CPU slows down
  once it has been loaded for a few minutes (a 15 W laptop part). The pause removes most of that.
- **Thread placement.** Even with the pause, the **2- and 4-thread rows** differ by up to 2x between
  runs, with run 2 faster in about 120 of the 130. The 1-thread and 8-thread rows do not. A likely
  cause: Windows puts 2 or 4 unpinned threads sometimes on two hyperthreads of one core and sometimes
  on separate cores; with 1 thread, or all 8, there is no choice. This was not confirmed.
  Trust the 1-thread and 8-thread rows; treat 2 and 4 threads as a range.

#### SIMD: scalar vs sequential (n = 1,000,000)

| op | scalar ms | sequential ms | SIMD gain |
|---|---|---|---|
| add | 0.6160 / 0.5920 | 0.4282 / 0.4183 | 1.4x / 1.4x |
| sub | 0.5827 / 0.6026 | 0.4430 / 0.4508 | 1.3x / 1.3x |
| mul | 0.5988 / 0.6053 | 0.4096 / 0.4279 | 1.5x / 1.4x |
| div | 0.8664 / 0.9279 | 0.4125 / 0.4182 | 2.1x / 2.2x |
| neg | 0.4817 / 0.5099 | 0.2152 / 0.2400 | 2.2x / 2.1x |
| abs | 0.4836 / 0.5281 | 0.2130 / 0.2351 | 2.3x / 2.2x |
| square | 0.4505 / 0.4652 | 0.2130 / 0.2290 | 2.1x / 2.0x |
| relu | 0.6031 / 0.6538 | 0.2384 / 0.2383 | 2.5x / 2.7x |
| sqrt | 0.8940 / 0.9140 | 0.8761 / 0.9145 | 1.0x / 1.0x |
| exp | 4.2683 / 4.7687 | 4.5447 / 4.6566 | 0.9x / 1.0x |
| sigmoid | 5.1545 / 5.1511 | 4.7137 / 5.1112 | 1.1x / 1.0x |
| axpy | 2.1016 / 2.1866 | 1.8800 / 1.9339 | 1.1x / 1.1x |
| bias_add | 0.5760 / 0.5796 | 0.2669 / 0.2537 | 2.2x / 2.3x |
| dot | 1.2667 / 1.2651 | 1.1704 / 1.1823 | 1.1x / 1.1x |

- Simple arithmetic gains **1.3-2.7x** from SIMD (the larger gains are the ops that read and write
  fewer arrays: `neg`, `abs`, `square`, `relu`, `bias_add`).
- **`sqrt`, `exp`, `sigmoid`, `axpy` and `dot` gain nothing (0.9-1.1x).** The compiler did not
  vectorize them. Likely reasons, not checked: calls into the C library (`exp`, `fma`) and `sqrt`
  with error handling cannot be vectorized as written, and `dot` accumulates in order. These are the
  ops where a hand-written SIMD kernel could help most.

#### Threads at n = 100,000 (below the point where threads pay off)

| op | sequential ms | 1 thread | 8 threads | best of 2, 4, 8 threads |
|---|---|---|---|---|
| add | 0.02079 / 0.02069 | 1.11x / 0.96x | 0.36x / 0.32x | 0.59x / 0.50x |
| sub | 0.01919 / 0.02267 | 0.94x / 1.05x | 0.28x / 0.34x | 0.37x / 0.64x |
| mul | 0.02016 / 0.02313 | 0.98x / 1.09x | 0.30x / 0.34x | 0.38x / 0.66x |
| div | 0.02063 / 0.02261 | 0.98x / 0.99x | 0.31x / 0.35x | 0.39x / 0.56x |
| neg | 0.01566 / 0.01660 | 1.00x / 1.00x | 0.24x / 0.26x | 0.31x / 0.46x |
| abs | 0.01590 / 0.01755 | 0.99x / 1.00x | 0.24x / 0.26x | 0.33x / 0.54x |
| square | 0.01657 / 0.01695 | 1.01x / 1.01x | 0.23x / 0.27x | 0.32x / 0.47x |
| relu | 0.01751 / 0.01852 | 1.03x / 1.03x | 0.26x / 0.27x | 0.35x / 0.57x |
| sqrt | 0.08388 / 0.08929 | 1.00x / 1.01x | 0.96x / 1.06x | 0.99x / 1.40x |
| exp | 0.4516 / 0.4871 | 1.00x / 1.00x | 2.31x / 2.37x | 2.31x / 2.86x |
| sigmoid | 0.5180 / 0.5475 | 0.99x / 0.99x | 2.30x / 2.33x | 2.30x / 2.76x |
| axpy | 0.1908 / 0.1960 | 1.00x / 1.17x | 1.77x / 1.87x | 1.80x / 2.23x |
| bias_add | 0.02020 / 0.02148 | 0.93x / 0.90x | 0.27x / 0.30x | 0.37x / 0.56x |
| dot | 0.1149 / 0.1119 | 1.21x / 1.00x | 1.35x / 1.32x | 1.42x / 1.60x |

- For the **memory-bound** ops (`add`, `sub`, `mul`, `div`, `neg`, `abs`, `square`, `relu`,
  `bias_add`) 8 threads are **3-4x slower** than sequential (0.23-0.36x), and no thread count beats
  sequential. The single cutoff of 32,768 elements is too low for them.
- The **compute-bound** ops already benefit at this size: `exp` and `sigmoid` 2.3x, `axpy` 1.8x,
  `dot` 1.3-1.4x; `sqrt` breaks even.

#### Threads at n = 10,000,000

| op | sequential ms | 1 thread | 8 threads | best of 2, 4, 8 threads |
|---|---|---|---|---|
| add | 6.6756 / 6.5806 | 1.11x / 1.03x | 1.16x / 1.07x | 1.16x / 1.10x |
| sub | 6.4391 / 6.5929 | 1.02x / 1.01x | 1.12x / 1.09x | 1.12x / 1.18x |
| mul | 5.9621 / 6.4490 | 1.00x / 1.09x | 1.00x / 1.08x | 1.00x / 1.08x |
| div | 6.6056 / 6.6819 | 1.00x / 1.02x | 1.09x / 1.11x | 1.12x / 1.17x |
| neg | 4.2196 / 4.3606 | 1.04x / 1.02x | 0.88x / 0.92x | 0.99x / 0.94x |
| abs | 4.2196 / 4.2737 | 0.84x / 1.01x | 0.89x / 0.92x | 1.07x / 0.94x |
| square | 5.1371 / 4.2686 | 1.25x / 0.80x | 1.10x / 0.94x | 1.28x / 0.94x |
| relu | 4.3281 / 4.3852 | 1.05x / 1.03x | 0.94x / 0.97x | 0.94x / 0.97x |
| sqrt | 8.9500 / 9.1125 | 1.05x / 1.00x | 1.84x / 1.93x | 1.84x / 1.93x |
| exp | 43.0849 / 45.7848 | 1.01x / 1.00x | 3.47x / 3.49x | 3.47x / 3.49x |
| sigmoid | 46.0249 / 49.9925 | 0.93x / 1.00x | 3.06x / 3.12x | 3.06x / 3.53x |
| axpy | 19.2260 / 19.1744 | 0.99x / 1.03x | 2.99x / 3.11x | 2.99x / 3.11x |
| bias_add | 5.3904 / 4.3526 | 1.25x / 1.00x | 1.17x / 0.93x | 1.36x / 1.00x |
| dot | 12.5836 / 11.9880 | 1.03x / 1.00x | 4.63x / 4.46x | 4.63x / 4.46x |

- Memory-bound ops: **0.8-1.3x**. The sequential loop already moves data at the laptop's memory
  bandwidth, so threads add little and `neg`, `abs`, `relu` are even slightly slower.
- Compute-bound ops scale: `dot` 4.5x, `exp` 3.5x, `sigmoid` 3.1x, `axpy` 3.0x, `sqrt` 1.9x.

#### Matrix multiplication (one run, GFLOP/s, square matrices)

| n | scalar | sequential | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|---|---|
| 64 | 3.887 | 12.073 | 11.181 | 9.718 | 8.402 | 6.487 |
| 128 | 3.705 | 11.749 | 11.863 | 19.566 | 19.817 | 22.973 |
| 256 | 4.069 | 14.003 | 12.615 | 26.189 | 27.289 | 36.336 |
| 512 | 4.269 | 12.096 | 13.777 | 27.706 | 52.134 | 49.869 |
| 1024 | 4.301 | 12.616 | 11.423 | 13.112 | 23.803 | 38.989 |

- SIMD gives about 3x (scalar 3.7-4.3, sequential 11.7-14.0 GFLOP/s).
- Threads reach **39 GFLOP/s at n = 1024** (8 threads, about 3.1x sequential, CPU/wall 7.2 so the
  threads were busy) and **52 GFLOP/s at n = 512** (4 threads). At n = 64 threads are slower.
- This is one run, so the 2- and 4-thread values are uncertain (at n = 1024 with 2 threads the min
  was 103 ms and the median 164 ms). The earlier matmul numbers below used the other toolchain
  and were measured on battery; do not mix them with these.

Raw outputs of these runs are not stored in the repository (about 1,500 lines each); rerun the
benchmarks to regenerate them.

### Earlier results (before the scalar baseline; sequential as the only baseline)


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
| Code | branch `feat/benchmark-harness`, commit `dcdc3c7` (clean working tree) |
| Power state | Desktop on mains power, custom power plan **"Dedera", not High performance** (checked on 2026-11-01 before the runs): minimum processor state 80 %, maximum 95 %. A maximum below 100 % usually disables boost on Ryzen, so the CPU probably ran near its 3.7 GHz base clock. About 11 % CPU load before the runs, with Firefox open and no other heavy programs |
| Build | `cmake -S . -B bench -G "Visual Studio 18 2026" -A x64 -DMENSURFLOW_CUDA=OFF -DMENSURFLOW_NATIVE=OFF`, then `cmake --build bench --config Release`. All 14 tests pass. Benchmark header: `compiler: msvc 1951 \| OpenMP: yes (2.0) \| AVX2: yes \| build: Release` |
| Measurement date | 2026-11-01 |

> **The `scalar` rows are not scalar on MSVC.** MSVC does not know the `/Qvec-` flag that
> `CMakeLists.txt` sets for `src/backends/cpu/scalar.cpp` (`cl : command line warning D9002:
> ignoring unknown option '/Qvec-'`), so that file is auto-vectorized like `sequential.cpp`. The
> scalar rows match the sequential ones within noise (0.71-1.24x across all ops, 1.00x for
> `matmul`), so the `vs scalar` column carries no information in the desktop outputs. Everything
> below uses `vs seq`.

### Results (two runs, benchmark harness, sequential implementation as the baseline)

Speedups are the `vs seq` column: the sequential median time divided by the parallel median time.
Cells show run 1 / run 2. "24 threads" is the parallel kernel with all 24 hardware threads; "best"
is the fastest of 1, 2, 4, 8, 12, 16 and 24 threads, in brackets. The table uses Machine 1's four
summary sizes; the full outputs below also cover 1,000 to 75,000 and 200,000 to 5,000,000 elements.

| Op | n | Sequential (ms) | Speedup, 24 threads | Speedup, best (threads) |
|---|---|---|---|---|
| add | 100,000 | 0.013 / 0.013 | 0.30x / 0.30x | 0.99x (1) / 0.99x (1) |
| add | 1,000,000 | 0.127 / 0.127 | 1.16x / 1.08x | 1.48x (16) / 1.30x (8) |
| add | 10,000,000 | 3.62 / 3.59 | 1.25x / 1.05x | 1.25x (24) / 1.39x (8) |
| add | 33,554,432 | 15.2 / 15.0 | 1.08x / 1.07x | 1.15x (4) / 1.09x (12) |
| sub | 100,000 | 0.013 / 0.013 | 0.29x / 0.30x | 0.99x (1) / 0.99x (1) |
| sub | 1,000,000 | 0.128 / 0.128 | 1.11x / 1.13x | 1.43x (12) / 1.39x (8) |
| sub | 10,000,000 | 3.67 / 3.60 | 1.05x / 1.07x | 1.43x (8) / 1.43x (8) |
| sub | 33,554,432 | 14.9 / 15.0 | 1.07x / 1.05x | 1.11x (8) / 1.11x (8) |
| mul | 100,000 | 0.013 / 0.013 | 0.29x / 0.29x | 1.00x (1) / 0.99x (1) |
| mul | 1,000,000 | 0.127 / 0.128 | 1.08x / 1.11x | 1.31x (12) / 1.51x (12) |
| mul | 10,000,000 | 3.75 / 3.77 | 1.07x / 1.10x | 1.30x (8) / 1.48x (8) |
| mul | 33,554,432 | 15.1 / 14.9 | 1.07x / 1.06x | 1.12x (8) / 1.11x (8) |
| div | 100,000 | 0.023 / 0.023 | 0.50x / 0.51x | 0.99x (1) / 0.99x (1) |
| div | 1,000,000 | 0.228 / 0.229 | 1.88x / 1.93x | 1.88x (24) / 1.93x (24) |
| div | 10,000,000 | 3.74 / 3.66 | 1.13x / 1.07x | 1.35x (8) / 1.27x (8) |
| div | 33,554,432 | 14.9 / 14.9 | 1.07x / 1.06x | 1.10x (12) / 1.09x (12) |
| neg | 100,000 | 0.008 / 0.008 | 0.21x / 0.21x | 0.99x (1) / 0.99x (1) |
| neg | 1,000,000 | 0.085 / 0.085 | 0.75x / 0.76x | 1.01x (1) / 1.02x (1) |
| neg | 10,000,000 | 2.61 / 2.51 | 1.13x / 1.17x | 1.77x (8) / 1.80x (8) |
| neg | 33,554,432 | 10.7 / 11.1 | 1.03x / 1.08x | 1.15x (8) / 1.11x (12) |
| abs | 100,000 | 0.008 / 0.008 | 0.21x / 0.21x | 1.00x (1) / 0.99x (1) |
| abs | 1,000,000 | 0.084 / 0.085 | 0.74x / 0.76x | 1.00x (1) / 1.01x (1) |
| abs | 10,000,000 | 2.56 / 2.53 | 1.19x / 1.15x | 1.96x (8) / 2.07x (8) |
| abs | 33,554,432 | 10.7 / 10.8 | 1.02x / 1.05x | 1.15x (8) / 1.13x (8) |
| square | 100,000 | 0.008 / 0.008 | 0.21x / 0.21x | 1.00x (1) / 0.99x (1) |
| square | 1,000,000 | 0.084 / 0.085 | 0.77x / 0.77x | 1.01x (1) / 1.02x (1) |
| square | 10,000,000 | 2.60 / 2.49 | 1.18x / 1.11x | 1.80x (8) / 1.73x (8) |
| square | 33,554,432 | 10.6 / 10.8 | 1.05x / 1.04x | 1.11x (8) / 1.13x (8) |
| relu | 100,000 | 0.008 / 0.008 | 0.20x / 0.21x | 1.00x (1) / 0.99x (1) |
| relu | 1,000,000 | 0.086 / 0.086 | 0.74x / 0.78x | 1.02x (1) / 1.02x (1) |
| relu | 10,000,000 | 3.16 / 2.63 | 1.40x / 1.19x | 1.77x (12) / 1.84x (8) |
| relu | 33,554,432 | 11.5 / 11.0 | 1.12x / 1.08x | 1.12x (24) / 1.16x (8) |
| sqrt | 100,000 | 0.036 / 0.036 | 0.69x / 0.79x | 1.00x (1) / 1.00x (1) |
| sqrt | 1,000,000 | 0.360 / 0.359 | 2.20x / 2.35x | 2.20x (24) / 2.43x (12) |
| sqrt | 10,000,000 | 4.03 / 4.00 | 1.37x / 1.71x | 2.03x (8) / 2.17x (8) |
| sqrt | 33,554,432 | 14.1 / 13.6 | 1.21x / 1.24x | 1.23x (16) / 1.44x (8) |
| exp | 100,000 | 0.258 / 0.258 | 3.86x / 4.36x | 3.87x (12) / 5.40x (12) |
| exp | 1,000,000 | 2.60 / 2.59 | 9.26x / 9.92x | 9.26x (24) / 10.20x (12) |
| exp | 10,000,000 | 26.3 / 26.0 | 8.73x / 9.73x | 8.73x (24) / 9.73x (24) |
| exp | 33,554,432 | 88.3 / 87.3 | 8.22x / 7.75x | 8.22x (24) / 7.75x (24) |
| sigmoid | 100,000 | 0.321 / 0.320 | 4.47x / 5.05x | 4.47x (24) / 6.11x (12) |
| sigmoid | 1,000,000 | 3.29 / 3.21 | 10.65x / 10.80x | 10.65x (24) / 10.80x (24) |
| sigmoid | 10,000,000 | 33.0 / 32.2 | 11.50x / 11.22x | 11.50x (24) / 11.22x (24) |
| sigmoid | 33,554,432 | 108.1 / 108.1 | 9.71x / 9.89x | 9.71x (12) / 9.89x (24) |
| axpy | 100,000 | 0.165 / 0.165 | 3.37x / 3.36x | 4.19x (12) / 4.22x (12) |
| axpy | 1,000,000 | 1.65 / 1.65 | 12.00x / 8.94x | 12.00x (24) / 10.29x (16) |
| axpy | 10,000,000 | 16.7 / 16.7 | 4.67x / 4.79x | 5.93x (8) / 5.30x (12) |
| axpy | 33,554,432 | 57.3 / 57.3 | 3.96x / 4.06x | 4.31x (8) / 4.22x (8) |
| bias_add | 100,000 | 0.012 / 0.012 | 0.27x / 0.26x | 0.69x (1) / 0.69x (1) |
| bias_add | 1,000,000 | 0.107 / 0.106 | 0.92x / 0.92x | 1.01x (12) / 0.99x (12) |
| bias_add | 10,000,000 | 2.57 / 2.79 | 1.10x / 1.10x | 1.75x (8) / 1.85x (8) |
| bias_add | 33,554,432 | 10.7 / 11.1 | 1.06x / 1.09x | 1.15x (8) / 1.13x (12) |
| dot | 100,000 | 0.086 / 0.086 | 2.02x / 2.01x | 2.68x (8) / 2.77x (8) |
| dot | 1,000,000 | 0.860 / 0.860 | 11.17x / 11.10x | 11.17x (24) / 11.10x (24) |
| dot | 10,000,000 | 8.94 / 8.91 | 6.96x / 7.22x | 7.70x (8) / 9.13x (12) |
| dot | 33,554,432 | 29.9 / 29.9 | 4.36x / 4.95x | 4.55x (8) / 5.25x (12) |

What it shows:

- The simple elementwise ops (`add`, `sub`, `mul`, `div`, `neg`, `abs`, `square`, `relu`,
  `bias_add`) are limited by memory bandwidth on large arrays. The sequential `add`/`mul` already
  reach 31.8-33.5 GB/s at 10 million elements (dual-channel DDR4-3000 peaks at 48 GB/s in theory),
  and at 33,554,432 elements the best thread count gains only 1.09-1.16x.
- The ops with more arithmetic per element scale well: `exp` 7.8-9.9x and `sigmoid` 9.7-11.5x
  with 24 threads from 1 million elements on, `dot` 11.1-11.2x at 1 million and 4.4-5.0x at
  33,554,432, `sqrt` 2.2-2.4x at 1 million. `axpy` reaches 8.9-12x at 1 million: in this build
  without `MENSURFLOW_NATIVE` its sequential loop is slow (1.65 ms at 1 million, 13x slower than
  `add`), probably because `std::fma` has no hardware instruction to use, so it is limited by
  computation.
- Where extra threads stop helping: where threads help at all, the memory-bound ops are usually
  fastest with 8 threads (sometimes 4, 12 or 16), and 24 threads are slower than that. `div` is the
  exception, fastest with 24 threads at 1 and 2 million elements (and at 500,000 in run 1).
  `exp`, `sigmoid` and `axpy` keep improving up to 12-24 threads. `dot` is fastest with 24 threads from 1 to 5 million
  elements and with 8-12 from 10 million up.
- The cutoff (32,768 elements) is too low for the simple elementwise ops on this CPU. From 32,769
  to 200,000 elements no thread count is meaningfully faster than the sequential loop (24 threads:
  0.06-0.94x; the fastest is the parallel kernel with 1 thread, apart from about 1.0x for `div`
  with 8 threads at 200,000 and `bias_add` with 4 threads at 75,000). `add`, `mul` and `div` first
  gain at least 1.1x at 500,000 elements and `sub` at 1 million. The unary ops (`neg`, `abs`,
  `square`, `relu`) and `bias_add` gain at most 1.08x up to 2 million elements (one exception:
  `relu` 1.34x at 2 million in run 1) and first gain clearly at 5 million. For `exp`, `sigmoid`,
  `axpy` and `dot` the cutoff is fine: at 32,769 elements they already gain 2.5-2.7x, 2.4-3.1x,
  2.1x and 1.4x. `sqrt` does not gain until somewhere between 100,000 and 500,000.
- Surprising: the unary ops jump from about 1.05x at 2 million elements to 1.7-2.8x at 5 million
  and stay at 1.7-2.1x at 10 million, then fall to 1.1x at 33,554,432. A likely reason is the cache
  layout: 2 million elements in and out (16 MB) fit in one core complex's 32 MB L3, so one thread
  already runs at cache speed and threads add little. From 5 million up the sequential loop runs
  from memory, and threads spread over both core complexes get more of the two L3 caches and more
  memory requests in flight. Not confirmed (thread placement is not fixed).
- The parallel kernel with 1 thread is within ±5 % of the sequential loop in 202 of the 224 rows
  from 100,000 elements up. `bias_add` with 1 thread is consistently slower (0.69x at 100,000
  elements in both runs); the other outliers are scattered over the memory-bound ops.

Reliability: within each run the timings are steady: no row has a max of 3x its median or more,
and only 8 of 3,780 rows reach 2x. Between the two runs, 257 of the 1,542 rows with a median of
at least 0.005 ms (17 %) differ by more than 10 %, although the median row differs by only 1.3 %.
By the rule under "How to run" that counts as noisy. Two patterns:

- 247 of the 257 are parallel rows. The largest differences are almost exactly 2x, with 2-4
  threads, in `exp`, `sqrt` and `sigmoid` at sizes from 200,000 up and in `add` at 1 and
  2 million (for example `exp` at 10 million with 2 threads: 25.9 ms / 13.0 ms; run 1 was the slow
  one). A whole run at half speed for some thread counts fits two threads landing on the two
  hardware threads of one physical core, which gives no gain for compute-bound ops; affinity is not
  set, so Windows decides when the program starts. Not confirmed.
- The other 10 are sequential or scalar rows between 200,000 and 10 million elements, all but one
  (`exp` scalar at 500,000) of memory-bound ops, at sizes where the data is about the size of the
  L3 cache.

Treat the ratios for 2-8 threads and for 2 to 10 million elements as uncertain. The conclusions
above hold in both runs.

<details>
<summary>Full output: cpu_parallel_benchmark, run 1</summary>

```
=== Scalar vs sequential vs parallel on the CPU (float32) ===
cpu: AMD Ryzen 9 5900X 12-Core Processor | hardware threads: 24 | OpenMP threads: 24
compiler: msvc 1951 | OpenMP: yes (2.0) | AVX2: yes | build: Release
timing: 1 warm-up call(s), median of 9 samples (3 if a call takes over 0.2 s), each sample at least 2 ms
for comparable numbers: plugged in, High performance power plan, other programs closed, run twice
impl: scalar = sequential.cpp without auto-vectorization; sequential = the same code with the compiler's SIMD; parallel = the parallel kernel with N threads
vs scalar / vs seq = that baseline's median time / this row's median time
parallel cutoff: 32768 elements
axpy: alpha=1.250000, fused multiply-add
bias_add: features = largest power-of-two divisor of n up to 256; the first features entries of b are the bias; GB/s counts 8*n + 4*features bytes

== add ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    141.440      1.00x      0.99x         -
      1000   sequential        -      0.0001      0.0001      0.0001    142.161      1.01x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    135.948      0.96x      0.96x         -
      1000     parallel        2      0.0001      0.0001      0.0001    136.330      0.96x      0.96x         -
      1000     parallel        4      0.0001      0.0001      0.0001    136.349      0.96x      0.96x         -
      1000     parallel        8      0.0001      0.0001      0.0001    136.245      0.96x      0.96x         -
      1000     parallel       12      0.0001      0.0001      0.0001    136.127      0.96x      0.96x         -
      1000     parallel       16      0.0001      0.0001      0.0001    136.387      0.96x      0.96x         -
      1000     parallel       24      0.0001      0.0001      0.0001    136.183      0.96x      0.96x         -

     10000       scalar        -      0.0011      0.0011      0.0011    110.236      1.00x      1.00x         -
     10000   sequential        -      0.0011      0.0011      0.0011    110.216      1.00x      1.00x         -
     10000     parallel        1      0.0011      0.0011      0.0011    109.436      0.99x      0.99x         -
     10000     parallel        2      0.0011      0.0011      0.0011    109.524      0.99x      0.99x         -
     10000     parallel        4      0.0011      0.0011      0.0011    109.533      0.99x      0.99x         -
     10000     parallel        8      0.0011      0.0011      0.0011    109.587      0.99x      0.99x         -
     10000     parallel       12      0.0011      0.0011      0.0011    109.407      0.99x      0.99x         -
     10000     parallel       16      0.0011      0.0011      0.0011    109.455      0.99x      0.99x         -
     10000     parallel       24      0.0011      0.0011      0.0011    108.527      0.98x      0.98x         -

     32768       scalar        -      0.0036      0.0036      0.0037    108.339      1.00x      0.99x         -
     32768   sequential        -      0.0036      0.0036      0.0036    109.405      1.01x      1.00x         -
     32768     parallel        1      0.0036      0.0036      0.0036    108.655      1.00x      0.99x         -
     32768     parallel        2      0.0036      0.0036      0.0044    108.649      1.00x      0.99x         -
     32768     parallel        4      0.0036      0.0036      0.0036    109.159      1.01x      1.00x         -
     32768     parallel        8      0.0036      0.0036      0.0036    109.034      1.01x      1.00x         -
     32768     parallel       12      0.0036      0.0036      0.0041    108.952      1.01x      1.00x         -
     32768     parallel       16      0.0036      0.0036      0.0036    109.721      1.01x      1.00x         -
     32768     parallel       24      0.0036      0.0036      0.0037    109.712      1.01x      1.00x         -

     32769       scalar        -      0.0036      0.0036      0.0036    110.534      1.00x      1.01x         -
     32769   sequential        -      0.0036      0.0036      0.0036    109.745      0.99x      1.00x         -
     32769     parallel        1      0.0036      0.0036      0.0037    109.796      0.99x      1.00x         -
     32769     parallel        2      0.0162      0.0162      0.0180     24.209      0.22x      0.22x         -
     32769     parallel        4      0.0162      0.0162      0.0171     24.232      0.22x      0.22x         -
     32769     parallel        8      0.0238      0.0237      0.0247     16.502      0.15x      0.15x         -
     32769     parallel       12      0.0279      0.0278      0.0281     14.080      0.13x      0.13x         -
     32769     parallel       16      0.0306      0.0301      0.0310     12.866      0.12x      0.12x         -
     32769     parallel       24      0.0370      0.0364      0.0383     10.627      0.10x      0.10x         -

     40000       scalar        -      0.0047      0.0047      0.0047    102.362      1.00x      1.01x         -
     40000   sequential        -      0.0047      0.0047      0.0048    101.701      0.99x      1.00x         -
     40000     parallel        1      0.0047      0.0047      0.0047    102.293      1.00x      1.01x         -
     40000     parallel        2      0.0175      0.0175      0.0177     27.413      0.27x      0.27x         -
     40000     parallel        4      0.0168      0.0167      0.0172     28.645      0.28x      0.28x         -
     40000     parallel        8      0.0255      0.0228      0.0284     18.818      0.18x      0.19x         -
     40000     parallel       12      0.0286      0.0280      0.0295     16.781      0.16x      0.17x         -
     40000     parallel       16      0.0309      0.0304      0.0315     15.515      0.15x      0.15x         -
     40000     parallel       24      0.0372      0.0361      0.0396     12.897      0.13x      0.13x         -

     50000       scalar        -      0.0061      0.0060      0.0062     98.462      1.00x      0.98x         -
     50000   sequential        -      0.0060      0.0060      0.0061    100.669      1.02x      1.00x         -
     50000     parallel        1      0.0060      0.0060      0.0061     99.517      1.01x      0.99x         -
     50000     parallel        2      0.0185      0.0184      0.0209     32.382      0.33x      0.32x         -
     50000     parallel        4      0.0176      0.0175      0.0178     34.112      0.35x      0.34x         -
     50000     parallel        8      0.0244      0.0242      0.0277     24.637      0.25x      0.24x         -
     50000     parallel       12      0.0299      0.0294      0.0308     20.087      0.20x      0.20x         -
     50000     parallel       16      0.0313      0.0310      0.0321     19.173      0.19x      0.19x         -
     50000     parallel       24      0.0373      0.0363      0.0380     16.099      0.16x      0.16x         -

     75000       scalar        -      0.0095      0.0095      0.0096     95.042      1.00x      0.99x         -
     75000   sequential        -      0.0094      0.0094      0.0095     95.916      1.01x      1.00x         -
     75000     parallel        1      0.0094      0.0094      0.0095     95.258      1.00x      0.99x         -
     75000     parallel        2      0.0243      0.0242      0.0245     37.081      0.39x      0.39x         -
     75000     parallel        4      0.0191      0.0190      0.0196     47.155      0.50x      0.49x         -
     75000     parallel        8      0.0253      0.0251      0.0280     35.527      0.37x      0.37x         -
     75000     parallel       12      0.0319      0.0313      0.0322     28.219      0.30x      0.29x         -
     75000     parallel       16      0.0333      0.0323      0.0356     27.063      0.28x      0.28x         -
     75000     parallel       24      0.0394      0.0389      0.0422     22.853      0.24x      0.24x         -

    100000       scalar        -      0.0127      0.0127      0.0128     94.314      1.00x      0.99x         -
    100000   sequential        -      0.0126      0.0126      0.0127     95.238      1.01x      1.00x         -
    100000     parallel        1      0.0127      0.0127      0.0128     94.282      1.00x      0.99x         -
    100000     parallel        2      0.0288      0.0287      0.0291     41.611      0.44x      0.44x         -
    100000     parallel        4      0.0215      0.0214      0.0222     55.920      0.59x      0.59x         -
    100000     parallel        8      0.0271      0.0268      0.0272     44.321      0.47x      0.47x         -
    100000     parallel       12      0.0344      0.0340      0.0351     34.896      0.37x      0.37x         -
    100000     parallel       16      0.0347      0.0339      0.0352     34.612      0.37x      0.36x         -
    100000     parallel       24      0.0415      0.0405      0.0419     28.911      0.31x      0.30x         -

    200000       scalar        -      0.0253      0.0253      0.0255     94.730      1.00x      1.00x         -
    200000   sequential        -      0.0254      0.0254      0.0255     94.497      1.00x      1.00x         -
    200000     parallel        1      0.0253      0.0253      0.0255     94.806      1.00x      1.00x         -
    200000     parallel        2      0.0463      0.0459      0.0475     51.815      0.55x      0.55x         -
    200000     parallel        4      0.0313      0.0300      0.0330     76.628      0.81x      0.81x         -
    200000     parallel        8      0.0370      0.0368      0.0377     64.783      0.68x      0.69x         -
    200000     parallel       12      0.0429      0.0418      0.0445     56.005      0.59x      0.59x         -
    200000     parallel       16      0.0433      0.0426      0.0443     55.407      0.58x      0.59x         -
    200000     parallel       24      0.0474      0.0471      0.0489     50.665      0.53x      0.54x         -

    500000       scalar        -      0.0632      0.0631      0.0640     94.960      1.00x      1.00x         -
    500000   sequential        -      0.0634      0.0633      0.0641     94.661      1.00x      1.00x         -
    500000     parallel        1      0.0629      0.0629      0.0636     95.337      1.00x      1.01x         -
    500000     parallel        2      0.0965      0.0964      0.0974     62.154      0.65x      0.66x         -
    500000     parallel        4      0.0563      0.0562      0.0582    106.602      1.12x      1.13x         -
    500000     parallel        8      0.0562      0.0556      0.0571    106.854      1.13x      1.13x         -
    500000     parallel       12      0.0630      0.0614      0.0674     95.214      1.00x      1.01x         -
    500000     parallel       16      0.0712      0.0570      0.0723     84.296      0.89x      0.89x         -
    500000     parallel       24      0.0731      0.0712      0.0762     82.058      0.86x      0.87x         -

   1000000       scalar        -      0.1287      0.1272      0.1379     93.222      1.00x      0.99x         -
   1000000   sequential        -      0.1272      0.1266      0.1319     94.340      1.01x      1.00x         -
   1000000     parallel        1      0.1272      0.1269      0.1351     94.367      1.01x      1.00x         -
   1000000     parallel        2      0.3960      0.3913      0.4031     30.303      0.33x      0.32x         -
   1000000     parallel        4      0.2112      0.2048      0.2143     56.825      0.61x      0.60x         -
   1000000     parallel        8      0.1174      0.1159      0.1265    102.228      1.10x      1.08x         -
   1000000     parallel       12      0.0870      0.0857      0.1004    137.906      1.48x      1.46x         -
   1000000     parallel       16      0.0860      0.0827      0.0902    139.540      1.50x      1.48x         -
   1000000     parallel       24      0.1094      0.1010      0.1101    109.661      1.18x      1.16x         -

   2000000       scalar        -      0.2888      0.2572      0.3819     83.106      1.00x      0.91x         -
   2000000   sequential        -      0.2631      0.2571      0.3282     91.233      1.10x      1.00x         -
   2000000     parallel        1      0.2696      0.2577      0.2730     89.037      1.07x      0.98x         -
   2000000     parallel        2      0.7839      0.7778      1.0198     30.616      0.37x      0.34x         -
   2000000     parallel        4      0.4092      0.3989      0.4186     58.651      0.71x      0.64x         -
   2000000     parallel        8      0.2221      0.2167      0.2372    108.078      1.30x      1.18x         -
   2000000     parallel       12      0.1612      0.1569      0.1801    148.866      1.79x      1.63x         -
   2000000     parallel       16      0.1742      0.1656      0.1941    137.763      1.66x      1.51x         -
   2000000     parallel       24      0.2388      0.1888      0.2952    100.508      1.21x      1.10x         -

   5000000       scalar        -      1.5026      1.4913      1.5321     39.931      1.00x      1.00x         -
   5000000   sequential        -      1.5003      1.4762      1.5603     39.993      1.00x      1.00x         -
   5000000     parallel        1      1.4985      1.4515      1.5533     40.040      1.00x      1.00x         -
   5000000     parallel        2      2.0192      1.9994      2.0569     29.715      0.74x      0.74x         -
   5000000     parallel        4      1.4770      1.1906      1.5742     40.622      1.02x      1.02x         -
   5000000     parallel        8      1.4100      1.3095      1.5453     42.555      1.07x      1.06x         -
   5000000     parallel       12      1.0049      0.7128      1.3651     59.709      1.50x      1.49x         -
   5000000     parallel       16      0.9953      0.6952      1.4469     60.283      1.51x      1.51x         -
   5000000     parallel       24      1.2378      1.0070      1.2974     48.473      1.21x      1.21x         -

  10000000       scalar        -      3.7190      3.5902      3.8583     32.267      1.00x      0.97x         -
  10000000   sequential        -      3.6194      3.5480      3.8293     33.155      1.03x      1.00x         -
  10000000     parallel        1      3.6137      3.5807      3.8002     33.207      1.03x      1.00x         -
  10000000     parallel        2      4.3061      4.2444      6.5295     27.867      0.86x      0.84x         -
  10000000     parallel        4      3.4595      3.4124      3.7263     34.687      1.08x      1.05x         -
  10000000     parallel        8      3.3289      3.2861      3.4673     36.048      1.12x      1.09x         -
  10000000     parallel       12      3.5669      3.4639      3.8056     33.643      1.04x      1.01x         -
  10000000     parallel       16      3.0017      2.8167      3.2236     39.977      1.24x      1.21x         -
  10000000     parallel       24      2.8933      2.6932      2.9956     41.475      1.29x      1.25x         -

  33554432       scalar        -     15.0362     14.9324     15.9516     26.779      1.00x      1.01x      1.03
  33554432   sequential        -     15.1857     15.1148     15.3836     26.515      0.99x      1.00x      0.91
  33554432     parallel        1     15.3101     15.1223     19.1886     26.300      0.98x      0.99x      0.97
  33554432     parallel        2     15.5407     15.4161     17.7512     25.910      0.97x      0.98x      1.96
  33554432     parallel        4     13.1592     12.8489     15.5186     30.599      1.14x      1.15x      4.52
  33554432     parallel        8     13.7497     13.3310     14.5725     29.285      1.09x      1.10x      7.40
  33554432     parallel       12     14.1903     13.6008     15.0375     28.375      1.06x      1.07x     10.28
  33554432     parallel       16     13.9536     13.5075     14.3521     28.857      1.08x      1.09x     15.50
  33554432     parallel       24     14.0022     13.6043     14.3216     28.756      1.07x      1.08x     23.02


== sub ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    142.635      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    141.935      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    137.243      0.96x      0.97x         -
      1000     parallel        2      0.0001      0.0001      0.0001    137.325      0.96x      0.97x         -
      1000     parallel        4      0.0001      0.0001      0.0001    137.445      0.96x      0.97x         -
      1000     parallel        8      0.0001      0.0001      0.0001    137.119      0.96x      0.97x         -
      1000     parallel       12      0.0001      0.0001      0.0001    137.632      0.96x      0.97x         -
      1000     parallel       16      0.0001      0.0001      0.0001    137.397      0.96x      0.97x         -
      1000     parallel       24      0.0001      0.0001      0.0001    137.512      0.96x      0.97x         -

     10000       scalar        -      0.0011      0.0011      0.0011    110.389      1.00x      1.00x         -
     10000   sequential        -      0.0011      0.0011      0.0011    110.162      1.00x      1.00x         -
     10000     parallel        1      0.0011      0.0011      0.0011    109.807      0.99x      1.00x         -
     10000     parallel        2      0.0011      0.0011      0.0011    109.812      0.99x      1.00x         -
     10000     parallel        4      0.0011      0.0011      0.0011    109.798      0.99x      1.00x         -
     10000     parallel        8      0.0011      0.0011      0.0011    109.793      0.99x      1.00x         -
     10000     parallel       12      0.0011      0.0011      0.0011    109.783      0.99x      1.00x         -
     10000     parallel       16      0.0011      0.0011      0.0011    109.793      0.99x      1.00x         -
     10000     parallel       24      0.0011      0.0011      0.0011    109.793      0.99x      1.00x         -

     32768       scalar        -      0.0037      0.0037      0.0037    106.031      1.00x      0.99x         -
     32768   sequential        -      0.0037      0.0037      0.0037    106.949      1.01x      1.00x         -
     32768     parallel        1      0.0037      0.0037      0.0037    106.570      1.01x      1.00x         -
     32768     parallel        2      0.0037      0.0037      0.0037    106.480      1.00x      1.00x         -
     32768     parallel        4      0.0037      0.0037      0.0037    106.494      1.00x      1.00x         -
     32768     parallel        8      0.0037      0.0037      0.0039    106.598      1.01x      1.00x         -
     32768     parallel       12      0.0037      0.0037      0.0037    106.423      1.00x      1.00x         -
     32768     parallel       16      0.0037      0.0037      0.0037    106.483      1.00x      1.00x         -
     32768     parallel       24      0.0037      0.0037      0.0037    106.205      1.00x      0.99x         -

     32769       scalar        -      0.0037      0.0037      0.0038    105.567      1.00x      1.00x         -
     32769   sequential        -      0.0037      0.0037      0.0038    105.784      1.00x      1.00x         -
     32769     parallel        1      0.0037      0.0037      0.0037    106.365      1.01x      1.01x         -
     32769     parallel        2      0.0156      0.0153      0.0168     25.258      0.24x      0.24x         -
     32769     parallel        4      0.0153      0.0152      0.0154     25.730      0.24x      0.24x         -
     32769     parallel        8      0.0223      0.0222      0.0228     17.596      0.17x      0.17x         -
     32769     parallel       12      0.0281      0.0280      0.0282     14.009      0.13x      0.13x         -
     32769     parallel       16      0.0300      0.0295      0.0305     13.104      0.12x      0.12x         -
     32769     parallel       24      0.0368      0.0361      0.0374     10.690      0.10x      0.10x         -

     40000       scalar        -      0.0047      0.0047      0.0048    101.550      1.00x      0.97x         -
     40000   sequential        -      0.0046      0.0046      0.0047    104.388      1.03x      1.00x         -
     40000     parallel        1      0.0048      0.0047      0.0048    100.763      0.99x      0.97x         -
     40000     parallel        2      0.0174      0.0172      0.0185     27.648      0.27x      0.26x         -
     40000     parallel        4      0.0156      0.0156      0.0167     30.689      0.30x      0.29x         -
     40000     parallel        8      0.0238      0.0228      0.0243     20.185      0.20x      0.19x         -
     40000     parallel       12      0.0284      0.0282      0.0287     16.911      0.17x      0.16x         -
     40000     parallel       16      0.0304      0.0301      0.0310     15.814      0.16x      0.15x         -
     40000     parallel       24      0.0371      0.0361      0.0374     12.948      0.13x      0.12x         -

     50000       scalar        -      0.0059      0.0059      0.0060    101.166      1.00x      1.00x         -
     50000   sequential        -      0.0059      0.0059      0.0061    100.913      1.00x      1.00x         -
     50000     parallel        1      0.0060      0.0059      0.0060    100.537      0.99x      1.00x         -
     50000     parallel        2      0.0190      0.0190      0.0196     31.578      0.31x      0.31x         -
     50000     parallel        4      0.0168      0.0167      0.0170     35.809      0.35x      0.35x         -
     50000     parallel        8      0.0234      0.0232      0.0254     25.674      0.25x      0.25x         -
     50000     parallel       12      0.0292      0.0291      0.0299     20.572      0.20x      0.20x         -
     50000     parallel       16      0.0305      0.0303      0.0310     19.674      0.19x      0.19x         -
     50000     parallel       24      0.0378      0.0371      0.0383     15.877      0.16x      0.16x         -

     75000       scalar        -      0.0094      0.0094      0.0095     95.459      1.00x      0.99x         -
     75000   sequential        -      0.0093      0.0093      0.0094     96.313      1.01x      1.00x         -
     75000     parallel        1      0.0096      0.0095      0.0107     93.918      0.98x      0.98x         -
     75000     parallel        2      0.0264      0.0257      0.0271     34.086      0.36x      0.35x         -
     75000     parallel        4      0.0197      0.0188      0.0202     45.700      0.48x      0.47x         -
     75000     parallel        8      0.0263      0.0254      0.0290     34.204      0.36x      0.36x         -
     75000     parallel       12      0.0316      0.0312      0.0319     28.484      0.30x      0.30x         -
     75000     parallel       16      0.0327      0.0325      0.0373     27.499      0.29x      0.29x         -
     75000     parallel       24      0.0394      0.0383      0.0413     22.852      0.24x      0.24x         -

    100000       scalar        -      0.0127      0.0127      0.0130     94.196      1.00x      0.99x         -
    100000   sequential        -      0.0126      0.0126      0.0128     95.050      1.01x      1.00x         -
    100000     parallel        1      0.0128      0.0127      0.0129     94.017      1.00x      0.99x         -
    100000     parallel        2      0.0309      0.0292      0.0421     38.894      0.41x      0.41x         -
    100000     parallel        4      0.0216      0.0214      0.0223     55.666      0.59x      0.59x         -
    100000     parallel        8      0.0310      0.0303      0.0312     38.738      0.41x      0.41x         -
    100000     parallel       12      0.0334      0.0326      0.0338     35.901      0.38x      0.38x         -
    100000     parallel       16      0.0346      0.0339      0.0349     34.721      0.37x      0.37x         -
    100000     parallel       24      0.0432      0.0418      0.0443     27.763      0.29x      0.29x         -

    200000       scalar        -      0.0254      0.0253      0.0257     94.546      1.00x      1.00x         -
    200000   sequential        -      0.0255      0.0255      0.0258     94.092      1.00x      1.00x         -
    200000     parallel        1      0.0253      0.0253      0.0256     94.715      1.00x      1.01x         -
    200000     parallel        2      0.0460      0.0458      0.0465     52.156      0.55x      0.55x         -
    200000     parallel        4      0.0307      0.0305      0.0312     78.152      0.83x      0.83x         -
    200000     parallel        8      0.0360      0.0353      0.0402     66.629      0.70x      0.71x         -
    200000     parallel       12      0.0399      0.0387      0.0407     60.108      0.64x      0.64x         -
    200000     parallel       16      0.0425      0.0414      0.0434     56.460      0.60x      0.60x         -
    200000     parallel       24      0.0499      0.0482      0.0515     48.068      0.51x      0.51x         -

    500000       scalar        -      0.0630      0.0630      0.0638     95.196      1.00x      1.01x         -
    500000   sequential        -      0.0634      0.0633      0.0637     94.693      0.99x      1.00x         -
    500000     parallel        1      0.0630      0.0629      0.0636     95.252      1.00x      1.01x         -
    500000     parallel        2      0.1018      0.0989      0.1064     58.961      0.62x      0.62x         -
    500000     parallel        4      0.0621      0.0580      0.0694     96.677      1.02x      1.02x         -
    500000     parallel        8      0.0581      0.0578      0.0626    103.281      1.08x      1.09x         -
    500000     parallel       12      0.0692      0.0601      0.0769     86.705      0.91x      0.92x         -
    500000     parallel       16      0.0713      0.0682      0.0736     84.174      0.88x      0.89x         -
    500000     parallel       24      0.0755      0.0724      0.0939     79.497      0.84x      0.84x         -

   1000000       scalar        -      0.1272      0.1268      0.1310     94.326      1.00x      1.00x         -
   1000000   sequential        -      0.1278      0.1270      0.1291     93.897      1.00x      1.00x         -
   1000000     parallel        1      0.1273      0.1267      0.1654     94.242      1.00x      1.00x         -
   1000000     parallel        2      0.1889      0.1871      0.2211     63.540      0.67x      0.68x         -
   1000000     parallel        4      0.0988      0.0984      0.0992    121.411      1.29x      1.29x         -
   1000000     parallel        8      0.0951      0.0904      0.0966    126.245      1.34x      1.34x         -
   1000000     parallel       12      0.0894      0.0834      0.1027    134.172      1.42x      1.43x         -
   1000000     parallel       16      0.1020      0.0915      0.1051    117.683      1.25x      1.25x         -
   1000000     parallel       24      0.1156      0.1109      0.1184    103.840      1.10x      1.11x         -

   2000000       scalar        -      0.2577      0.2532      0.3035     93.132      1.00x      0.99x         -
   2000000   sequential        -      0.2563      0.2538      0.2981     93.649      1.01x      1.00x         -
   2000000     parallel        1      0.5231      0.4745      0.6276     45.878      0.49x      0.49x         -
   2000000     parallel        2      0.5673      0.4646      0.7132     42.308      0.45x      0.45x         -
   2000000     parallel        4      0.3171      0.2128      0.3993     75.689      0.81x      0.81x         -
   2000000     parallel        8      0.1806      0.1589      0.2287    132.909      1.43x      1.42x         -
   2000000     parallel       12      0.1621      0.1428      0.1896    148.051      1.59x      1.58x         -
   2000000     parallel       16      0.1695      0.1516      0.1933    141.603      1.52x      1.51x         -
   2000000     parallel       24      0.1906      0.1825      0.1945    125.935      1.35x      1.34x         -

   5000000       scalar        -      1.6064      1.5626      1.9222     37.352      1.00x      1.05x         -
   5000000   sequential        -      1.6822      1.5725      1.8419     35.668      0.95x      1.00x         -
   5000000     parallel        1      1.7848      1.5466      2.1740     33.618      0.90x      0.94x         -
   5000000     parallel        2      1.6511      1.4119      1.9824     36.339      0.97x      1.02x         -
   5000000     parallel        4      1.4366      1.1085      1.8123     41.765      1.12x      1.17x         -
   5000000     parallel        8      0.6708      0.6314      1.1772     89.452      2.39x      2.51x         -
   5000000     parallel       12      1.2103      1.1653      1.3448     49.577      1.33x      1.39x         -
   5000000     parallel       16      1.1758      1.0013      1.3678     51.029      1.37x      1.43x         -
   5000000     parallel       24      1.2513      1.1405      1.4161     47.948      1.28x      1.34x         -

  10000000       scalar        -      3.7141      3.5810      4.6423     32.309      1.00x      0.99x         -
  10000000   sequential        -      3.6683      3.5724      4.7822     32.713      1.01x      1.00x         -
  10000000     parallel        1      3.7256      3.6064      3.8708     32.210      1.00x      0.98x         -
  10000000     parallel        2      3.8045      3.2837      4.4273     31.542      0.98x      0.96x         -
  10000000     parallel        4      3.4935      3.3541      4.1724     34.350      1.06x      1.05x         -
  10000000     parallel        8      2.5607      2.5106      3.1989     46.862      1.45x      1.43x         -
  10000000     parallel       12      3.4180      3.1330      4.0836     35.108      1.09x      1.07x         -
  10000000     parallel       16      3.5662      3.3942      3.6646     33.649      1.04x      1.03x         -
  10000000     parallel       24      3.5066      3.3824      3.8258     34.221      1.06x      1.05x         -

  33554432       scalar        -     15.2391     14.9995     16.1691     26.422      1.00x      0.98x      1.02
  33554432   sequential        -     14.9363     14.8055     16.3089     26.958      1.02x      1.00x      1.04
  33554432     parallel        1     15.1984     14.8732     16.3595     26.493      1.00x      0.98x      1.02
  33554432     parallel        2     13.8220     13.7899     15.6763     29.131      1.10x      1.08x      2.22
  33554432     parallel        4     14.5106     14.2890     15.6273     27.749      1.05x      1.03x      4.25
  33554432     parallel        8     13.4255     13.1943     13.9553     29.992      1.14x      1.11x      6.29
  33554432     parallel       12     13.7276     13.2071     14.1711     29.332      1.11x      1.09x     11.14
  33554432     parallel       16     14.1814     13.9616     14.7570     28.393      1.07x      1.05x     15.52
  33554432     parallel       24     14.0004     13.9165     14.4469     28.760      1.09x      1.07x     21.97


== mul ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    142.837      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    142.563      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    137.014      0.96x      0.96x         -
      1000     parallel        2      0.0001      0.0001      0.0001    137.315      0.96x      0.96x         -
      1000     parallel        4      0.0001      0.0001      0.0001    137.028      0.96x      0.96x         -
      1000     parallel        8      0.0001      0.0001      0.0001    136.856      0.96x      0.96x         -
      1000     parallel       12      0.0001      0.0001      0.0001    136.795      0.96x      0.96x         -
      1000     parallel       16      0.0001      0.0001      0.0001    137.114      0.96x      0.96x         -
      1000     parallel       24      0.0001      0.0001      0.0001    137.320      0.96x      0.96x         -

     10000       scalar        -      0.0011      0.0011      0.0011    110.399      1.00x      1.00x         -
     10000   sequential        -      0.0011      0.0011      0.0011    110.444      1.00x      1.00x         -
     10000     parallel        1      0.0011      0.0011      0.0011    109.646      0.99x      0.99x         -
     10000     parallel        2      0.0011      0.0011      0.0011    109.749      0.99x      0.99x         -
     10000     parallel        4      0.0011      0.0011      0.0011    109.822      0.99x      0.99x         -
     10000     parallel        8      0.0011      0.0011      0.0011    109.817      0.99x      0.99x         -
     10000     parallel       12      0.0011      0.0011      0.0012    109.827      0.99x      0.99x         -
     10000     parallel       16      0.0011      0.0011      0.0011    109.798      0.99x      0.99x         -
     10000     parallel       24      0.0011      0.0011      0.0011    109.857      1.00x      0.99x         -

     32768       scalar        -      0.0035      0.0035      0.0036    111.282      1.00x      1.03x         -
     32768   sequential        -      0.0036      0.0036      0.0037    108.115      0.97x      1.00x         -
     32768     parallel        1      0.0035      0.0035      0.0036    111.110      1.00x      1.03x         -
     32768     parallel        2      0.0035      0.0035      0.0036    111.104      1.00x      1.03x         -
     32768     parallel        4      0.0035      0.0035      0.0038    111.098      1.00x      1.03x         -
     32768     parallel        8      0.0035      0.0035      0.0036    111.114      1.00x      1.03x         -
     32768     parallel       12      0.0035      0.0035      0.0036    111.120      1.00x      1.03x         -
     32768     parallel       16      0.0035      0.0035      0.0036    111.117      1.00x      1.03x         -
     32768     parallel       24      0.0035      0.0035      0.0036    111.117      1.00x      1.03x         -

     32769       scalar        -      0.0035      0.0035      0.0036    111.258      1.00x      1.03x         -
     32769   sequential        -      0.0036      0.0036      0.0037    108.116      0.97x      1.00x         -
     32769     parallel        1      0.0035      0.0035      0.0036    110.906      1.00x      1.03x         -
     32769     parallel        2      0.0160      0.0160      0.0164     24.522      0.22x      0.23x         -
     32769     parallel        4      0.0152      0.0151      0.0154     25.904      0.23x      0.24x         -
     32769     parallel        8      0.0223      0.0222      0.0227     17.602      0.16x      0.16x         -
     32769     parallel       12      0.0277      0.0276      0.0279     14.190      0.13x      0.13x         -
     32769     parallel       16      0.0304      0.0301      0.0306     12.949      0.12x      0.12x         -
     32769     parallel       24      0.0372      0.0359      0.0404     10.570      0.10x      0.10x         -

     40000       scalar        -      0.0048      0.0048      0.0049    100.155      1.00x      0.99x         -
     40000   sequential        -      0.0047      0.0047      0.0048    101.575      1.01x      1.00x         -
     40000     parallel        1      0.0048      0.0048      0.0049     99.684      1.00x      0.98x         -
     40000     parallel        2      0.0182      0.0175      0.0189     26.312      0.26x      0.26x         -
     40000     parallel        4      0.0159      0.0159      0.0162     30.205      0.30x      0.30x         -
     40000     parallel        8      0.0249      0.0244      0.0263     19.255      0.19x      0.19x         -
     40000     parallel       12      0.0286      0.0283      0.0296     16.796      0.17x      0.17x         -
     40000     parallel       16      0.0297      0.0295      0.0303     16.185      0.16x      0.16x         -
     40000     parallel       24      0.0376      0.0369      0.0463     12.775      0.13x      0.13x         -

     50000       scalar        -      0.0060      0.0060      0.0060    100.593      1.00x      1.00x         -
     50000   sequential        -      0.0060      0.0059      0.0060    100.834      1.00x      1.00x         -
     50000     parallel        1      0.0059      0.0059      0.0060    100.933      1.00x      1.00x         -
     50000     parallel        2      0.0195      0.0194      0.0197     30.826      0.31x      0.31x         -
     50000     parallel        4      0.0169      0.0168      0.0172     35.514      0.35x      0.35x         -
     50000     parallel        8      0.0235      0.0233      0.0237     25.568      0.25x      0.25x         -
     50000     parallel       12      0.0295      0.0292      0.0300     20.363      0.20x      0.20x         -
     50000     parallel       16      0.0313      0.0311      0.0327     19.184      0.19x      0.19x         -
     50000     parallel       24      0.0377      0.0369      0.0384     15.903      0.16x      0.16x         -

     75000       scalar        -      0.0094      0.0094      0.0095     95.824      1.00x      1.00x         -
     75000   sequential        -      0.0094      0.0093      0.0095     96.116      1.00x      1.00x         -
     75000     parallel        1      0.0094      0.0093      0.0095     96.068      1.00x      1.00x         -
     75000     parallel        2      0.0245      0.0244      0.0250     36.748      0.38x      0.38x         -
     75000     parallel        4      0.0193      0.0192      0.0196     46.691      0.49x      0.49x         -
     75000     parallel        8      0.0254      0.0254      0.0256     35.393      0.37x      0.37x         -
     75000     parallel       12      0.0316      0.0315      0.0319     28.477      0.30x      0.30x         -
     75000     parallel       16      0.0329      0.0321      0.0334     27.361      0.29x      0.28x         -
     75000     parallel       24      0.0400      0.0395      0.0406     22.479      0.23x      0.23x         -

    100000       scalar        -      0.0127      0.0126      0.0128     94.511      1.00x      0.99x         -
    100000   sequential        -      0.0126      0.0126      0.0127     95.457      1.01x      1.00x         -
    100000     parallel        1      0.0126      0.0126      0.0127     95.102      1.01x      1.00x         -
    100000     parallel        2      0.0290      0.0290      0.0294     41.314      0.44x      0.43x         -
    100000     parallel        4      0.0217      0.0217      0.0221     55.339      0.59x      0.58x         -
    100000     parallel        8      0.0272      0.0268      0.0290     44.163      0.47x      0.46x         -
    100000     parallel       12      0.0338      0.0335      0.0374     35.460      0.38x      0.37x         -
    100000     parallel       16      0.0352      0.0345      0.0363     34.121      0.36x      0.36x         -
    100000     parallel       24      0.0429      0.0422      0.0432     27.978      0.30x      0.29x         -

    200000       scalar        -      0.0253      0.0252      0.0257     94.991      1.00x      1.15x         -
    200000   sequential        -      0.0291      0.0253      0.0297     82.423      0.87x      1.00x         -
    200000     parallel        1      0.0253      0.0252      0.0270     95.014      1.00x      1.15x         -
    200000     parallel        2      0.0459      0.0458      0.0469     52.329      0.55x      0.63x         -
    200000     parallel        4      0.0318      0.0309      0.0326     75.464      0.79x      0.92x         -
    200000     parallel        8      0.0341      0.0340      0.0346     70.449      0.74x      0.85x         -
    200000     parallel       12      0.0397      0.0387      0.0412     60.468      0.64x      0.73x         -
    200000     parallel       16      0.0425      0.0420      0.0441     56.485      0.59x      0.69x         -
    200000     parallel       24      0.0492      0.0488      0.0506     48.798      0.51x      0.59x         -

    500000       scalar        -      0.0628      0.0627      0.0637     95.613      1.00x      1.01x         -
    500000   sequential        -      0.0633      0.0632      0.0641     94.834      0.99x      1.00x         -
    500000     parallel        1      0.0627      0.0626      0.0629     95.694      1.00x      1.01x         -
    500000     parallel        2      0.0978      0.0970      0.1000     61.375      0.64x      0.65x         -
    500000     parallel        4      0.0562      0.0562      0.0570    106.708      1.12x      1.13x         -
    500000     parallel        8      0.0561      0.0556      0.0631    107.041      1.12x      1.13x         -
    500000     parallel       12      0.0594      0.0566      0.0619    100.936      1.06x      1.06x         -
    500000     parallel       16      0.0710      0.0659      0.0721     84.544      0.88x      0.89x         -
    500000     parallel       24      0.0734      0.0722      0.0749     81.726      0.85x      0.86x         -

   1000000       scalar        -      0.1270      0.1268      0.1345     94.521      1.00x      1.00x         -
   1000000   sequential        -      0.1273      0.1265      0.1389     94.270      1.00x      1.00x         -
   1000000     parallel        1      0.1264      0.1260      0.1327     94.965      1.00x      1.01x         -
   1000000     parallel        2      0.1815      0.1811      0.1825     66.118      0.70x      0.70x         -
   1000000     parallel        4      0.0992      0.0985      0.0996    121.006      1.28x      1.28x         -
   1000000     parallel        8      0.0997      0.0893      0.1328    120.395      1.27x      1.28x         -
   1000000     parallel       12      0.0971      0.0889      0.1098    123.612      1.31x      1.31x         -
   1000000     parallel       16      0.1052      0.0995      0.1077    114.068      1.21x      1.21x         -
   1000000     parallel       24      0.1178      0.1164      0.1194    101.908      1.08x      1.08x         -

   2000000       scalar        -      0.2558      0.2547      0.3061     93.837      1.00x      1.06x         -
   2000000   sequential        -      0.2714      0.2535      0.2998     88.439      0.94x      1.00x         -
   2000000     parallel        1      0.2525      0.2518      0.2671     95.045      1.01x      1.07x         -
   2000000     parallel        2      0.3503      0.3491      0.3593     68.510      0.73x      0.77x         -
   2000000     parallel        4      0.1852      0.1828      0.2145    129.568      1.38x      1.47x         -
   2000000     parallel        8      0.1724      0.1572      0.1851    139.236      1.48x      1.57x         -
   2000000     parallel       12      0.2155      0.2002      0.2451    111.359      1.19x      1.26x         -
   2000000     parallel       16      0.3299      0.2757      0.4769     72.752      0.78x      0.82x         -
   2000000     parallel       24      0.3681      0.2678      0.5393     65.200      0.69x      0.74x         -

   5000000       scalar        -      1.4526      1.4402      1.4640     41.305      1.00x      1.07x         -
   5000000   sequential        -      1.5496      1.5008      1.6998     38.720      0.94x      1.00x         -
   5000000     parallel        1      1.5507      1.5214      1.6699     38.693      0.94x      1.00x         -
   5000000     parallel        2      1.4402      1.4108      1.6606     41.662      1.01x      1.08x         -
   5000000     parallel        4      1.2264      1.1685      1.3181     48.924      1.18x      1.26x         -
   5000000     parallel        8      0.8184      0.7715      1.5937     73.314      1.77x      1.89x         -
   5000000     parallel       12      1.1722      1.0926      1.2536     51.188      1.24x      1.32x         -
   5000000     parallel       16      1.1818      0.9816      1.2921     50.768      1.23x      1.31x         -
   5000000     parallel       24      1.2142      1.1588      1.2757     49.415      1.20x      1.28x         -

  10000000       scalar        -      3.6142      3.5521      3.7914     33.202      1.00x      1.04x         -
  10000000   sequential        -      3.7475      3.7162      4.1593     32.021      0.96x      1.00x         -
  10000000     parallel        1      3.6463      3.5646      4.0406     32.910      0.99x      1.03x         -
  10000000     parallel        2      3.4235      3.2953      3.6235     35.052      1.06x      1.09x         -
  10000000     parallel        4      3.2705      3.2141      3.4370     36.692      1.11x      1.15x         -
  10000000     parallel        8      2.8730      2.5338      3.2445     41.768      1.26x      1.30x         -
  10000000     parallel       12      3.1517      2.9868      3.3828     38.075      1.15x      1.19x         -
  10000000     parallel       16      3.6276      3.3199      3.8790     33.080      1.00x      1.03x         -
  10000000     parallel       24      3.5168      3.4025      3.6139     34.122      1.03x      1.07x         -

  33554432       scalar        -     14.8993     14.8079     15.2640     27.025      1.00x      1.02x      1.05
  33554432   sequential        -     15.1480     14.9015     16.0298     26.581      0.98x      1.00x      1.02
  33554432     parallel        1     15.0458     14.7964     16.3189     26.762      0.99x      1.01x      1.13
  33554432     parallel        2     13.9938     13.8554     15.2127     28.774      1.06x      1.08x      1.96
  33554432     parallel        4     14.5830     14.2994     15.5133     27.611      1.02x      1.04x      3.91
  33554432     parallel        8     13.4698     13.3040     14.4558     29.893      1.11x      1.12x      8.03
  33554432     parallel       12     13.8725     13.4325     14.3735     29.025      1.07x      1.09x     11.99
  33554432     parallel       16     14.2523     13.8240     14.9115     28.252      1.05x      1.06x     15.17
  33554432     parallel       24     14.1619     13.7915     14.5498     28.432      1.05x      1.07x     25.07


== div ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0003      0.0003      0.0003     46.791      1.00x      1.00x         -
      1000   sequential        -      0.0003      0.0003      0.0003     46.811      1.00x      1.00x         -
      1000     parallel        1      0.0003      0.0003      0.0003     46.937      1.00x      1.00x         -
      1000     parallel        2      0.0003      0.0003      0.0003     46.843      1.00x      1.00x         -
      1000     parallel        4      0.0003      0.0002      0.0003     47.004      1.00x      1.00x         -
      1000     parallel        8      0.0003      0.0003      0.0003     46.928      1.00x      1.00x         -
      1000     parallel       12      0.0003      0.0003      0.0003     46.984      1.00x      1.00x         -
      1000     parallel       16      0.0003      0.0003      0.0003     47.002      1.00x      1.00x         -
      1000     parallel       24      0.0003      0.0003      0.0003     47.033      1.01x      1.00x         -

     10000       scalar        -      0.0025      0.0025      0.0025     47.925      1.00x      1.00x         -
     10000   sequential        -      0.0025      0.0025      0.0025     47.923      1.00x      1.00x         -
     10000     parallel        1      0.0023      0.0023      0.0023     51.971      1.08x      1.08x         -
     10000     parallel        2      0.0023      0.0023      0.0023     51.925      1.08x      1.08x         -
     10000     parallel        4      0.0023      0.0023      0.0023     52.017      1.09x      1.09x         -
     10000     parallel        8      0.0023      0.0023      0.0024     52.256      1.09x      1.09x         -
     10000     parallel       12      0.0023      0.0023      0.0024     51.722      1.08x      1.08x         -
     10000     parallel       16      0.0023      0.0023      0.0024     52.201      1.09x      1.09x         -
     10000     parallel       24      0.0023      0.0023      0.0023     51.842      1.08x      1.08x         -

     32768       scalar        -      0.0075      0.0075      0.0078     52.507      1.00x      1.07x         -
     32768   sequential        -      0.0080      0.0078      0.0082     49.154      0.94x      1.00x         -
     32768     parallel        1      0.0081      0.0080      0.0082     48.691      0.93x      0.99x         -
     32768     parallel        2      0.0080      0.0080      0.0081     48.873      0.93x      0.99x         -
     32768     parallel        4      0.0081      0.0080      0.0081     48.821      0.93x      0.99x         -
     32768     parallel        8      0.0081      0.0080      0.0081     48.823      0.93x      0.99x         -
     32768     parallel       12      0.0080      0.0080      0.0081     48.866      0.93x      0.99x         -
     32768     parallel       16      0.0080      0.0080      0.0081     48.908      0.93x      0.99x         -
     32768     parallel       24      0.0080      0.0080      0.0081     48.861      0.93x      0.99x         -

     32769       scalar        -      0.0075      0.0075      0.0076     52.236      1.00x      1.00x         -
     32769   sequential        -      0.0075      0.0074      0.0106     52.462      1.00x      1.00x         -
     32769     parallel        1      0.0081      0.0080      0.0089     48.692      0.93x      0.93x         -
     32769     parallel        2      0.0269      0.0268      0.0275     14.631      0.28x      0.28x         -
     32769     parallel        4      0.0206      0.0205      0.0209     19.117      0.37x      0.36x         -
     32769     parallel        8      0.0240      0.0238      0.0241     16.400      0.31x      0.31x         -
     32769     parallel       12      0.0289      0.0286      0.0291     13.622      0.26x      0.26x         -
     32769     parallel       16      0.0305      0.0302      0.0307     12.890      0.25x      0.25x         -
     32769     parallel       24      0.0395      0.0390      0.0405      9.945      0.19x      0.19x         -

     40000       scalar        -      0.0097      0.0095      0.0100     49.427      1.00x      1.01x         -
     40000   sequential        -      0.0098      0.0098      0.0099     48.911      0.99x      1.00x         -
     40000     parallel        1      0.0091      0.0090      0.0092     52.815      1.07x      1.08x         -
     40000     parallel        2      0.0309      0.0307      0.0316     15.550      0.31x      0.32x         -
     40000     parallel        4      0.0226      0.0225      0.0238     21.280      0.43x      0.44x         -
     40000     parallel        8      0.0254      0.0253      0.0256     18.927      0.38x      0.39x         -
     40000     parallel       12      0.0293      0.0291      0.0298     16.387      0.33x      0.34x         -
     40000     parallel       16      0.0317      0.0312      0.0332     15.136      0.31x      0.31x         -
     40000     parallel       24      0.0403      0.0400      0.0412     11.914      0.24x      0.24x         -

     50000       scalar        -      0.0116      0.0115      0.0117     51.815      1.00x      1.02x         -
     50000   sequential        -      0.0118      0.0117      0.0119     50.928      0.98x      1.00x         -
     50000     parallel        1      0.0123      0.0122      0.0124     48.864      0.94x      0.96x         -
     50000     parallel        2      0.0359      0.0356      0.0363     16.716      0.32x      0.33x         -
     50000     parallel        4      0.0250      0.0249      0.0252     23.974      0.46x      0.47x         -
     50000     parallel        8      0.0263      0.0260      0.0271     22.787      0.44x      0.45x         -
     50000     parallel       12      0.0308      0.0305      0.0312     19.486      0.38x      0.38x         -
     50000     parallel       16      0.0320      0.0318      0.0327     18.745      0.36x      0.37x         -
     50000     parallel       24      0.0417      0.0411      0.0425     14.386      0.28x      0.28x         -

     75000       scalar        -      0.0173      0.0172      0.0174     52.058      1.00x      1.00x         -
     75000   sequential        -      0.0173      0.0172      0.0175     51.993      1.00x      1.00x         -
     75000     parallel        1      0.0175      0.0173      0.0177     51.337      0.99x      0.99x         -
     75000     parallel        2      0.0493      0.0489      0.0497     18.258      0.35x      0.35x         -
     75000     parallel        4      0.0315      0.0312      0.0333     28.573      0.55x      0.55x         -
     75000     parallel        8      0.0293      0.0292      0.0297     30.669      0.59x      0.59x         -
     75000     parallel       12      0.0332      0.0327      0.0371     27.097      0.52x      0.52x         -
     75000     parallel       16      0.0345      0.0343      0.0349     26.057      0.50x      0.50x         -
     75000     parallel       24      0.0438      0.0435      0.0447     20.535      0.39x      0.39x         -

    100000       scalar        -      0.0230      0.0228      0.0232     52.192      1.00x      1.01x         -
    100000   sequential        -      0.0232      0.0230      0.0236     51.641      0.99x      1.00x         -
    100000     parallel        1      0.0235      0.0230      0.0238     51.161      0.98x      0.99x         -
    100000     parallel        2      0.0621      0.0620      0.0624     19.329      0.37x      0.37x         -
    100000     parallel        4      0.0379      0.0379      0.0388     31.628      0.61x      0.61x         -
    100000     parallel        8      0.0326      0.0323      0.0333     36.819      0.71x      0.71x         -
    100000     parallel       12      0.0366      0.0345      0.0391     32.801      0.63x      0.64x         -
    100000     parallel       16      0.0383      0.0376      0.0418     31.360      0.60x      0.61x         -
    100000     parallel       24      0.0467      0.0460      0.0493     25.669      0.49x      0.50x         -

    200000       scalar        -      0.0459      0.0457      0.0466     52.307      1.00x      1.00x         -
    200000   sequential        -      0.0459      0.0457      0.0460     52.313      1.00x      1.00x         -
    200000     parallel        1      0.0463      0.0459      0.0475     51.829      0.99x      0.99x         -
    200000     parallel        2      0.1112      0.1099      0.1124     21.574      0.41x      0.41x         -
    200000     parallel        4      0.0649      0.0648      0.0654     36.987      0.71x      0.71x         -
    200000     parallel        8      0.0452      0.0449      0.0467     53.105      1.02x      1.02x         -
    200000     parallel       12      0.0459      0.0450      0.0505     52.330      1.00x      1.00x         -
    200000     parallel       16      0.0506      0.0504      0.0519     47.473      0.91x      0.91x         -
    200000     parallel       24      0.0517      0.0510      0.0525     46.399      0.89x      0.89x         -

    500000       scalar        -      0.1149      0.1143      0.1163     52.236      1.00x      1.00x         -
    500000   sequential        -      0.1149      0.1138      0.1169     52.199      1.00x      1.00x         -
    500000     parallel        1      0.1167      0.1138      0.1190     51.396      0.98x      0.98x         -
    500000     parallel        2      0.2538      0.2521      0.2552     23.640      0.45x      0.45x         -
    500000     parallel        4      0.1376      0.1369      0.1411     43.621      0.84x      0.84x         -
    500000     parallel        8      0.0843      0.0824      0.0861     71.180      1.36x      1.36x         -
    500000     parallel       12      0.0799      0.0746      0.1097     75.114      1.44x      1.44x         -
    500000     parallel       16      0.0884      0.0880      0.0887     67.866      1.30x      1.30x         -
    500000     parallel       24      0.0775      0.0766      0.0786     77.463      1.48x      1.48x         -

   1000000       scalar        -      0.2286      0.2281      0.2306     52.505      1.00x      1.00x         -
   1000000   sequential        -      0.2282      0.2275      0.2296     52.596      1.00x      1.00x         -
   1000000     parallel        1      0.2313      0.2283      0.2336     51.875      0.99x      0.99x         -
   1000000     parallel        2      0.4940      0.4914      0.4974     24.291      0.46x      0.46x         -
   1000000     parallel        4      0.2595      0.2583      0.2619     46.247      0.88x      0.88x         -
   1000000     parallel        8      0.1443      0.1439      0.1457     83.171      1.58x      1.58x         -
   1000000     parallel       12      0.1219      0.1195      0.1255     98.477      1.88x      1.87x         -
   1000000     parallel       16      0.1466      0.1462      0.1504     81.859      1.56x      1.56x         -
   1000000     parallel       24      0.1213      0.1177      0.1270     98.918      1.88x      1.88x         -

   2000000       scalar        -      0.4787      0.4532      0.5024     50.136      1.00x      0.96x         -
   2000000   sequential        -      0.4592      0.4577      0.4765     52.270      1.04x      1.00x         -
   2000000     parallel        1      0.4621      0.4595      0.4682     51.940      1.04x      0.99x         -
   2000000     parallel        2      0.9747      0.9620      0.9827     24.624      0.49x      0.47x         -
   2000000     parallel        4      0.5076      0.4973      0.5601     47.279      0.94x      0.90x         -
   2000000     parallel        8      0.2685      0.2671      0.2857     89.394      1.78x      1.71x         -
   2000000     parallel       12      0.2374      0.2095      0.2635    101.074      2.02x      1.93x         -
   2000000     parallel       16      0.2637      0.2620      0.3428     91.004      1.82x      1.74x         -
   2000000     parallel       24      0.2074      0.2026      0.2149    115.704      2.31x      2.21x         -

   5000000       scalar        -      1.5434      1.5313      1.5663     38.875      1.00x      1.01x         -
   5000000   sequential        -      1.5614      1.5338      1.5968     38.428      0.99x      1.00x         -
   5000000     parallel        1      1.5844      1.5309      1.7657     37.869      0.97x      0.99x         -
   5000000     parallel        2      2.5675      2.5473      2.6506     23.369      0.60x      0.61x         -
   5000000     parallel        4      1.4649      1.3480      1.8465     40.958      1.05x      1.07x         -
   5000000     parallel        8      0.7332      0.6812      0.7961     81.836      2.11x      2.13x         -
   5000000     parallel       12      1.1989      0.8677      1.3361     50.046      1.29x      1.30x         -
   5000000     parallel       16      1.4399      1.2110      1.5847     41.669      1.07x      1.08x         -
   5000000     parallel       24      1.2934      1.1653      1.3301     46.388      1.19x      1.21x         -

  10000000       scalar        -      3.6499      3.5600      3.9213     32.878      1.00x      1.02x         -
  10000000   sequential        -      3.7388      3.5675      4.6726     32.096      0.98x      1.00x         -
  10000000     parallel        1      3.6640      3.5947      4.2252     32.751      1.00x      1.02x         -
  10000000     parallel        2      5.1503      5.1276      5.3077     23.300      0.71x      0.73x         -
  10000000     parallel        4      3.3065      3.2695      4.2493     36.292      1.10x      1.13x         -
  10000000     parallel        8      2.7616      2.7145      3.0544     43.453      1.32x      1.35x         -
  10000000     parallel       12      3.1536      3.0546      3.2787     38.052      1.16x      1.19x         -
  10000000     parallel       16      3.3813      3.3363      3.5050     35.489      1.08x      1.11x         -
  10000000     parallel       24      3.3104      3.1945      3.5239     36.249      1.10x      1.13x         -

  33554432       scalar        -     14.8278     14.6508     15.8024     27.155      1.00x      1.00x      0.93
  33554432   sequential        -     14.8641     14.7893     14.9946     27.089      1.00x      1.00x      0.82
  33554432     parallel        1     14.7964     14.7163     15.4958     27.213      1.00x      1.00x      0.93
  33554432     parallel        2     18.1635     18.0500     19.1737     22.168      0.82x      0.82x      2.28
  33554432     parallel        4     14.4735     14.3916     14.9300     27.820      1.02x      1.03x      4.31
  33554432     parallel        8     13.8805     13.3920     14.5828     29.009      1.07x      1.07x      6.85
  33554432     parallel       12     13.5094     13.0382     13.8709     29.805      1.10x      1.10x     11.61
  33554432     parallel       16     14.3175     14.0262     14.9777     28.123      1.04x      1.04x     12.08
  33554432     parallel       24     13.9509     13.8284     14.2245     28.862      1.06x      1.07x     20.97


== neg ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    110.848      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    110.862      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    110.862      1.00x      1.00x         -
      1000     parallel        2      0.0001      0.0001      0.0001    110.778      1.00x      1.00x         -
      1000     parallel        4      0.0001      0.0001      0.0001    110.787      1.00x      1.00x         -
      1000     parallel        8      0.0001      0.0001      0.0001    110.778      1.00x      1.00x         -
      1000     parallel       12      0.0001      0.0001      0.0001    110.801      1.00x      1.00x         -
      1000     parallel       16      0.0001      0.0001      0.0001    110.656      1.00x      1.00x         -
      1000     parallel       24      0.0001      0.0001      0.0001    110.824      1.00x      1.00x         -

     10000       scalar        -      0.0007      0.0007      0.0007    109.416      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    109.405      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0007    108.597      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0007    108.611      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0007    108.626      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0007    108.593      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0007    108.618      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0007    108.611      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0008    108.593      0.99x      0.99x         -

     32768       scalar        -      0.0024      0.0024      0.0024    110.608      1.00x      1.00x         -
     32768   sequential        -      0.0024      0.0024      0.0024    110.345      1.00x      1.00x         -
     32768     parallel        1      0.0024      0.0024      0.0024    110.317      1.00x      1.00x         -
     32768     parallel        2      0.0024      0.0024      0.0024    110.340      1.00x      1.00x         -
     32768     parallel        4      0.0024      0.0024      0.0024    110.331      1.00x      1.00x         -
     32768     parallel        8      0.0024      0.0024      0.0024    110.336      1.00x      1.00x         -
     32768     parallel       12      0.0024      0.0024      0.0024    110.336      1.00x      1.00x         -
     32768     parallel       16      0.0024      0.0024      0.0024    110.345      1.00x      1.00x         -
     32768     parallel       24      0.0024      0.0024      0.0024    110.249      1.00x      1.00x         -

     32769       scalar        -      0.0024      0.0024      0.0024    110.575      1.00x      1.00x         -
     32769   sequential        -      0.0024      0.0024      0.0024    110.566      1.00x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0024    110.312      1.00x      1.00x         -
     32769     parallel        2      0.0155      0.0155      0.0156     16.901      0.15x      0.15x         -
     32769     parallel        4      0.0148      0.0148      0.0157     17.672      0.16x      0.16x         -
     32769     parallel        8      0.0222      0.0222      0.0224     11.788      0.11x      0.11x         -
     32769     parallel       12      0.0278      0.0274      0.0282      9.430      0.09x      0.09x         -
     32769     parallel       16      0.0300      0.0299      0.0306      8.742      0.08x      0.08x         -
     32769     parallel       24      0.0366      0.0360      0.0369      7.166      0.06x      0.06x         -

     40000       scalar        -      0.0029      0.0029      0.0030    109.530      1.00x      0.99x         -
     40000   sequential        -      0.0029      0.0029      0.0029    110.308      1.01x      1.00x         -
     40000     parallel        1      0.0029      0.0029      0.0029    109.114      1.00x      0.99x         -
     40000     parallel        2      0.0166      0.0165      0.0168     19.288      0.18x      0.17x         -
     40000     parallel        4      0.0153      0.0152      0.0155     20.959      0.19x      0.19x         -
     40000     parallel        8      0.0244      0.0222      0.0259     13.136      0.12x      0.12x         -
     40000     parallel       12      0.0283      0.0281      0.0290     11.301      0.10x      0.10x         -
     40000     parallel       16      0.0306      0.0304      0.0311     10.441      0.10x      0.09x         -
     40000     parallel       24      0.0367      0.0360      0.0378      8.729      0.08x      0.08x         -

     50000       scalar        -      0.0036      0.0036      0.0037    110.196      1.00x      1.02x         -
     50000   sequential        -      0.0037      0.0037      0.0037    107.747      0.98x      1.00x         -
     50000     parallel        1      0.0036      0.0036      0.0037    110.087      1.00x      1.02x         -
     50000     parallel        2      0.0181      0.0181      0.0183     22.111      0.20x      0.21x         -
     50000     parallel        4      0.0161      0.0160      0.0166     24.899      0.23x      0.23x         -
     50000     parallel        8      0.0233      0.0232      0.0241     17.135      0.16x      0.16x         -
     50000     parallel       12      0.0293      0.0290      0.0296     13.674      0.12x      0.13x         -
     50000     parallel       16      0.0311      0.0304      0.0330     12.860      0.12x      0.12x         -
     50000     parallel       24      0.0375      0.0368      0.0398     10.664      0.10x      0.10x         -

     75000       scalar        -      0.0061      0.0061      0.0061     98.966      1.00x      1.00x         -
     75000   sequential        -      0.0061      0.0061      0.0061     98.969      1.00x      1.00x         -
     75000     parallel        1      0.0061      0.0061      0.0062     97.928      0.99x      0.99x         -
     75000     parallel        2      0.0218      0.0218      0.0220     27.538      0.28x      0.28x         -
     75000     parallel        4      0.0188      0.0185      0.0192     31.975      0.32x      0.32x         -
     75000     parallel        8      0.0247      0.0244      0.0271     24.323      0.25x      0.25x         -
     75000     parallel       12      0.0314      0.0310      0.0320     19.108      0.19x      0.19x         -
     75000     parallel       16      0.0328      0.0322      0.0335     18.273      0.18x      0.18x         -
     75000     parallel       24      0.0391      0.0385      0.0398     15.360      0.16x      0.16x         -

    100000       scalar        -      0.0082      0.0082      0.0083     97.181      1.00x      1.00x         -
    100000   sequential        -      0.0083      0.0083      0.0083     96.741      1.00x      1.00x         -
    100000     parallel        1      0.0084      0.0083      0.0085     95.750      0.99x      0.99x         -
    100000     parallel        2      0.0261      0.0260      0.0269     30.686      0.32x      0.32x         -
    100000     parallel        4      0.0204      0.0196      0.0215     39.169      0.40x      0.40x         -
    100000     parallel        8      0.0301      0.0294      0.0306     26.579      0.27x      0.27x         -
    100000     parallel       12      0.0328      0.0323      0.0333     24.416      0.25x      0.25x         -
    100000     parallel       16      0.0339      0.0334      0.0360     23.610      0.24x      0.24x         -
    100000     parallel       24      0.0386      0.0383      0.0395     20.720      0.21x      0.21x         -

    200000       scalar        -      0.0168      0.0168      0.0206     95.123      1.00x      1.00x         -
    200000   sequential        -      0.0168      0.0167      0.0169     95.212      1.00x      1.00x         -
    200000     parallel        1      0.0168      0.0168      0.0170     95.123      1.00x      1.00x         -
    200000     parallel        2      0.0409      0.0408      0.0412     39.108      0.41x      0.41x         -
    200000     parallel        4      0.0280      0.0277      0.0286     57.152      0.60x      0.60x         -
    200000     parallel        8      0.0339      0.0337      0.0349     47.239      0.50x      0.50x         -
    200000     parallel       12      0.0393      0.0391      0.0396     40.711      0.43x      0.43x         -
    200000     parallel       16      0.0431      0.0418      0.0438     37.155      0.39x      0.39x         -
    200000     parallel       24      0.0464      0.0457      0.0480     34.487      0.36x      0.36x         -

    500000       scalar        -      0.0416      0.0415      0.0420     96.251      1.00x      1.01x         -
    500000   sequential        -      0.0421      0.0420      0.0425     95.111      0.99x      1.00x         -
    500000     parallel        1      0.0418      0.0416      0.0426     95.615      0.99x      1.01x         -
    500000     parallel        2      0.0843      0.0841      0.0849     47.462      0.49x      0.50x         -
    500000     parallel        4      0.0503      0.0502      0.0509     79.518      0.83x      0.84x         -
    500000     parallel        8      0.0551      0.0550      0.0557     72.544      0.75x      0.76x         -
    500000     parallel       12      0.0597      0.0588      0.0622     67.007      0.70x      0.70x         -
    500000     parallel       16      0.0684      0.0666      0.0700     58.520      0.61x      0.62x         -
    500000     parallel       24      0.0728      0.0687      0.0793     54.943      0.57x      0.58x         -

   1000000       scalar        -      0.0840      0.0835      0.0897     95.266      1.00x      1.01x         -
   1000000   sequential        -      0.0849      0.0845      0.0905     94.284      0.99x      1.00x         -
   1000000     parallel        1      0.0839      0.0837      0.0913     95.320      1.00x      1.01x         -
   1000000     parallel        2      0.1564      0.1561      0.1721     51.157      0.54x      0.54x         -
   1000000     parallel        4      0.0874      0.0860      0.0906     91.579      0.96x      0.97x         -
   1000000     parallel        8      0.1074      0.0963      0.1224     74.479      0.78x      0.79x         -
   1000000     parallel       12      0.0970      0.0934      0.1026     82.466      0.87x      0.87x         -
   1000000     parallel       16      0.1119      0.1042      0.1374     71.504      0.75x      0.76x         -
   1000000     parallel       24      0.1137      0.1096      0.1155     70.374      0.74x      0.75x         -

   2000000       scalar        -      0.1675      0.1673      0.1850     95.512      1.00x      1.00x         -
   2000000   sequential        -      0.1678      0.1665      0.1810     95.345      1.00x      1.00x         -
   2000000     parallel        1      0.1693      0.1683      0.1923     94.482      0.99x      0.99x         -
   2000000     parallel        2      0.3004      0.2994      0.3176     53.267      0.56x      0.56x         -
   2000000     parallel        4      0.1727      0.1580      0.1908     92.656      0.97x      0.97x         -
   2000000     parallel        8      0.1620      0.1606      0.1875     98.746      1.03x      1.04x         -
   2000000     parallel       12      0.1841      0.1734      0.2181     86.912      0.91x      0.91x         -
   2000000     parallel       16      0.2321      0.1946      0.3444     68.949      0.72x      0.72x         -
   2000000     parallel       24      0.2405      0.2247      0.2659     66.528      0.70x      0.70x         -

   5000000       scalar        -      1.0161      0.9613      1.1334     39.366      1.00x      1.02x         -
   5000000   sequential        -      1.0372      1.0153      1.1365     38.567      0.98x      1.00x         -
   5000000     parallel        1      0.9770      0.9496      1.4386     40.944      1.04x      1.06x         -
   5000000     parallel        2      0.9425      0.8848      0.9757     42.441      1.08x      1.10x         -
   5000000     parallel        4      0.6041      0.5443      0.8130     66.211      1.68x      1.72x         -
   5000000     parallel        8      0.6326      0.5505      0.7691     63.234      1.61x      1.64x         -
   5000000     parallel       12      0.5711      0.5353      1.0794     70.043      1.78x      1.82x         -
   5000000     parallel       16      0.6018      0.5169      0.9471     66.462      1.69x      1.72x         -
   5000000     parallel       24      0.6170      0.5339      0.7867     64.830      1.65x      1.68x         -

  10000000       scalar        -      2.4990      2.4400      2.6330     32.013      1.00x      1.04x         -
  10000000   sequential        -      2.6112      2.5203      3.0767     30.637      0.96x      1.00x         -
  10000000     parallel        1      2.5106      2.4165      2.7146     31.865      1.00x      1.04x         -
  10000000     parallel        2      2.1502      2.1293      2.2438     37.206      1.16x      1.21x         -
  10000000     parallel        4      2.0055      1.9611      2.4637     39.890      1.25x      1.30x         -
  10000000     parallel        8      1.4788      1.3213      2.3115     54.098      1.69x      1.77x         -
  10000000     parallel       12      2.2035      2.0071      2.3226     36.306      1.13x      1.19x         -
  10000000     parallel       16      2.1330      1.8933      2.2031     37.505      1.17x      1.22x         -
  10000000     parallel       24      2.3081      2.1436      2.4599     34.661      1.08x      1.13x         -

  33554432       scalar        -     10.9031     10.8161     11.7680     24.620      1.00x      0.98x         -
  33554432   sequential        -     10.6886     10.5357     11.7514     25.114      1.02x      1.00x         -
  33554432     parallel        1     11.0044     10.7949     11.9419     24.393      0.99x      0.97x      1.09
  33554432     parallel        2     10.4161     10.2446     11.4462     25.771      1.05x      1.03x         -
  33554432     parallel        4     10.4924      9.9631     11.0668     25.584      1.04x      1.02x         -
  33554432     parallel        8      9.2953      9.0831      9.9254     28.879      1.17x      1.15x         -
  33554432     parallel       12     10.0420      9.7588     10.3987     26.731      1.09x      1.06x         -
  33554432     parallel       16      9.9880      9.8603     10.6054     26.876      1.09x      1.07x         -
  33554432     parallel       24     10.3875      9.9694     10.5644     25.842      1.05x      1.03x         -


== abs ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    110.885      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    110.885      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    110.297      0.99x      0.99x         -
      1000     parallel        2      0.0001      0.0001      0.0001    110.731      1.00x      1.00x         -
      1000     parallel        4      0.0001      0.0001      0.0001    110.881      1.00x      1.00x         -
      1000     parallel        8      0.0001      0.0001      0.0001    110.885      1.00x      1.00x         -
      1000     parallel       12      0.0001      0.0001      0.0001    110.745      1.00x      1.00x         -
      1000     parallel       16      0.0001      0.0001      0.0001    110.881      1.00x      1.00x         -
      1000     parallel       24      0.0001      0.0001      0.0001    110.881      1.00x      1.00x         -

     10000       scalar        -      0.0007      0.0007      0.0007    109.365      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    109.318      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0007    108.561      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0008    108.550      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0008    108.564      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0007    108.546      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0007    108.421      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0008    108.525      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0007    108.536      0.99x      0.99x         -

     32768       scalar        -      0.0024      0.0024      0.0024    108.634      1.00x      0.98x         -
     32768   sequential        -      0.0024      0.0024      0.0025    110.327      1.02x      1.00x         -
     32768     parallel        1      0.0024      0.0024      0.0025    108.201      1.00x      0.98x         -
     32768     parallel        2      0.0024      0.0024      0.0024    108.393      1.00x      0.98x         -
     32768     parallel        4      0.0024      0.0024      0.0024    108.284      1.00x      0.98x         -
     32768     parallel        8      0.0024      0.0024      0.0025    108.358      1.00x      0.98x         -
     32768     parallel       12      0.0024      0.0024      0.0024    108.327      1.00x      0.98x         -
     32768     parallel       16      0.0024      0.0024      0.0025    108.402      1.00x      0.98x         -
     32768     parallel       24      0.0024      0.0024      0.0025    108.358      1.00x      0.98x         -

     32769       scalar        -      0.0024      0.0024      0.0025    109.075      1.00x      0.99x         -
     32769   sequential        -      0.0024      0.0024      0.0024    110.316      1.01x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0025    108.920      1.00x      0.99x         -
     32769     parallel        2      0.0144      0.0143      0.0154     18.197      0.17x      0.16x         -
     32769     parallel        4      0.0148      0.0147      0.0310     17.664      0.16x      0.16x         -
     32769     parallel        8      0.0223      0.0222      0.0226     11.781      0.11x      0.11x         -
     32769     parallel       12      0.0278      0.0277      0.0281      9.429      0.09x      0.09x         -
     32769     parallel       16      0.0305      0.0302      0.0307      8.597      0.08x      0.08x         -
     32769     parallel       24      0.0362      0.0353      0.0366      7.235      0.07x      0.07x         -

     40000       scalar        -      0.0029      0.0029      0.0030    109.354      1.00x      1.04x         -
     40000   sequential        -      0.0031      0.0030      0.0031    104.878      0.96x      1.00x         -
     40000     parallel        1      0.0029      0.0029      0.0030    108.976      1.00x      1.04x         -
     40000     parallel        2      0.0162      0.0161      0.0163     19.813      0.18x      0.19x         -
     40000     parallel        4      0.0152      0.0150      0.0155     21.050      0.19x      0.20x         -
     40000     parallel        8      0.0227      0.0227      0.0231     14.097      0.13x      0.13x         -
     40000     parallel       12      0.0282      0.0282      0.0288     11.332      0.10x      0.11x         -
     40000     parallel       16      0.0305      0.0303      0.0309     10.489      0.10x      0.10x         -
     40000     parallel       24      0.0368      0.0362      0.0381      8.701      0.08x      0.08x         -

     50000       scalar        -      0.0037      0.0036      0.0037    109.577      1.00x      1.00x         -
     50000   sequential        -      0.0037      0.0036      0.0037    109.451      1.00x      1.00x         -
     50000     parallel        1      0.0037      0.0036      0.0037    109.536      1.00x      1.00x         -
     50000     parallel        2      0.0176      0.0176      0.0193     22.693      0.21x      0.21x         -
     50000     parallel        4      0.0158      0.0157      0.0161     25.255      0.23x      0.23x         -
     50000     parallel        8      0.0233      0.0232      0.0239     17.142      0.16x      0.16x         -
     50000     parallel       12      0.0296      0.0291      0.0304     13.494      0.12x      0.12x         -
     50000     parallel       16      0.0319      0.0314      0.0328     12.534      0.11x      0.11x         -
     50000     parallel       24      0.0372      0.0365      0.0375     10.766      0.10x      0.10x         -

     75000       scalar        -      0.0060      0.0060      0.0060    100.626      1.00x      1.01x         -
     75000   sequential        -      0.0060      0.0060      0.0061     99.411      0.99x      1.00x         -
     75000     parallel        1      0.0061      0.0061      0.0062     98.648      0.98x      0.99x         -
     75000     parallel        2      0.0216      0.0214      0.0218     27.744      0.28x      0.28x         -
     75000     parallel        4      0.0178      0.0176      0.0182     33.683      0.33x      0.34x         -
     75000     parallel        8      0.0251      0.0250      0.0253     23.918      0.24x      0.24x         -
     75000     parallel       12      0.0314      0.0311      0.0327     19.137      0.19x      0.19x         -
     75000     parallel       16      0.0329      0.0323      0.0331     18.234      0.18x      0.18x         -
     75000     parallel       24      0.0385      0.0377      0.0389     15.583      0.15x      0.16x         -

    100000       scalar        -      0.0083      0.0082      0.0083     96.947      1.00x      1.00x         -
    100000   sequential        -      0.0083      0.0083      0.0083     96.782      1.00x      1.00x         -
    100000     parallel        1      0.0083      0.0083      0.0084     96.422      0.99x      1.00x         -
    100000     parallel        2      0.0255      0.0254      0.0259     31.367      0.32x      0.32x         -
    100000     parallel        4      0.0195      0.0194      0.0200     41.050      0.42x      0.42x         -
    100000     parallel        8      0.0266      0.0265      0.0286     30.088      0.31x      0.31x         -
    100000     parallel       12      0.0336      0.0334      0.0346     23.796      0.25x      0.25x         -
    100000     parallel       16      0.0344      0.0340      0.0351     23.235      0.24x      0.24x         -
    100000     parallel       24      0.0394      0.0388      0.0414     20.291      0.21x      0.21x         -

    200000       scalar        -      0.0168      0.0167      0.0171     95.340      1.00x      1.00x         -
    200000   sequential        -      0.0167      0.0167      0.0171     95.571      1.00x      1.00x         -
    200000     parallel        1      0.0168      0.0168      0.0171     95.013      1.00x      0.99x         -
    200000     parallel        2      0.0409      0.0407      0.0411     39.099      0.41x      0.41x         -
    200000     parallel        4      0.0277      0.0276      0.0280     57.767      0.61x      0.60x         -
    200000     parallel        8      0.0340      0.0339      0.0343     47.061      0.49x      0.49x         -
    200000     parallel       12      0.0395      0.0389      0.0405     40.498      0.42x      0.42x         -
    200000     parallel       16      0.0427      0.0423      0.0432     37.458      0.39x      0.39x         -
    200000     parallel       24      0.0467      0.0455      0.0488     34.265      0.36x      0.36x         -

    500000       scalar        -      0.0416      0.0416      0.0421     96.085      1.00x      1.01x         -
    500000   sequential        -      0.0422      0.0421      0.0425     94.825      0.99x      1.00x         -
    500000     parallel        1      0.0418      0.0417      0.0422     95.665      1.00x      1.01x         -
    500000     parallel        2      0.0839      0.0838      0.0881     47.653      0.50x      0.50x         -
    500000     parallel        4      0.0499      0.0497      0.0526     80.083      0.83x      0.84x         -
    500000     parallel        8      0.0614      0.0601      0.0694     65.175      0.68x      0.69x         -
    500000     parallel       12      0.0556      0.0523      0.0610     71.975      0.75x      0.76x         -
    500000     parallel       16      0.0662      0.0653      0.0668     60.437      0.63x      0.64x         -
    500000     parallel       24      0.0719      0.0709      0.0726     55.604      0.58x      0.59x         -

   1000000       scalar        -      0.0834      0.0831      0.0888     95.884      1.00x      1.01x         -
   1000000   sequential        -      0.0839      0.0838      0.0898     95.334      0.99x      1.00x         -
   1000000     parallel        1      0.0839      0.0836      0.0892     95.323      0.99x      1.00x         -
   1000000     parallel        2      0.1560      0.1558      0.1584     51.278      0.53x      0.54x         -
   1000000     parallel        4      0.0857      0.0855      0.0928     93.349      0.97x      0.98x         -
   1000000     parallel        8      0.0904      0.0900      0.0915     88.535      0.92x      0.93x         -
   1000000     parallel       12      0.0979      0.0937      0.1002     81.740      0.85x      0.86x         -
   1000000     parallel       16      0.1116      0.1083      0.1169     71.685      0.75x      0.75x         -
   1000000     parallel       24      0.1132      0.1058      0.1153     70.654      0.74x      0.74x         -

   2000000       scalar        -      0.1694      0.1675      0.1872     94.461      1.00x      1.02x         -
   2000000   sequential        -      0.1734      0.1666      0.1897     92.272      0.98x      1.00x         -
   2000000     parallel        1      0.1691      0.1681      0.1907     94.640      1.00x      1.03x         -
   2000000     parallel        2      0.3010      0.3002      0.3033     53.154      0.56x      0.58x         -
   2000000     parallel        4      0.1613      0.1580      0.1779     99.213      1.05x      1.08x         -
   2000000     parallel        8      0.1662      0.1603      0.1704     96.262      1.02x      1.04x         -
   2000000     parallel       12      0.2224      0.1686      0.2469     71.928      0.76x      0.78x         -
   2000000     parallel       16      0.1838      0.1729      0.1929     87.045      0.92x      0.94x         -
   2000000     parallel       24      0.2005      0.1792      0.2277     79.793      0.84x      0.86x         -

   5000000       scalar        -      0.9734      0.9613      1.0520     41.093      1.00x      1.03x         -
   5000000   sequential        -      1.0065      0.9835      1.0454     39.744      0.97x      1.00x         -
   5000000     parallel        1      0.9798      0.9485      1.0430     40.825      0.99x      1.03x         -
   5000000     parallel        2      0.9361      0.8813      0.9883     42.728      1.04x      1.08x         -
   5000000     parallel        4      0.7416      0.6327      0.8418     53.939      1.31x      1.36x         -
   5000000     parallel        8      0.7300      0.5722      1.4276     54.791      1.33x      1.38x         -
   5000000     parallel       12      0.4442      0.3979      0.5164     90.044      2.19x      2.27x         -
   5000000     parallel       16      0.5357      0.4481      0.6303     74.665      1.82x      1.88x         -
   5000000     parallel       24      0.5468      0.4728      0.8777     73.146      1.78x      1.84x         -

  10000000       scalar        -      2.5171      2.4155      2.6275     31.783      1.00x      1.02x         -
  10000000   sequential        -      2.5569      2.4807      2.8032     31.288      0.98x      1.00x         -
  10000000     parallel        1      2.4873      2.4016      2.7342     32.163      1.01x      1.03x         -
  10000000     parallel        2      2.1785      2.1385      2.6718     36.723      1.16x      1.17x         -
  10000000     parallel        4      2.5206      1.9958      2.9596     31.738      1.00x      1.01x         -
  10000000     parallel        8      1.3063      1.2558      1.8815     61.242      1.93x      1.96x         -
  10000000     parallel       12      1.8678      1.7319      1.9566     42.831      1.35x      1.37x         -
  10000000     parallel       16      2.1700      1.9647      2.2609     36.867      1.16x      1.18x         -
  10000000     parallel       24      2.1510      2.0689      2.2467     37.192      1.17x      1.19x         -

  33554432       scalar        -     11.1952     10.7544     11.9280     23.978      1.00x      0.95x      1.08
  33554432   sequential        -     10.6686     10.6212     10.9124     25.161      1.05x      1.00x         -
  33554432     parallel        1     10.8653     10.8064     12.2920     24.706      1.03x      0.98x         -
  33554432     parallel        2     10.3167     10.1801     11.3277     26.020      1.09x      1.03x         -
  33554432     parallel        4     10.4536      9.9871     11.5032     25.679      1.07x      1.02x         -
  33554432     parallel        8      9.2729      9.1636      9.4429     28.948      1.21x      1.15x         -
  33554432     parallel       12     10.1898      9.8174     10.5371     26.344      1.10x      1.05x         -
  33554432     parallel       16      9.9448      9.8667     10.0545     26.993      1.13x      1.07x         -
  33554432     parallel       24     10.4853     10.0905     11.0952     25.601      1.07x      1.02x         -


== square ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    109.122      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    109.158      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    109.154      1.00x      1.00x         -
      1000     parallel        2      0.0001      0.0001      0.0001    109.149      1.00x      1.00x         -
      1000     parallel        4      0.0001      0.0001      0.0001    109.140      1.00x      1.00x         -
      1000     parallel        8      0.0001      0.0001      0.0001    109.154      1.00x      1.00x         -
      1000     parallel       12      0.0001      0.0001      0.0001    109.149      1.00x      1.00x         -
      1000     parallel       16      0.0001      0.0001      0.0001    109.072      1.00x      1.00x         -
      1000     parallel       24      0.0001      0.0001      0.0001    109.145      1.00x      1.00x         -

     10000       scalar        -      0.0007      0.0007      0.0007    109.526      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    109.343      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0007    108.582      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0007    108.608      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0007    108.600      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0007    108.590      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0007    108.586      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0007    108.582      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0007    108.593      0.99x      0.99x         -

     32768       scalar        -      0.0024      0.0024      0.0024    108.489      1.00x      0.98x         -
     32768   sequential        -      0.0024      0.0024      0.0024    110.277      1.02x      1.00x         -
     32768     parallel        1      0.0025      0.0024      0.0026    106.968      0.99x      0.97x         -
     32768     parallel        2      0.0032      0.0024      0.0035     81.364      0.75x      0.74x         -
     32768     parallel        4      0.0024      0.0024      0.0024    108.127      1.00x      0.98x         -
     32768     parallel        8      0.0024      0.0024      0.0025    108.075      1.00x      0.98x         -
     32768     parallel       12      0.0024      0.0024      0.0024    108.131      1.00x      0.98x         -
     32768     parallel       16      0.0024      0.0024      0.0025    108.166      1.00x      0.98x         -
     32768     parallel       24      0.0024      0.0024      0.0024    108.175      1.00x      0.98x         -

     32769       scalar        -      0.0024      0.0024      0.0024    108.942      1.00x      0.99x         -
     32769   sequential        -      0.0024      0.0024      0.0024    110.153      1.01x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0024    108.523      1.00x      0.99x         -
     32769     parallel        2      0.0155      0.0154      0.0158     16.867      0.15x      0.15x         -
     32769     parallel        4      0.0148      0.0148      0.0150     17.695      0.16x      0.16x         -
     32769     parallel        8      0.0218      0.0216      0.0222     12.023      0.11x      0.11x         -
     32769     parallel       12      0.0278      0.0277      0.0280      9.436      0.09x      0.09x         -
     32769     parallel       16      0.0305      0.0303      0.0310      8.605      0.08x      0.08x         -
     32769     parallel       24      0.0366      0.0361      0.0375      7.157      0.07x      0.06x         -

     40000       scalar        -      0.0031      0.0030      0.0032    104.065      1.00x      0.97x         -
     40000   sequential        -      0.0030      0.0030      0.0031    106.910      1.03x      1.00x         -
     40000     parallel        1      0.0030      0.0030      0.0030    106.493      1.02x      1.00x         -
     40000     parallel        2      0.0164      0.0164      0.0166     19.475      0.19x      0.18x         -
     40000     parallel        4      0.0163      0.0163      0.0166     19.640      0.19x      0.18x         -
     40000     parallel        8      0.0238      0.0226      0.0268     13.456      0.13x      0.13x         -
     40000     parallel       12      0.0283      0.0281      0.0286     11.326      0.11x      0.11x         -
     40000     parallel       16      0.0306      0.0302      0.0312     10.442      0.10x      0.10x         -
     40000     parallel       24      0.0369      0.0359      0.0373      8.683      0.08x      0.08x         -

     50000       scalar        -      0.0039      0.0039      0.0039    103.385      1.00x      1.00x         -
     50000   sequential        -      0.0039      0.0039      0.0039    102.938      1.00x      1.00x         -
     50000     parallel        1      0.0038      0.0038      0.0038    105.101      1.02x      1.02x         -
     50000     parallel        2      0.0176      0.0176      0.0179     22.731      0.22x      0.22x         -
     50000     parallel        4      0.0158      0.0157      0.0160     25.343      0.25x      0.25x         -
     50000     parallel        8      0.0234      0.0232      0.0237     17.119      0.17x      0.17x         -
     50000     parallel       12      0.0295      0.0289      0.0302     13.582      0.13x      0.13x         -
     50000     parallel       16      0.0306      0.0302      0.0310     13.065      0.13x      0.13x         -
     50000     parallel       24      0.0397      0.0392      0.0401     10.080      0.10x      0.10x         -

     75000       scalar        -      0.0061      0.0060      0.0061     99.142      1.00x      1.01x         -
     75000   sequential        -      0.0061      0.0061      0.0062     98.421      0.99x      1.00x         -
     75000     parallel        1      0.0061      0.0061      0.0064     98.807      1.00x      1.00x         -
     75000     parallel        2      0.0214      0.0214      0.0234     28.051      0.28x      0.29x         -
     75000     parallel        4      0.0176      0.0175      0.0178     34.126      0.34x      0.35x         -
     75000     parallel        8      0.0282      0.0268      0.0290     21.300      0.21x      0.22x         -
     75000     parallel       12      0.0320      0.0318      0.0326     18.725      0.19x      0.19x         -
     75000     parallel       16      0.0331      0.0329      0.0339     18.102      0.18x      0.18x         -
     75000     parallel       24      0.0381      0.0376      0.0387     15.738      0.16x      0.16x         -

    100000       scalar        -      0.0083      0.0083      0.0084     96.431      1.00x      1.00x         -
    100000   sequential        -      0.0083      0.0083      0.0084     95.966      1.00x      1.00x         -
    100000     parallel        1      0.0083      0.0083      0.0085     96.390      1.00x      1.00x         -
    100000     parallel        2      0.0257      0.0256      0.0264     31.179      0.32x      0.32x         -
    100000     parallel        4      0.0196      0.0195      0.0203     40.885      0.42x      0.43x         -
    100000     parallel        8      0.0264      0.0263      0.0301     30.280      0.31x      0.32x         -
    100000     parallel       12      0.0337      0.0331      0.0344     23.732      0.25x      0.25x         -
    100000     parallel       16      0.0343      0.0335      0.0349     23.305      0.24x      0.24x         -
    100000     parallel       24      0.0402      0.0396      0.0408     19.910      0.21x      0.21x         -

    200000       scalar        -      0.0168      0.0168      0.0170     95.052      1.00x      1.00x         -
    200000   sequential        -      0.0168      0.0168      0.0169     95.167      1.00x      1.00x         -
    200000     parallel        1      0.0169      0.0168      0.0170     94.872      1.00x      1.00x         -
    200000     parallel        2      0.0408      0.0407      0.0414     39.189      0.41x      0.41x         -
    200000     parallel        4      0.0277      0.0276      0.0279     57.773      0.61x      0.61x         -
    200000     parallel        8      0.0341      0.0338      0.0387     46.871      0.49x      0.49x         -
    200000     parallel       12      0.0394      0.0388      0.0401     40.616      0.43x      0.43x         -
    200000     parallel       16      0.0423      0.0415      0.0438     37.818      0.40x      0.40x         -
    200000     parallel       24      0.0455      0.0450      0.0470     35.182      0.37x      0.37x         -

    500000       scalar        -      0.0417      0.0417      0.0429     95.816      1.00x      1.01x         -
    500000   sequential        -      0.0423      0.0423      0.0427     94.514      0.99x      1.00x         -
    500000     parallel        1      0.0423      0.0417      0.0430     94.500      0.99x      1.00x         -
    500000     parallel        2      0.0839      0.0837      0.0866     47.669      0.50x      0.50x         -
    500000     parallel        4      0.0504      0.0497      0.0535     79.400      0.83x      0.84x         -
    500000     parallel        8      0.0550      0.0547      0.0555     72.754      0.76x      0.77x         -
    500000     parallel       12      0.0596      0.0581      0.0619     67.093      0.70x      0.71x         -
    500000     parallel       16      0.0694      0.0672      0.0706     57.624      0.60x      0.61x         -
    500000     parallel       24      0.0714      0.0706      0.0727     56.030      0.58x      0.59x         -

   1000000       scalar        -      0.0840      0.0837      0.0859     95.263      1.00x      1.01x         -
   1000000   sequential        -      0.0844      0.0841      0.0918     94.783      0.99x      1.00x         -
   1000000     parallel        1      0.0839      0.0836      0.0942     95.341      1.00x      1.01x         -
   1000000     parallel        2      0.1558      0.1549      0.1571     51.350      0.54x      0.54x         -
   1000000     parallel        4      0.0867      0.0855      0.0985     92.232      0.97x      0.97x         -
   1000000     parallel        8      0.0905      0.0895      0.0936     88.428      0.93x      0.93x         -
   1000000     parallel       12      0.0967      0.0890      0.1008     82.738      0.87x      0.87x         -
   1000000     parallel       16      0.1100      0.1022      0.1120     72.756      0.76x      0.77x         -
   1000000     parallel       24      0.1091      0.1066      0.1150     73.325      0.77x      0.77x         -

   2000000       scalar        -      0.1684      0.1681      0.1949     95.015      1.00x      1.00x         -
   2000000   sequential        -      0.1686      0.1671      0.1949     94.892      1.00x      1.00x         -
   2000000     parallel        1      0.1703      0.1678      0.1913     93.973      0.99x      0.99x         -
   2000000     parallel        2      0.3016      0.2990      0.3242     53.057      0.56x      0.56x         -
   2000000     parallel        4      0.1608      0.1590      0.1875     99.533      1.05x      1.05x         -
   2000000     parallel        8      0.1902      0.1674      0.2218     84.114      0.89x      0.89x         -
   2000000     parallel       12      0.1866      0.1747      0.2181     85.754      0.90x      0.90x         -
   2000000     parallel       16      0.1938      0.1440      0.2035     82.549      0.87x      0.87x         -
   2000000     parallel       24      0.2411      0.2335      0.3257     66.373      0.70x      0.70x         -

   5000000       scalar        -      0.9477      0.9405      0.9802     42.207      1.00x      1.03x         -
   5000000   sequential        -      0.9768      0.9633      1.0003     40.948      0.97x      1.00x         -
   5000000     parallel        1      0.9758      0.9549      1.0389     40.990      0.97x      1.00x         -
   5000000     parallel        2      0.9266      0.8893      1.0192     43.166      1.02x      1.05x         -
   5000000     parallel        4      0.5611      0.5328      0.6056     71.295      1.69x      1.74x         -
   5000000     parallel        8      0.3956      0.3325      0.5903    101.119      2.40x      2.47x         -
   5000000     parallel       12      0.6297      0.4854      0.9828     63.522      1.51x      1.55x         -
   5000000     parallel       16      0.5388      0.4336      0.9851     74.239      1.76x      1.81x         -
   5000000     parallel       24      0.6196      0.5035      0.8870     64.558      1.53x      1.58x         -

  10000000       scalar        -      2.4643      2.3844      2.6254     32.464      1.00x      1.05x         -
  10000000   sequential        -      2.5962      2.5160      2.7588     30.814      0.95x      1.00x         -
  10000000     parallel        1      2.4839      2.4448      2.6713     32.207      0.99x      1.05x         -
  10000000     parallel        2      2.1796      2.1387      3.1241     36.704      1.13x      1.19x         -
  10000000     parallel        4      2.1206      2.0062      2.3834     37.725      1.16x      1.22x         -
  10000000     parallel        8      1.4392      1.4070      2.5099     55.586      1.71x      1.80x         -
  10000000     parallel       12      1.8368      1.8065      2.5245     43.554      1.34x      1.41x         -
  10000000     parallel       16      1.9858      1.8725      2.1129     40.286      1.24x      1.31x         -
  10000000     parallel       24      2.2040      2.0991      2.3762     36.298      1.12x      1.18x         -

  33554432       scalar        -     10.8382     10.7224     11.1967     24.768      1.00x      0.98x         -
  33554432   sequential        -     10.5888     10.5635     10.9749     25.351      1.02x      1.00x         -
  33554432     parallel        1     10.7829     10.6908     11.0746     24.895      1.01x      0.98x         -
  33554432     parallel        2     10.3536     10.2369     11.6949     25.927      1.05x      1.02x         -
  33554432     parallel        4     10.0025      9.9095     10.5777     26.837      1.08x      1.06x         -
  33554432     parallel        8      9.5176      9.2386     10.3133     28.204      1.14x      1.11x         -
  33554432     parallel       12      9.6077      9.5040     10.3561     27.940      1.13x      1.10x         -
  33554432     parallel       16     10.0545      9.8343     10.6852     26.698      1.08x      1.05x         -
  33554432     parallel       24     10.0958      9.9449     10.5539     26.589      1.07x      1.05x         -


== relu ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    109.063      1.00x      1.01x         -
      1000   sequential        -      0.0001      0.0001      0.0001    107.976      0.99x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    108.877      1.00x      1.01x         -
      1000     parallel        2      0.0001      0.0001      0.0001    109.131      1.00x      1.01x         -
      1000     parallel        4      0.0001      0.0001      0.0001    109.090      1.00x      1.01x         -
      1000     parallel        8      0.0001      0.0001      0.0001    109.136      1.00x      1.01x         -
      1000     parallel       12      0.0001      0.0001      0.0001    109.136      1.00x      1.01x         -
      1000     parallel       16      0.0001      0.0001      0.0001    109.104      1.00x      1.01x         -
      1000     parallel       24      0.0001      0.0001      0.0001    109.140      1.00x      1.01x         -

     10000       scalar        -      0.0007      0.0007      0.0008    107.708      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    108.031      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0008    107.033      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0008    107.036      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0008    107.022      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0008    107.022      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0008    107.005      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0008    107.043      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0007    107.022      0.99x      0.99x         -

     32768       scalar        -      0.0025      0.0025      0.0025    106.831      1.00x      0.98x         -
     32768   sequential        -      0.0024      0.0024      0.0024    109.213      1.02x      1.00x         -
     32768     parallel        1      0.0025      0.0025      0.0025    106.666      1.00x      0.98x         -
     32768     parallel        2      0.0025      0.0025      0.0025    106.670      1.00x      0.98x         -
     32768     parallel        4      0.0025      0.0025      0.0025    106.526      1.00x      0.98x         -
     32768     parallel        8      0.0025      0.0025      0.0025    106.662      1.00x      0.98x         -
     32768     parallel       12      0.0025      0.0025      0.0025    106.657      1.00x      0.98x         -
     32768     parallel       16      0.0025      0.0025      0.0025    106.628      1.00x      0.98x         -
     32768     parallel       24      0.0025      0.0025      0.0025    106.615      1.00x      0.98x         -

     32769       scalar        -      0.0024      0.0024      0.0025    107.438      1.00x      0.98x         -
     32769   sequential        -      0.0024      0.0024      0.0026    109.181      1.02x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0025    107.240      1.00x      0.98x         -
     32769     parallel        2      0.0154      0.0153      0.0157     16.993      0.16x      0.16x         -
     32769     parallel        4      0.0150      0.0148      0.0151     17.423      0.16x      0.16x         -
     32769     parallel        8      0.0225      0.0224      0.0231     11.669      0.11x      0.11x         -
     32769     parallel       12      0.0280      0.0277      0.0282      9.379      0.09x      0.09x         -
     32769     parallel       16      0.0306      0.0303      0.0308      8.569      0.08x      0.08x         -
     32769     parallel       24      0.0368      0.0360      0.0394      7.119      0.07x      0.07x         -

     40000       scalar        -      0.0029      0.0029      0.0030    108.579      1.00x      1.01x         -
     40000   sequential        -      0.0030      0.0030      0.0030    108.024      0.99x      1.00x         -
     40000     parallel        1      0.0030      0.0029      0.0030    108.331      1.00x      1.00x         -
     40000     parallel        2      0.0165      0.0164      0.0167     19.431      0.18x      0.18x         -
     40000     parallel        4      0.0154      0.0154      0.0159     20.756      0.19x      0.19x         -
     40000     parallel        8      0.0226      0.0222      0.0247     14.141      0.13x      0.13x         -
     40000     parallel       12      0.0284      0.0279      0.0293     11.285      0.10x      0.10x         -
     40000     parallel       16      0.0308      0.0305      0.0310     10.395      0.10x      0.10x         -
     40000     parallel       24      0.0363      0.0356      0.0374      8.817      0.08x      0.08x         -

     50000       scalar        -      0.0038      0.0038      0.0038    104.591      1.00x      1.00x         -
     50000   sequential        -      0.0038      0.0038      0.0039    104.511      1.00x      1.00x         -
     50000     parallel        1      0.0040      0.0038      0.0044    101.019      0.97x      0.97x         -
     50000     parallel        2      0.0190      0.0187      0.0193     21.015      0.20x      0.20x         -
     50000     parallel        4      0.0171      0.0167      0.0174     23.420      0.22x      0.22x         -
     50000     parallel        8      0.0233      0.0230      0.0260     17.177      0.16x      0.16x         -
     50000     parallel       12      0.0293      0.0293      0.0306     13.634      0.13x      0.13x         -
     50000     parallel       16      0.0313      0.0308      0.0318     12.770      0.12x      0.12x         -
     50000     parallel       24      0.0372      0.0369      0.0378     10.753      0.10x      0.10x         -

     75000       scalar        -      0.0062      0.0062      0.0062     97.098      1.00x      0.99x         -
     75000   sequential        -      0.0061      0.0061      0.0062     97.626      1.01x      1.00x         -
     75000     parallel        1      0.0062      0.0062      0.0063     96.686      1.00x      0.99x         -
     75000     parallel        2      0.0218      0.0218      0.0226     27.522      0.28x      0.28x         -
     75000     parallel        4      0.0179      0.0179      0.0181     33.522      0.35x      0.34x         -
     75000     parallel        8      0.0251      0.0250      0.0259     23.909      0.25x      0.24x         -
     75000     parallel       12      0.0315      0.0313      0.0318     19.064      0.20x      0.20x         -
     75000     parallel       16      0.0327      0.0325      0.0339     18.346      0.19x      0.19x         -
     75000     parallel       24      0.0384      0.0378      0.0394     15.616      0.16x      0.16x         -

    100000       scalar        -      0.0084      0.0084      0.0085     95.621      1.00x      1.00x         -
    100000   sequential        -      0.0084      0.0084      0.0085     95.451      1.00x      1.00x         -
    100000     parallel        1      0.0084      0.0084      0.0085     95.429      1.00x      1.00x         -
    100000     parallel        2      0.0262      0.0261      0.0270     30.504      0.32x      0.32x         -
    100000     parallel        4      0.0200      0.0198      0.0218     39.924      0.42x      0.42x         -
    100000     parallel        8      0.0269      0.0265      0.0282     29.698      0.31x      0.31x         -
    100000     parallel       12      0.0336      0.0330      0.0345     23.818      0.25x      0.25x         -
    100000     parallel       16      0.0348      0.0340      0.0351     22.996      0.24x      0.24x         -
    100000     parallel       24      0.0427      0.0409      0.0451     18.720      0.20x      0.20x         -

    200000       scalar        -      0.0172      0.0169      0.0199     93.112      1.00x      1.06x         -
    200000   sequential        -      0.0182      0.0170      0.0192     87.780      0.94x      1.00x         -
    200000     parallel        1      0.0177      0.0170      0.0193     90.228      0.97x      1.03x         -
    200000     parallel        2      0.0429      0.0408      0.0458     37.297      0.40x      0.42x         -
    200000     parallel        4      0.0300      0.0293      0.0339     53.392      0.57x      0.61x         -
    200000     parallel        8      0.0388      0.0351      0.0414     41.247      0.44x      0.47x         -
    200000     parallel       12      0.0394      0.0372      0.0438     40.645      0.44x      0.46x         -
    200000     parallel       16      0.0430      0.0421      0.0443     37.194      0.40x      0.42x         -
    200000     parallel       24      0.0511      0.0496      0.0544     31.303      0.34x      0.36x         -

    500000       scalar        -      0.0422      0.0417      0.0564     94.787      1.00x      1.12x         -
    500000   sequential        -      0.0473      0.0445      0.0504     84.636      0.89x      1.00x         -
    500000     parallel        1      0.0436      0.0419      0.0470     91.806      0.97x      1.08x         -
    500000     parallel        2      0.0861      0.0840      0.0891     46.446      0.49x      0.55x         -
    500000     parallel        4      0.0512      0.0504      0.0522     78.104      0.82x      0.92x         -
    500000     parallel        8      0.0568      0.0555      0.0581     70.411      0.74x      0.83x         -
    500000     parallel       12      0.0604      0.0581      0.0622     66.251      0.70x      0.78x         -
    500000     parallel       16      0.0687      0.0669      0.0702     58.208      0.61x      0.69x         -
    500000     parallel       24      0.0722      0.0712      0.0748     55.397      0.58x      0.65x         -

   1000000       scalar        -      0.0847      0.0841      0.0912     94.507      1.00x      1.01x         -
   1000000   sequential        -      0.0858      0.0857      0.0918     93.240      0.99x      1.00x         -
   1000000     parallel        1      0.0845      0.0843      0.0911     94.727      1.00x      1.02x         -
   1000000     parallel        2      0.1577      0.1572      0.1616     50.745      0.54x      0.54x         -
   1000000     parallel        4      0.0877      0.0860      0.0986     91.256      0.97x      0.98x         -
   1000000     parallel        8      0.0942      0.0900      0.1044     84.945      0.90x      0.91x         -
   1000000     parallel       12      0.0967      0.0908      0.0986     82.727      0.88x      0.89x         -
   1000000     parallel       16      0.1021      0.0953      0.1048     78.374      0.83x      0.84x         -
   1000000     parallel       24      0.1156      0.1069      0.1379     69.230      0.73x      0.74x         -

   2000000       scalar        -      0.1746      0.1721      0.1989     91.661      1.00x      1.24x         -
   2000000   sequential        -      0.2166      0.1725      0.5058     73.865      0.81x      1.00x         -
   2000000     parallel        1      0.1769      0.1701      0.2459     90.437      0.99x      1.22x         -
   2000000     parallel        2      0.3009      0.3001      0.3178     53.183      0.58x      0.72x         -
   2000000     parallel        4      0.1613      0.1577      0.1854     99.186      1.08x      1.34x         -
   2000000     parallel        8      0.1639      0.1594      0.1697     97.632      1.07x      1.32x         -
   2000000     parallel       12      0.2021      0.1843      0.2144     79.174      0.86x      1.07x         -
   2000000     parallel       16      0.1936      0.1857      0.3989     82.626      0.90x      1.12x         -
   2000000     parallel       24      0.2447      0.2293      0.3629     65.386      0.71x      0.89x         -

   5000000       scalar        -      1.0766      1.0633      1.0929     37.154      1.00x      1.06x         -
   5000000   sequential        -      1.1389      1.0948      1.2264     35.122      0.95x      1.00x         -
   5000000     parallel        1      1.2755      1.1781      1.4994     31.360      0.84x      0.89x         -
   5000000     parallel        2      1.1081      1.0341      1.3172     36.096      0.97x      1.03x         -
   5000000     parallel        4      0.9395      0.8291      1.0922     42.574      1.15x      1.21x         -
   5000000     parallel        8      0.4292      0.3066      0.7508     93.202      2.51x      2.65x         -
   5000000     parallel       12      0.5406      0.4388      0.6089     73.995      1.99x      2.11x         -
   5000000     parallel       16      0.8903      0.5205      1.2082     44.927      1.21x      1.28x         -
   5000000     parallel       24      0.6251      0.4652      0.7654     63.990      1.72x      1.82x         -

  10000000       scalar        -      2.9297      2.6498      3.5240     27.307      1.00x      1.08x         -
  10000000   sequential        -      3.1621      2.7361      3.5844     25.300      0.93x      1.00x         -
  10000000     parallel        1      3.1864      2.8726      3.8123     25.107      0.92x      0.99x         -
  10000000     parallel        2      2.3444      2.1938      2.6226     34.124      1.25x      1.35x         -
  10000000     parallel        4      2.0952      2.0515      2.2291     38.183      1.40x      1.51x         -
  10000000     parallel        8      1.8868      1.5125      2.5413     42.400      1.55x      1.68x         -
  10000000     parallel       12      1.7818      1.5943      1.9436     44.898      1.64x      1.77x         -
  10000000     parallel       16      2.1541      2.0693      2.2467     37.138      1.36x      1.47x         -
  10000000     parallel       24      2.2543      2.0230      2.3501     35.488      1.30x      1.40x         -

  33554432       scalar        -     11.3241     11.2368     11.7371     23.705      1.00x      1.02x      1.07
  33554432   sequential        -     11.4991     10.7071     12.1885     23.344      0.98x      1.00x      0.76
  33554432     parallel        1     11.5426     11.1675     18.8994     23.256      0.98x      1.00x      0.82
  33554432     parallel        2     11.9402     10.7282     15.4362     22.482      0.95x      0.96x      1.59
  33554432     parallel        4     10.7077     10.1570     15.4914     25.069      1.06x      1.07x      4.20
  33554432     parallel        8     10.9583     10.4470     13.2300     24.496      1.03x      1.05x      7.55
  33554432     parallel       12     11.9132      9.9750     13.2679     22.533      0.95x      0.97x      9.85
  33554432     parallel       16     10.6856     10.2227     11.3189     25.121      1.06x      1.08x         -
  33554432     parallel       24     10.2368     10.0512     11.2189     26.223      1.11x      1.12x         -


== sqrt ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0004      0.0004      0.0004     22.340      1.00x      1.00x         -
      1000   sequential        -      0.0004      0.0004      0.0004     22.345      1.00x      1.00x         -
      1000     parallel        1      0.0004      0.0004      0.0004     22.269      1.00x      1.00x         -
      1000     parallel        2      0.0004      0.0004      0.0004     22.331      1.00x      1.00x         -
      1000     parallel        4      0.0004      0.0004      0.0004     22.325      1.00x      1.00x         -
      1000     parallel        8      0.0004      0.0004      0.0004     22.334      1.00x      1.00x         -
      1000     parallel       12      0.0004      0.0004      0.0004     22.306      1.00x      1.00x         -
      1000     parallel       16      0.0004      0.0004      0.0004     22.191      0.99x      0.99x         -
      1000     parallel       24      0.0004      0.0004      0.0004     22.310      1.00x      1.00x         -

     10000       scalar        -      0.0036      0.0036      0.0036     22.283      1.00x      1.00x         -
     10000   sequential        -      0.0036      0.0036      0.0036     22.273      1.00x      1.00x         -
     10000     parallel        1      0.0036      0.0036      0.0036     22.254      1.00x      1.00x         -
     10000     parallel        2      0.0036      0.0036      0.0036     22.250      1.00x      1.00x         -
     10000     parallel        4      0.0036      0.0036      0.0037     22.302      1.00x      1.00x         -
     10000     parallel        8      0.0036      0.0036      0.0036     22.329      1.00x      1.00x         -
     10000     parallel       12      0.0036      0.0036      0.0036     22.317      1.00x      1.00x         -
     10000     parallel       16      0.0036      0.0036      0.0036     22.326      1.00x      1.00x         -
     10000     parallel       24      0.0036      0.0036      0.0036     22.345      1.00x      1.00x         -

     32768       scalar        -      0.0117      0.0117      0.0119     22.324      1.00x      1.00x         -
     32768   sequential        -      0.0117      0.0117      0.0118     22.345      1.00x      1.00x         -
     32768     parallel        1      0.0117      0.0117      0.0118     22.344      1.00x      1.00x         -
     32768     parallel        2      0.0117      0.0117      0.0119     22.350      1.00x      1.00x         -
     32768     parallel        4      0.0117      0.0117      0.0119     22.338      1.00x      1.00x         -
     32768     parallel        8      0.0117      0.0117      0.0118     22.344      1.00x      1.00x         -
     32768     parallel       12      0.0118      0.0117      0.0118     22.300      1.00x      1.00x         -
     32768     parallel       16      0.0117      0.0117      0.0119     22.317      1.00x      1.00x         -
     32768     parallel       24      0.0117      0.0117      0.0119     22.318      1.00x      1.00x         -

     32769       scalar        -      0.0117      0.0117      0.0118     22.341      1.00x      1.00x         -
     32769   sequential        -      0.0117      0.0117      0.0119     22.341      1.00x      1.00x         -
     32769     parallel        1      0.0118      0.0117      0.0169     22.297      1.00x      1.00x         -
     32769     parallel        2      0.0563      0.0560      0.0567      4.653      0.21x      0.21x         -
     32769     parallel        4      0.0371      0.0369      0.0378      7.060      0.32x      0.32x         -
     32769     parallel        8      0.0290      0.0287      0.0297      9.051      0.41x      0.41x         -
     32769     parallel       12      0.0274      0.0271      0.0303      9.584      0.43x      0.43x         -
     32769     parallel       16      0.0368      0.0365      0.0389      7.125      0.32x      0.32x         -
     32769     parallel       24      0.0473      0.0455      0.0502      5.538      0.25x      0.25x         -

     40000       scalar        -      0.0143      0.0143      0.0144     22.321      1.00x      1.00x         -
     40000   sequential        -      0.0143      0.0143      0.0144     22.329      1.00x      1.00x         -
     40000     parallel        1      0.0143      0.0143      0.0145     22.340      1.00x      1.00x         -
     40000     parallel        2      0.0667      0.0482      0.0682      4.794      0.21x      0.21x         -
     40000     parallel        4      0.0425      0.0423      0.0429      7.532      0.34x      0.34x         -
     40000     parallel        8      0.0317      0.0314      0.0319     10.082      0.45x      0.45x         -
     40000     parallel       12      0.0295      0.0289      0.0311     10.856      0.49x      0.49x         -
     40000     parallel       16      0.0378      0.0368      0.0395      8.473      0.38x      0.38x         -
     40000     parallel       24      0.0461      0.0458      0.0478      6.940      0.31x      0.31x         -

     50000       scalar        -      0.0179      0.0179      0.0183     22.319      1.00x      1.00x         -
     50000   sequential        -      0.0179      0.0179      0.0181     22.351      1.00x      1.00x         -
     50000     parallel        1      0.0179      0.0179      0.0181     22.348      1.00x      1.00x         -
     50000     parallel        2      0.0813      0.0810      0.0822      4.917      0.22x      0.22x         -
     50000     parallel        4      0.0496      0.0494      0.0512      8.063      0.36x      0.36x         -
     50000     parallel        8      0.0355      0.0354      0.0360     11.263      0.50x      0.50x         -
     50000     parallel       12      0.0314      0.0311      0.0321     12.719      0.57x      0.57x         -
     50000     parallel       16      0.0397      0.0390      0.0424     10.078      0.45x      0.45x         -
     50000     parallel       24      0.0468      0.0462      0.0483      8.549      0.38x      0.38x         -

     75000       scalar        -      0.0270      0.0269      0.0272     22.254      1.00x      1.00x         -
     75000   sequential        -      0.0270      0.0269      0.0272     22.258      1.00x      1.00x         -
     75000     parallel        1      0.0269      0.0269      0.0272     22.282      1.00x      1.00x         -
     75000     parallel        2      0.1170      0.1140      0.1176      5.130      0.23x      0.23x         -
     75000     parallel        4      0.0679      0.0676      0.0684      8.842      0.40x      0.40x         -
     75000     parallel        8      0.0445      0.0444      0.0450     13.473      0.61x      0.61x         -
     75000     parallel       12      0.0380      0.0374      0.0417     15.804      0.71x      0.71x         -
     75000     parallel       16      0.0438      0.0433      0.0457     13.683      0.61x      0.61x         -
     75000     parallel       24      0.0495      0.0487      0.0513     12.117      0.54x      0.54x         -

    100000       scalar        -      0.0359      0.0359      0.0577     22.255      1.00x      1.00x         -
    100000   sequential        -      0.0360      0.0359      0.0423     22.241      1.00x      1.00x         -
    100000     parallel        1      0.0359      0.0359      0.0368     22.259      1.00x      1.00x         -
    100000     parallel        2      0.1530      0.1528      0.1533      5.229      0.23x      0.24x         -
    100000     parallel        4      0.0861      0.0858      0.0872      9.293      0.42x      0.42x         -
    100000     parallel        8      0.0551      0.0532      0.0595     14.511      0.65x      0.65x         -
    100000     parallel       12      0.0441      0.0435      0.0471     18.151      0.82x      0.82x         -
    100000     parallel       16      0.0482      0.0477      0.0541     16.586      0.75x      0.75x         -
    100000     parallel       24      0.0522      0.0519      0.0528     15.319      0.69x      0.69x         -

    200000       scalar        -      0.0717      0.0717      0.0727     22.303      1.00x      1.00x         -
    200000   sequential        -      0.0718      0.0717      0.0719     22.287      1.00x      1.00x         -
    200000     parallel        1      0.0718      0.0718      0.0727     22.281      1.00x      1.00x         -
    200000     parallel        2      0.2975      0.2974      0.3016      5.377      0.24x      0.24x         -
    200000     parallel        4      0.1584      0.1582      0.1597     10.100      0.45x      0.45x         -
    200000     parallel        8      0.0902      0.0897      0.0931     17.736      0.80x      0.80x         -
    200000     parallel       12      0.0735      0.0682      0.0783     21.774      0.98x      0.98x         -
    200000     parallel       16      0.0670      0.0664      0.0713     23.882      1.07x      1.07x         -
    200000     parallel       24      0.0646      0.0636      0.0730     24.777      1.11x      1.11x         -

    500000       scalar        -      0.1794      0.1792      0.1826     22.292      1.00x      1.00x         -
    500000   sequential        -      0.1794      0.1792      0.1834     22.293      1.00x      1.00x         -
    500000     parallel        1      0.1794      0.1794      0.1811     22.293      1.00x      1.00x         -
    500000     parallel        2      0.7269      0.6397      0.7286      5.502      0.25x      0.25x         -
    500000     parallel        4      0.3745      0.3743      0.3781     10.680      0.48x      0.48x         -
    500000     parallel        8      0.1989      0.1980      0.2108     20.108      0.90x      0.90x         -
    500000     parallel       12      0.1452      0.1409      0.1557     27.555      1.24x      1.24x         -
    500000     parallel       16      0.1235      0.1208      0.1364     32.394      1.45x      1.45x         -
    500000     parallel       24      0.1074      0.1025      0.1145     37.253      1.67x      1.67x         -

   1000000       scalar        -      0.3595      0.3587      0.3640     22.254      1.00x      1.00x         -
   1000000   sequential        -      0.3597      0.3591      0.3637     22.242      1.00x      1.00x         -
   1000000     parallel        1      0.3612      0.3594      0.3662     22.151      1.00x      1.00x         -
   1000000     parallel        2      1.4422      0.7600      1.5142      5.547      0.25x      0.25x         -
   1000000     parallel        4      0.7354      0.7338      0.7679     10.879      0.49x      0.49x         -
   1000000     parallel        8      0.3842      0.3771      0.4198     20.820      0.94x      0.94x         -
   1000000     parallel       12      0.2638      0.2610      0.2803     30.323      1.36x      1.36x         -
   1000000     parallel       16      0.2130      0.2114      0.2197     37.552      1.69x      1.69x         -
   1000000     parallel       24      0.1634      0.1613      0.1695     48.960      2.20x      2.20x         -

   2000000       scalar        -      0.7188      0.7174      0.7270     22.259      1.00x      1.00x         -
   2000000   sequential        -      0.7223      0.7196      0.7357     22.150      1.00x      1.00x         -
   2000000     parallel        1      0.7210      0.7196      0.7240     22.190      1.00x      1.00x         -
   2000000     parallel        2      2.8745      1.4629      2.8788      5.566      0.25x      0.25x         -
   2000000     parallel        4      1.4492      1.4472      1.4513     11.041      0.50x      0.50x         -
   2000000     parallel        8      0.7358      0.7346      0.7430     21.746      0.98x      0.98x         -
   2000000     parallel       12      0.5018      0.5005      0.5082     31.885      1.43x      1.44x         -
   2000000     parallel       16      0.3901      0.3888      0.3983     41.015      1.84x      1.85x         -
   2000000     parallel       24      0.2867      0.2779      0.2918     55.800      2.51x      2.52x         -

   5000000       scalar        -      1.9685      1.9366      1.9885     20.321      1.00x      1.00x         -
   5000000   sequential        -      1.9697      1.9260      2.1684     20.308      1.00x      1.00x         -
   5000000     parallel        1      1.9602      1.9342      2.0373     20.406      1.00x      1.00x         -
   5000000     parallel        2      7.1740      3.6320      7.1827      5.576      0.27x      0.27x         -
   5000000     parallel        4      1.8378      1.8177      1.8642     21.765      1.07x      1.07x         -
   5000000     parallel        8      0.9238      0.9195      0.9413     43.301      2.13x      2.13x         -
   5000000     parallel       12      1.0740      0.9349      1.1143     37.242      1.83x      1.83x         -
   5000000     parallel       16      1.0440      0.9504      1.1025     38.312      1.89x      1.89x         -
   5000000     parallel       24      0.6618      0.6425      0.7192     60.439      2.97x      2.98x         -

  10000000       scalar        -      4.0802      4.0105      5.5411     19.607      1.00x      0.99x         -
  10000000   sequential        -      4.0256      3.9850      4.2067     19.873      1.01x      1.00x         -
  10000000     parallel        1      4.0292      3.9856      4.2385     19.855      1.01x      1.00x         -
  10000000     parallel        2      7.2576      7.2282      7.2710     11.023      0.56x      0.55x         -
  10000000     parallel        4      3.6883      3.6450      4.6513     21.690      1.11x      1.09x         -
  10000000     parallel        8      1.9835      1.8773      2.2296     40.333      2.06x      2.03x         -
  10000000     parallel       12      2.0456      1.8853      3.4509     39.108      1.99x      1.97x         -
  10000000     parallel       16      3.1046      2.8729      3.4093     25.768      1.31x      1.30x         -
  10000000     parallel       24      2.9373      2.5445      3.0342     27.236      1.39x      1.37x         -

  33554432       scalar        -     15.2535     14.7175     17.2724     17.598      1.00x      0.93x      1.01
  33554432   sequential        -     14.1146     13.7346     14.9354     19.018      1.08x      1.00x      1.10
  33554432     parallel        1     14.7401     14.5389     18.1648     18.211      1.03x      0.96x      0.92
  33554432     parallel        2     48.1347     46.7146     48.2259      5.577      0.32x      0.29x      1.95
  33554432     parallel        4     24.1052     19.4780     24.4435     11.136      0.63x      0.59x      4.19
  33554432     parallel        8     13.0738     12.2754     13.6474     20.532      1.17x      1.08x      8.25
  33554432     parallel       12     12.1789     11.6754     13.4652     22.041      1.25x      1.16x     10.42
  33554432     parallel       16     11.4816     10.6038     11.9667     23.380      1.33x      1.23x     13.54
  33554432     parallel       24     11.6700     11.3838     12.8175     23.002      1.31x      1.21x     17.06


== exp ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0027      0.0026      0.0031      2.926      1.00x      0.98x         -
      1000   sequential        -      0.0027      0.0026      0.0028      2.992      1.02x      1.00x         -
      1000     parallel        1      0.0027      0.0026      0.0027      2.991      1.02x      1.00x         -
      1000     parallel        2      0.0028      0.0026      0.0029      2.903      0.99x      0.97x         -
      1000     parallel        4      0.0027      0.0026      0.0027      3.016      1.03x      1.01x         -
      1000     parallel        8      0.0026      0.0026      0.0027      3.020      1.03x      1.01x         -
      1000     parallel       12      0.0026      0.0026      0.0027      3.088      1.06x      1.03x         -
      1000     parallel       16      0.0027      0.0026      0.0033      2.911      0.99x      0.97x         -
      1000     parallel       24      0.0027      0.0026      0.0027      3.007      1.03x      1.00x         -

     10000       scalar        -      0.0259      0.0258      0.0280      3.094      1.00x      1.02x         -
     10000   sequential        -      0.0263      0.0258      0.0269      3.037      0.98x      1.00x         -
     10000     parallel        1      0.0268      0.0258      0.0279      2.984      0.96x      0.98x         -
     10000     parallel        2      0.0259      0.0258      0.0277      3.086      1.00x      1.02x         -
     10000     parallel        4      0.0260      0.0258      0.0268      3.074      0.99x      1.01x         -
     10000     parallel        8      0.0258      0.0258      0.0269      3.097      1.00x      1.02x         -
     10000     parallel       12      0.0259      0.0258      0.0262      3.094      1.00x      1.02x         -
     10000     parallel       16      0.0258      0.0258      0.0261      3.100      1.00x      1.02x         -
     10000     parallel       24      0.0260      0.0258      0.0270      3.080      1.00x      1.01x         -

     32768       scalar        -      0.0859      0.0845      0.0908      3.050      1.00x      1.02x         -
     32768   sequential        -      0.0874      0.0845      0.0922      2.999      0.98x      1.00x         -
     32768     parallel        1      0.0858      0.0845      0.1621      3.054      1.00x      1.02x         -
     32768     parallel        2      0.0855      0.0845      0.0881      3.067      1.01x      1.02x         -
     32768     parallel        4      0.0847      0.0845      0.0892      3.095      1.01x      1.03x         -
     32768     parallel        8      0.0847      0.0845      0.0854      3.096      1.01x      1.03x         -
     32768     parallel       12      0.0850      0.0845      0.0880      3.083      1.01x      1.03x         -
     32768     parallel       16      0.0854      0.0845      0.0893      3.070      1.01x      1.02x         -
     32768     parallel       24      0.0848      0.0845      0.0889      3.093      1.01x      1.03x         -

     32769       scalar        -      0.0854      0.0845      0.0879      3.069      1.00x      1.00x         -
     32769   sequential        -      0.0850      0.0845      0.0885      3.084      1.00x      1.00x         -
     32769     parallel        1      0.0863      0.0845      0.0888      3.036      0.99x      0.98x         -
     32769     parallel        2      0.0975      0.0958      0.0981      2.689      0.88x      0.87x         -
     32769     parallel        4      0.0577      0.0568      0.0605      4.543      1.48x      1.47x         -
     32769     parallel        8      0.0396      0.0386      0.0401      6.623      2.16x      2.15x         -
     32769     parallel       12      0.0341      0.0335      0.0411      7.695      2.51x      2.50x         -
     32769     parallel       16      0.0459      0.0445      0.0496      5.712      1.86x      1.85x         -
     32769     parallel       24      0.0515      0.0503      0.0532      5.089      1.66x      1.65x         -

     40000       scalar        -      0.1032      0.1031      0.1054      3.102      1.00x      1.01x         -
     40000   sequential        -      0.1044      0.1035      0.1084      3.066      0.99x      1.00x         -
     40000     parallel        1      0.1088      0.1037      0.1151      2.941      0.95x      0.96x         -
     40000     parallel        2      0.1169      0.1154      0.1240      2.738      0.88x      0.89x         -
     40000     parallel        4      0.0681      0.0658      0.0742      4.701      1.52x      1.53x         -
     40000     parallel        8      0.0510      0.0451      0.0534      6.277      2.02x      2.05x         -
     40000     parallel       12      0.0480      0.0462      0.0539      6.671      2.15x      2.18x         -
     40000     parallel       16      0.0496      0.0469      0.0561      6.458      2.08x      2.11x         -
     40000     parallel       24      0.0562      0.0532      0.0586      5.699      1.84x      1.86x         -

     50000       scalar        -      0.1333      0.1292      0.1367      3.000      1.00x      1.02x         -
     50000   sequential        -      0.1356      0.1291      0.1425      2.950      0.98x      1.00x         -
     50000     parallel        1      0.1396      0.1306      0.1557      2.866      0.96x      0.97x         -
     50000     parallel        2      0.1421      0.1364      0.1587      2.815      0.94x      0.95x         -
     50000     parallel        4      0.0819      0.0797      0.0866      4.882      1.63x      1.65x         -
     50000     parallel        8      0.0580      0.0531      0.0629      6.901      2.30x      2.34x         -
     50000     parallel       12      0.0500      0.0410      0.0537      8.008      2.67x      2.71x         -
     50000     parallel       16      0.0541      0.0493      0.0587      7.398      2.47x      2.51x         -
     50000     parallel       24      0.0579      0.0538      0.0617      6.905      2.30x      2.34x         -

     75000       scalar        -      0.1990      0.1934      0.2504      3.015      1.00x      1.02x         -
     75000   sequential        -      0.2031      0.1949      0.2257      2.954      0.98x      1.00x         -
     75000     parallel        1      0.2232      0.1936      0.2335      2.688      0.89x      0.91x         -
     75000     parallel        2      0.2027      0.1901      0.2087      2.959      0.98x      1.00x         -
     75000     parallel        4      0.1174      0.1131      0.1200      5.110      1.70x      1.73x         -
     75000     parallel        8      0.0714      0.0664      0.0749      8.398      2.79x      2.84x         -
     75000     parallel       12      0.0585      0.0519      0.0630     10.257      3.40x      3.47x         -
     75000     parallel       16      0.0610      0.0585      0.0647      9.842      3.26x      3.33x         -
     75000     parallel       24      0.0675      0.0617      0.0775      8.894      2.95x      3.01x         -

    100000       scalar        -      0.2588      0.2583      0.2617      3.091      1.00x      1.00x         -
    100000   sequential        -      0.2583      0.2583      0.2650      3.097      1.00x      1.00x         -
    100000     parallel        1      0.2633      0.2582      0.2781      3.038      0.98x      0.98x         -
    100000     parallel        2      0.2727      0.2724      0.2750      2.933      0.95x      0.95x         -
    100000     parallel        4      0.1490      0.1429      0.1602      5.371      1.74x      1.73x         -
    100000     parallel        8      0.0918      0.0839      0.1140      8.713      2.82x      2.81x         -
    100000     parallel       12      0.0667      0.0634      0.0774     11.999      3.88x      3.87x         -
    100000     parallel       16      0.0697      0.0657      0.0741     11.485      3.72x      3.71x         -
    100000     parallel       24      0.0670      0.0632      0.0742     11.943      3.86x      3.86x         -

    200000       scalar        -      0.5167      0.5166      0.5255      3.097      1.00x      1.00x         -
    200000   sequential        -      0.5167      0.5166      0.5190      3.097      1.00x      1.00x         -
    200000     parallel        1      0.5168      0.5167      0.5618      3.096      1.00x      1.00x         -
    200000     parallel        2      0.5300      0.4716      0.5567      3.019      0.97x      0.97x         -
    200000     parallel        4      0.2842      0.2736      0.2966      5.631      1.82x      1.82x         -
    200000     parallel        8      0.1531      0.1477      0.1620     10.451      3.37x      3.37x         -
    200000     parallel       12      0.1135      0.1077      0.1183     14.099      4.55x      4.55x         -
    200000     parallel       16      0.1059      0.0989      0.1099     15.109      4.88x      4.88x         -
    200000     parallel       24      0.1056      0.0994      0.1314     15.158      4.89x      4.89x         -

    500000       scalar        -      1.5038      1.3180      2.0334      2.660      1.00x      0.86x         -
    500000   sequential        -      1.2935      1.2900      2.3218      3.092      1.16x      1.00x         -
    500000     parallel        1      1.2935      1.2912      1.4147      3.092      1.16x      1.00x         -
    500000     parallel        2      1.3045      0.9784      1.3234      3.066      1.15x      0.99x         -
    500000     parallel        4      0.6612      0.6595      0.7093      6.050      2.27x      1.96x         -
    500000     parallel        8      0.3428      0.3415      0.3598     11.668      4.39x      3.77x         -
    500000     parallel       12      0.2531      0.2408      0.2644     15.801      5.94x      5.11x         -
    500000     parallel       16      0.2087      0.1981      0.2147     19.169      7.21x      6.20x         -
    500000     parallel       24      0.1700      0.1519      0.2000     23.534      8.85x      7.61x         -

   1000000       scalar        -      2.7926      2.6252      3.8684      2.865      1.00x      0.93x         -
   1000000   sequential        -      2.5995      2.5824      2.6982      3.078      1.07x      1.00x         -
   1000000     parallel        1      2.5955      2.5887      2.9636      3.082      1.08x      1.00x         -
   1000000     parallel        2      2.5992      2.5968      2.6238      3.078      1.07x      1.00x         -
   1000000     parallel        4      1.3079      1.3029      1.3207      6.117      2.14x      1.99x         -
   1000000     parallel        8      0.6900      0.6647      0.7275     11.594      4.05x      3.77x         -
   1000000     parallel       12      0.4824      0.4527      0.5423     16.584      5.79x      5.39x         -
   1000000     parallel       16      0.3865      0.3732      0.4045     20.698      7.23x      6.73x         -
   1000000     parallel       24      0.2807      0.2651      0.2943     28.504      9.95x      9.26x         -

   2000000       scalar        -      5.1989      5.1734      5.3515      3.078      1.00x      1.00x         -
   2000000   sequential        -      5.2185      5.1700      5.2732      3.066      1.00x      1.00x         -
   2000000     parallel        1      5.3067      5.1717      6.0043      3.015      0.98x      0.98x         -
   2000000     parallel        2      5.1813      5.1786      5.2388      3.088      1.00x      1.01x         -
   2000000     parallel        4      2.6093      2.5995      2.6665      6.132      1.99x      2.00x         -
   2000000     parallel        8      1.3101      1.3074      1.3248     12.213      3.97x      3.98x         -
   2000000     parallel       12      0.8852      0.8830      0.9028     18.076      5.87x      5.90x         -
   2000000     parallel       16      0.7263      0.7163      0.9003     22.029      7.16x      7.18x         -
   2000000     parallel       24      0.5039      0.4972      0.5622     31.751     10.32x     10.36x         -

   5000000       scalar        -     13.2313     13.0315     15.3045      3.023      1.00x      1.00x      1.28
   5000000   sequential        -     13.2206     13.0636     13.5470      3.026      1.00x      1.00x      1.05
   5000000     parallel        1     13.2247     13.0977     13.3453      3.025      1.00x      1.00x      1.05
   5000000     parallel        2     12.9442      7.3781     12.9782      3.090      1.02x      1.02x      1.97
   5000000     parallel        4      6.5105      6.4861      6.6437      6.144      2.03x      2.03x         -
   5000000     parallel        8      3.2611      3.2533      3.6648     12.266      4.06x      4.05x         -
   5000000     parallel       12      2.5616      2.2699      2.8150     15.615      5.17x      5.16x         -
   5000000     parallel       16      2.0401      1.8972      2.0806     19.607      6.49x      6.48x         -
   5000000     parallel       24      1.3033      1.2584      1.9285     30.692     10.15x     10.14x         -

  10000000       scalar        -     26.3606     25.9774     36.8610      3.035      1.00x      1.00x      0.94
  10000000   sequential        -     26.2531     26.0646     28.6831      3.047      1.00x      1.00x      0.98
  10000000     parallel        1     26.0457     25.9927     26.3592      3.072      1.01x      1.01x      1.06
  10000000     parallel        2     25.9146     25.8850     25.9409      3.087      1.02x      1.01x      2.01
  10000000     parallel        4     12.9874     12.9436     13.0675      6.160      2.03x      2.02x      3.74
  10000000     parallel        8      6.5276      6.5200      6.5414     12.256      4.04x      4.02x         -
  10000000     parallel       12      4.3627      4.3459      4.4044     18.337      6.04x      6.02x         -
  10000000     parallel       16      3.6668      3.4855      3.9582     21.817      7.19x      7.16x         -
  10000000     parallel       24      3.0071      2.7863      3.5934     26.604      8.77x      8.73x         -

  33554432       scalar        -     88.8719     87.9204     95.2058      3.020      1.00x      0.99x      1.00
  33554432   sequential        -     88.2834     87.3470     95.5422      3.041      1.01x      1.00x      0.96
  33554432     parallel        1     88.7820     87.6890     98.8779      3.024      1.00x      0.99x      1.01
  33554432     parallel        2     86.9108     86.8407     87.0551      3.089      1.02x      1.02x      2.00
  33554432     parallel        4     44.0142     43.4407     44.5533      6.099      2.02x      2.01x      3.71
  33554432     parallel        8     21.7697     21.7494     21.9882     12.331      4.08x      4.06x      8.20
  33554432     parallel       12     14.6412     14.5238     15.8672     18.334      6.07x      6.03x     12.36
  33554432     parallel       16     11.6505     11.1535     12.3012     23.041      7.63x      7.58x     14.02
  33554432     parallel       24     10.7415     10.6491     11.2246     24.991      8.27x      8.22x         -


== sigmoid ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0032      0.0032      0.0033      2.474      1.00x      0.99x         -
      1000   sequential        -      0.0032      0.0032      0.0033      2.491      1.01x      1.00x         -
      1000     parallel        1      0.0033      0.0032      0.0033      2.452      0.99x      0.98x         -
      1000     parallel        2      0.0032      0.0032      0.0033      2.474      1.00x      0.99x         -
      1000     parallel        4      0.0032      0.0032      0.0034      2.468      1.00x      0.99x         -
      1000     parallel        8      0.0033      0.0032      0.0034      2.446      0.99x      0.98x         -
      1000     parallel       12      0.0032      0.0032      0.0033      2.472      1.00x      0.99x         -
      1000     parallel       16      0.0032      0.0032      0.0033      2.469      1.00x      0.99x         -
      1000     parallel       24      0.0032      0.0032      0.0039      2.473      1.00x      0.99x         -

     10000       scalar        -      0.0320      0.0320      0.0333      2.500      1.00x      1.01x         -
     10000   sequential        -      0.0323      0.0320      0.0357      2.475      0.99x      1.00x         -
     10000     parallel        1      0.0323      0.0322      0.0325      2.478      0.99x      1.00x         -
     10000     parallel        2      0.0323      0.0322      0.0350      2.479      0.99x      1.00x         -
     10000     parallel        4      0.0323      0.0323      0.0333      2.477      0.99x      1.00x         -
     10000     parallel        8      0.0323      0.0322      0.0329      2.479      0.99x      1.00x         -
     10000     parallel       12      0.0323      0.0322      0.0333      2.480      0.99x      1.00x         -
     10000     parallel       16      0.0324      0.0323      0.0347      2.467      0.99x      1.00x         -
     10000     parallel       24      0.0324      0.0323      0.0347      2.469      0.99x      1.00x         -

     32768       scalar        -      0.1049      0.1048      0.1115      2.499      1.00x      1.00x         -
     32768   sequential        -      0.1051      0.1047      0.1091      2.495      1.00x      1.00x         -
     32768     parallel        1      0.1060      0.1056      0.1084      2.474      0.99x      0.99x         -
     32768     parallel        2      0.1058      0.1056      0.1097      2.478      0.99x      0.99x         -
     32768     parallel        4      0.1058      0.1056      0.1085      2.478      0.99x      0.99x         -
     32768     parallel        8      0.1059      0.1056      0.1099      2.476      0.99x      0.99x         -
     32768     parallel       12      0.1059      0.1055      0.1062      2.476      0.99x      0.99x         -
     32768     parallel       16      0.1058      0.1056      0.1068      2.479      0.99x      0.99x         -
     32768     parallel       24      0.1058      0.1057      0.1069      2.477      0.99x      0.99x         -

     32769       scalar        -      0.1049      0.1047      0.1088      2.500      1.00x      1.00x         -
     32769   sequential        -      0.1050      0.1047      0.1085      2.497      1.00x      1.00x         -
     32769     parallel        1      0.1057      0.1056      0.1067      2.479      0.99x      0.99x         -
     32769     parallel        2      0.1149      0.1120      0.1184      2.281      0.91x      0.91x         -
     32769     parallel        4      0.0658      0.0647      0.0711      3.982      1.59x      1.60x         -
     32769     parallel        8      0.0441      0.0428      0.0499      5.948      2.38x      2.38x         -
     32769     parallel       12      0.0461      0.0432      0.0507      5.681      2.27x      2.28x         -
     32769     parallel       16      0.0465      0.0451      0.0492      5.637      2.25x      2.26x         -
     32769     parallel       24      0.0532      0.0512      0.0576      4.926      1.97x      1.97x         -

     40000       scalar        -      0.1279      0.1279      0.1397      2.502      1.00x      1.00x         -
     40000   sequential        -      0.1282      0.1281      0.1336      2.497      1.00x      1.00x         -
     40000     parallel        1      0.1290      0.1288      0.1335      2.480      0.99x      0.99x         -
     40000     parallel        2      0.1353      0.1328      0.1362      2.366      0.95x      0.95x         -
     40000     parallel        4      0.0765      0.0759      0.0787      4.182      1.67x      1.67x         -
     40000     parallel        8      0.0517      0.0491      0.0532      6.188      2.47x      2.48x         -
     40000     parallel       12      0.0500      0.0487      0.0505      6.399      2.56x      2.56x         -
     40000     parallel       16      0.0496      0.0482      0.0509      6.456      2.58x      2.59x         -
     40000     parallel       24      0.0549      0.0536      0.0618      5.833      2.33x      2.34x         -

     50000       scalar        -      0.1672      0.1600      0.1746      2.393      1.00x      1.00x         -
     50000   sequential        -      0.1675      0.1599      0.1728      2.388      1.00x      1.00x         -
     50000     parallel        1      0.1739      0.1614      0.1781      2.300      0.96x      0.96x         -
     50000     parallel        2      0.1655      0.1624      0.1673      2.417      1.01x      1.01x         -
     50000     parallel        4      0.0916      0.0908      0.0991      4.368      1.83x      1.83x         -
     50000     parallel        8      0.0577      0.0563      0.0591      6.933      2.90x      2.90x         -
     50000     parallel       12      0.0479      0.0452      0.0524      8.344      3.49x      3.49x         -
     50000     parallel       16      0.0531      0.0516      0.0555      7.533      3.15x      3.15x         -
     50000     parallel       24      0.0549      0.0544      0.0583      7.287      3.05x      3.05x         -

     75000       scalar        -      0.2430      0.2402      0.2454      2.469      1.00x      0.99x         -
     75000   sequential        -      0.2405      0.2401      0.2465      2.494      1.01x      1.00x         -
     75000     parallel        1      0.2437      0.2420      0.2527      2.462      1.00x      0.99x         -
     75000     parallel        2      0.2418      0.2342      0.2475      2.481      1.00x      0.99x         -
     75000     parallel        4      0.1294      0.1289      0.1322      4.636      1.88x      1.86x         -
     75000     parallel        8      0.0759      0.0750      0.0766      7.903      3.20x      3.17x         -
     75000     parallel       12      0.0595      0.0589      0.0602     10.087      4.08x      4.04x         -
     75000     parallel       16      0.0620      0.0616      0.0662      9.680      3.92x      3.88x         -
     75000     parallel       24      0.0648      0.0618      0.0669      9.265      3.75x      3.71x         -

    100000       scalar        -      0.3204      0.3200      0.3272      2.497      1.00x      1.00x         -
    100000   sequential        -      0.3209      0.3201      0.3266      2.493      1.00x      1.00x         -
    100000     parallel        1      0.3234      0.3224      0.3997      2.474      0.99x      0.99x         -
    100000     parallel        2      0.3192      0.3178      0.3211      2.506      1.00x      1.01x         -
    100000     parallel        4      0.1680      0.1664      0.1687      4.762      1.91x      1.91x         -
    100000     parallel        8      0.0950      0.0945      0.1163      8.424      3.37x      3.38x         -
    100000     parallel       12      0.0783      0.0743      0.0819     10.213      4.09x      4.10x         -
    100000     parallel       16      0.0719      0.0712      0.0730     11.128      4.46x      4.46x         -
    100000     parallel       24      0.0717      0.0707      0.0744     11.151      4.47x      4.47x         -

    200000       scalar        -      0.6406      0.6402      0.6513      2.498      1.00x      1.00x         -
    200000   sequential        -      0.6420      0.6403      0.6604      2.492      1.00x      1.00x         -
    200000     parallel        1      0.6458      0.6450      0.6563      2.478      0.99x      0.99x         -
    200000     parallel        2      0.6234      0.6220      0.6349      2.566      1.03x      1.03x         -
    200000     parallel        4      0.3221      0.3200      0.3267      4.967      1.99x      1.99x         -
    200000     parallel        8      0.1726      0.1717      0.1754      9.269      3.71x      3.72x         -
    200000     parallel       12      0.1243      0.1232      0.1249     12.870      5.15x      5.16x         -
    200000     parallel       16      0.1106      0.1097      0.1130     14.473      5.79x      5.81x         -
    200000     parallel       24      0.0968      0.0927      0.1078     16.523      6.61x      6.63x         -

    500000       scalar        -      1.6018      1.5998      1.7469      2.497      1.00x      1.00x         -
    500000   sequential        -      1.6035      1.6001      2.0235      2.495      1.00x      1.00x         -
    500000     parallel        1      1.6152      1.6133      1.6515      2.476      0.99x      0.99x         -
    500000     parallel        2      1.5364      1.5333      1.5482      2.603      1.04x      1.04x         -
    500000     parallel        4      0.7803      0.7776      0.7860      5.126      2.05x      2.05x         -
    500000     parallel        8      0.4032      0.4001      0.4089      9.922      3.97x      3.98x         -
    500000     parallel       12      0.2781      0.2764      0.2819     14.385      5.76x      5.77x         -
    500000     parallel       16      0.2268      0.2257      0.2295     17.641      7.06x      7.07x         -
    500000     parallel       24      0.1726      0.1699      0.1803     23.170      9.28x      9.29x         -

   1000000       scalar        -      3.2429      3.2003      3.4482      2.467      1.00x      1.02x         -
   1000000   sequential        -      3.2942      3.1998      3.3194      2.429      0.98x      1.00x         -
   1000000     parallel        1      3.3141      3.2295      3.3999      2.414      0.98x      0.99x         -
   1000000     parallel        2      3.0810      3.0687      3.2012      2.597      1.05x      1.07x         -
   1000000     parallel        4      1.5439      1.5369      1.5962      5.182      2.10x      2.13x         -
   1000000     parallel        8      0.7861      0.7838      0.7923     10.176      4.13x      4.19x         -
   1000000     parallel       12      0.5374      0.5361      0.5431     14.885      6.03x      6.13x         -
   1000000     parallel       16      0.4221      0.4187      0.4647     18.952      7.68x      7.80x         -
   1000000     parallel       24      0.3093      0.3011      0.3388     25.867     10.49x     10.65x         -

   2000000       scalar        -      6.4536      6.4086      6.5829      2.479      1.00x      1.00x         -
   2000000   sequential        -      6.4857      6.4252      6.7880      2.467      1.00x      1.00x         -
   2000000     parallel        1      6.5864      6.5070      6.6727      2.429      0.98x      0.98x         -
   2000000     parallel        2      6.1255      6.1138      6.1466      2.612      1.05x      1.06x         -
   2000000     parallel        4      3.0713      3.0642      3.1106      5.210      2.10x      2.11x         -
   2000000     parallel        8      1.5504      1.5458      1.5589     10.320      4.16x      4.18x         -
   2000000     parallel       12      1.0442      1.0420      1.0556     15.323      6.18x      6.21x         -
   2000000     parallel       16      0.8270      0.8161      0.8589     19.348      7.80x      7.84x         -
   2000000     parallel       24      0.8352      0.5633      0.9972     19.158      7.73x      7.77x         -

   5000000       scalar        -     16.7608     16.6791     17.0281      2.387      1.00x      0.98x      0.93
   5000000   sequential        -     16.3828     16.1547     16.9815      2.442      1.02x      1.00x      0.84
   5000000     parallel        1     16.4136     16.2976     17.7777      2.437      1.02x      1.00x      0.94
   5000000     parallel        2     15.7721     15.7002     15.9649      2.536      1.06x      1.04x      1.87
   5000000     parallel        4      7.8759      7.8350      7.9377      5.079      2.13x      2.08x         -
   5000000     parallel        8      3.9511      3.8689      4.0464     10.124      4.24x      4.15x         -
   5000000     parallel       12      2.6244      2.5853      2.9213     15.242      6.39x      6.24x         -
   5000000     parallel       16      2.1094      2.0479      2.2148     18.963      7.95x      7.77x         -
   5000000     parallel       24      1.4222      1.3599      1.5119     28.125     11.79x     11.52x         -

  10000000       scalar        -     33.1477     32.1971     34.8773      2.413      1.00x      0.99x      0.99
  10000000   sequential        -     32.9774     32.5951     34.0690      2.426      1.01x      1.00x      1.05
  10000000     parallel        1     33.2259     32.5081     34.2068      2.408      1.00x      0.99x      0.99
  10000000     parallel        2     16.1412     16.0972     16.5684      4.956      2.05x      2.04x      2.03
  10000000     parallel        4      8.1944      8.1776      8.2362      9.763      4.05x      4.02x         -
  10000000     parallel        8      4.4052      4.2061      4.5731     18.160      7.52x      7.49x         -
  10000000     parallel       12      3.1474      2.9602      5.2289     25.418     10.53x     10.48x         -
  10000000     parallel       16      3.8535      3.8489      3.8672     20.760      8.60x      8.56x         -
  10000000     parallel       24      2.8686      2.7558      4.9131     27.888     11.56x     11.50x         -

  33554432       scalar        -    110.7062    110.2794    110.9076      2.425      1.00x      0.98x      0.99
  33554432   sequential        -    108.1150    107.9643    108.3100      2.483      1.02x      1.00x      0.98
  33554432     parallel        1    111.5895    111.1182    113.5483      2.406      0.99x      0.97x      0.98
  33554432     parallel        2     55.7992     55.4434     57.5614      4.811      1.98x      1.94x      2.07
  33554432     parallel        4     27.9086     27.7654     48.1704      9.618      3.97x      3.87x      4.03
  33554432     parallel        8     14.2664     14.1336     17.7495     18.816      7.76x      7.58x      7.59
  33554432     parallel       12     11.1292     10.6538     16.3371     24.120      9.95x      9.71x      8.17
  33554432     parallel       16     12.9510     12.9207     13.3781     20.727      8.55x      8.35x     13.62
  33554432     parallel       24     11.1292     10.8629     14.2581     24.120      9.95x      9.71x     19.82


== axpy ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0017      0.0017      0.0017      7.251      1.00x      1.00x         -
      1000   sequential        -      0.0017      0.0017      0.0017      7.250      1.00x      1.00x         -
      1000     parallel        1      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -
      1000     parallel        2      0.0017      0.0017      0.0017      7.235      1.00x      1.00x         -
      1000     parallel        4      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -
      1000     parallel        8      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -
      1000     parallel       12      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -
      1000     parallel       16      0.0017      0.0017      0.0017      7.236      1.00x      1.00x         -
      1000     parallel       24      0.0017      0.0017      0.0017      7.236      1.00x      1.00x         -

     10000       scalar        -      0.0165      0.0165      0.0167      7.286      1.00x      1.00x         -
     10000   sequential        -      0.0165      0.0165      0.0167      7.287      1.00x      1.00x         -
     10000     parallel        1      0.0165      0.0165      0.0165      7.275      1.00x      1.00x         -
     10000     parallel        2      0.0165      0.0165      0.0166      7.273      1.00x      1.00x         -
     10000     parallel        4      0.0165      0.0165      0.0165      7.285      1.00x      1.00x         -
     10000     parallel        8      0.0165      0.0165      0.0167      7.285      1.00x      1.00x         -
     10000     parallel       12      0.0165      0.0165      0.0167      7.285      1.00x      1.00x         -
     10000     parallel       16      0.0165      0.0165      0.0165      7.285      1.00x      1.00x         -
     10000     parallel       24      0.0165      0.0165      0.0168      7.285      1.00x      1.00x         -

     32768       scalar        -      0.0540      0.0539      0.0548      7.282      1.00x      1.00x         -
     32768   sequential        -      0.0540      0.0540      0.0548      7.283      1.00x      1.00x         -
     32768     parallel        1      0.0544      0.0539      0.0545      7.226      0.99x      0.99x         -
     32768     parallel        2      0.0540      0.0540      0.0547      7.284      1.00x      1.00x         -
     32768     parallel        4      0.0540      0.0539      0.0544      7.283      1.00x      1.00x         -
     32768     parallel        8      0.0540      0.0539      0.0544      7.284      1.00x      1.00x         -
     32768     parallel       12      0.0540      0.0539      0.0544      7.284      1.00x      1.00x         -
     32768     parallel       16      0.0540      0.0539      0.0546      7.283      1.00x      1.00x         -
     32768     parallel       24      0.0540      0.0540      0.0544      7.282      1.00x      1.00x         -

     32769       scalar        -      0.0540      0.0540      0.0543      7.286      1.00x      1.00x         -
     32769   sequential        -      0.0541      0.0540      0.0544      7.275      1.00x      1.00x         -
     32769     parallel        1      0.0540      0.0540      0.0548      7.283      1.00x      1.00x         -
     32769     parallel        2      0.0359      0.0355      0.0366     10.945      1.50x      1.50x         -
     32769     parallel        4      0.0254      0.0252      0.0270     15.489      2.13x      2.13x         -
     32769     parallel        8      0.0270      0.0269      0.0272     14.552      2.00x      2.00x         -
     32769     parallel       12      0.0303      0.0302      0.0305     12.992      1.78x      1.79x         -
     32769     parallel       16      0.0323      0.0320      0.0329     12.161      1.67x      1.67x         -
     32769     parallel       24      0.0418      0.0410      0.0423      9.397      1.29x      1.29x         -

     40000       scalar        -      0.0659      0.0659      0.0668      7.285      1.00x      1.00x         -
     40000   sequential        -      0.0661      0.0661      0.0669      7.262      1.00x      1.00x         -
     40000     parallel        1      0.0659      0.0659      0.0660      7.285      1.00x      1.00x         -
     40000     parallel        2      0.0425      0.0422      0.0433     11.293      1.55x      1.55x         -
     40000     parallel        4      0.0283      0.0282      0.0287     16.982      2.33x      2.34x         -
     40000     parallel        8      0.0289      0.0287      0.0292     16.605      2.28x      2.29x         -
     40000     parallel       12      0.0315      0.0313      0.0324     15.247      2.09x      2.10x         -
     40000     parallel       16      0.0337      0.0329      0.0342     14.223      1.95x      1.96x         -
     40000     parallel       24      0.0436      0.0424      0.0456     11.000      1.51x      1.51x         -

     50000       scalar        -      0.0826      0.0826      0.0845      7.263      1.00x      1.00x         -
     50000   sequential        -      0.0824      0.0824      0.0834      7.277      1.00x      1.00x         -
     50000     parallel        1      0.0827      0.0826      0.0835      7.257      1.00x      1.00x         -
     50000     parallel        2      0.0500      0.0500      0.0505     11.991      1.65x      1.65x         -
     50000     parallel        4      0.0330      0.0330      0.0333     18.185      2.50x      2.50x         -
     50000     parallel        8      0.0315      0.0315      0.0320     19.021      2.62x      2.61x         -
     50000     parallel       12      0.0337      0.0336      0.0341     17.781      2.45x      2.44x         -
     50000     parallel       16      0.0342      0.0340      0.0346     17.568      2.42x      2.41x         -
     50000     parallel       24      0.0438      0.0436      0.0441     13.694      1.89x      1.88x         -

     75000       scalar        -      0.1237      0.1237      0.1255      7.275      1.00x      1.00x         -
     75000   sequential        -      0.1237      0.1237      0.1247      7.275      1.00x      1.00x         -
     75000     parallel        1      0.1238      0.1237      0.1252      7.272      1.00x      1.00x         -
     75000     parallel        2      0.0706      0.0706      0.0715     12.741      1.75x      1.75x         -
     75000     parallel        4      0.0439      0.0425      0.0445     20.485      2.82x      2.82x         -
     75000     parallel        8      0.0381      0.0377      0.0435     23.622      3.25x      3.25x         -
     75000     parallel       12      0.0382      0.0356      0.0399     23.569      3.24x      3.24x         -
     75000     parallel       16      0.0408      0.0382      0.0418     22.065      3.03x      3.03x         -
     75000     parallel       24      0.0474      0.0465      0.0485     18.972      2.61x      2.61x         -

    100000       scalar        -      0.1649      0.1648      0.1678      7.278      1.00x      1.00x         -
    100000   sequential        -      0.1648      0.1648      0.1666      7.280      1.00x      1.00x         -
    100000     parallel        1      0.1649      0.1649      0.1654      7.276      1.00x      1.00x         -
    100000     parallel        2      0.0907      0.0907      0.0914     13.227      1.82x      1.82x         -
    100000     parallel        4      0.0525      0.0524      0.0532     22.878      3.14x      3.14x         -
    100000     parallel        8      0.0406      0.0405      0.0410     29.550      4.06x      4.06x         -
    100000     parallel       12      0.0394      0.0390      0.0397     30.493      4.19x      4.19x         -
    100000     parallel       16      0.0403      0.0401      0.0411     29.778      4.09x      4.09x         -
    100000     parallel       24      0.0489      0.0485      0.0515     24.559      3.37x      3.37x         -

    200000       scalar        -      0.3299      0.3299      0.3342      7.275      1.00x      1.00x         -
    200000   sequential        -      0.3301      0.3298      0.3337      7.269      1.00x      1.00x         -
    200000     parallel        1      0.3300      0.3299      0.3306      7.274      1.00x      1.00x         -
    200000     parallel        2      0.1700      0.1698      0.1730     14.120      1.94x      1.94x         -
    200000     parallel        4      0.0931      0.0927      0.0938     25.766      3.54x      3.54x         -
    200000     parallel        8      0.0617      0.0615      0.0632     38.920      5.35x      5.35x         -
    200000     parallel       12      0.0533      0.0529      0.0576     45.055      6.19x      6.20x         -
    200000     parallel       16      0.0537      0.0535      0.0553     44.698      6.14x      6.15x         -
    200000     parallel       24      0.0591      0.0586      0.0602     40.580      5.58x      5.58x         -

    500000       scalar        -      0.8249      0.8243      0.8321      7.274      1.00x      1.00x         -
    500000   sequential        -      0.8249      0.8243      0.8318      7.273      1.00x      1.00x         -
    500000     parallel        1      0.8252      0.8245      0.8436      7.271      1.00x      1.00x         -
    500000     parallel        2      0.4068      0.4067      0.4070     14.750      2.03x      2.03x         -
    500000     parallel        4      0.2122      0.2117      0.2140     28.273      3.89x      3.89x         -
    500000     parallel        8      0.1266      0.1257      0.1317     47.384      6.51x      6.51x         -
    500000     parallel       12      0.1147      0.1115      0.1210     52.325      7.19x      7.19x         -
    500000     parallel       16      0.1006      0.1003      0.1012     59.633      8.20x      8.20x         -
    500000     parallel       24      0.0880      0.0865      0.0909     68.206      9.38x      9.38x         -

   1000000       scalar        -      1.6517      1.6485      1.6742      7.265      1.00x      1.00x         -
   1000000   sequential        -      1.6516      1.6502      1.6653      7.266      1.00x      1.00x         -
   1000000     parallel        1      1.6519      1.6506      1.6660      7.265      1.00x      1.00x         -
   1000000     parallel        2      0.8018      0.8009      0.8032     14.965      2.06x      2.06x         -
   1000000     parallel        4      0.4094      0.4093      0.4254     29.308      4.03x      4.03x         -
   1000000     parallel        8      0.2263      0.2256      0.2287     53.025      7.30x      7.30x         -
   1000000     parallel       12      0.1640      0.1634      0.1669     73.190     10.07x     10.07x         -
   1000000     parallel       16      0.1612      0.1602      0.1615     74.450     10.25x     10.25x         -
   1000000     parallel       24      0.1376      0.1350      0.1656     87.193     12.00x     12.00x         -

   2000000       scalar        -      3.3194      3.3100      3.3294      7.230      1.00x      1.00x         -
   2000000   sequential        -      3.3056      3.3028      3.3589      7.260      1.00x      1.00x         -
   2000000     parallel        1      3.3062      3.3011      3.3355      7.259      1.00x      1.00x         -
   2000000     parallel        2      1.5925      1.5917      1.5978     15.071      2.08x      2.08x         -
   2000000     parallel        4      0.8092      0.8042      0.8158     29.657      4.10x      4.08x         -
   2000000     parallel        8      0.4442      0.4439      0.4513     54.033      7.47x      7.44x         -
   2000000     parallel       12      0.4090      0.4022      0.4288     58.673      8.11x      8.08x         -
   2000000     parallel       16      0.3262      0.3237      0.3301     73.577     10.18x     10.13x         -
   2000000     parallel       24      0.2401      0.2305      0.2954     99.971     13.83x     13.77x         -

   5000000       scalar        -      8.3666      8.3250      8.3870      7.171      1.00x      1.00x         -
   5000000   sequential        -      8.3665      8.3255      8.3853      7.171      1.00x      1.00x         -
   5000000     parallel        1      8.3655      8.3322      8.3782      7.172      1.00x      1.00x         -
   5000000     parallel        2      4.0288      3.9996      4.1129     14.893      2.08x      2.08x         -
   5000000     parallel        4      2.0312      2.0227      2.2760     29.539      4.12x      4.12x         -
   5000000     parallel        8      1.2588      1.1432      1.3468     47.664      6.65x      6.65x         -
   5000000     parallel       12      1.2689      1.1927      1.5401     47.285      6.59x      6.59x         -
   5000000     parallel       16      1.2974      1.1897      1.5906     46.248      6.45x      6.45x         -
   5000000     parallel       24      1.2411      1.1175      1.3169     48.342      6.74x      6.74x         -

  10000000       scalar        -     16.7119     16.6789     16.7878      7.181      1.00x      1.00x      1.04
  10000000   sequential        -     16.7037     16.6811     16.8388      7.184      1.00x      1.00x      0.93
  10000000     parallel        1     16.7258     16.6858     17.5614      7.175      1.00x      1.00x      1.13
  10000000     parallel        2      8.0579      8.0205      8.2087     14.892      2.07x      2.07x         -
  10000000     parallel        4      4.0847      4.0753      4.1632     29.378      4.09x      4.09x         -
  10000000     parallel        8      2.8182      2.4866      3.4930     42.580      5.93x      5.93x         -
  10000000     parallel       12      3.2683      3.1459      3.4030     36.716      5.11x      5.11x         -
  10000000     parallel       16      3.6919      3.6014      3.8869     32.504      4.53x      4.52x         -
  10000000     parallel       24      3.5730      3.3066      3.7500     33.585      4.68x      4.67x         -

  33554432       scalar        -     56.4043     56.3258     56.4954      7.139      1.00x      1.02x      0.95
  33554432   sequential        -     57.2627     57.1778     57.5194      7.032      0.99x      1.00x      1.03
  33554432     parallel        1     56.4137     56.2489     56.5243      7.138      1.00x      1.02x      0.99
  33554432     parallel        2     27.2593     27.1198     28.1860     14.771      2.07x      2.10x      2.03
  33554432     parallel        4     15.2716     14.7801     20.6739     26.366      3.69x      3.75x      4.20
  33554432     parallel        8     13.2749     13.2138     14.0754     30.332      4.25x      4.31x      6.35
  33554432     parallel       12     13.7105     13.6128     15.3269     29.368      4.11x      4.18x     10.27
  33554432     parallel       16     14.1547     13.9732     14.6339     28.447      3.98x      4.05x     14.22
  33554432     parallel       24     14.4558     14.0607     16.0789     27.854      3.90x      3.96x     23.97


== bias_add ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0004      0.0004      0.0004     21.701      1.00x      1.01x         -
      1000   sequential        -      0.0004      0.0004      0.0004     21.538      0.99x      1.00x         -
      1000     parallel        1      0.0006      0.0006      0.0006     13.653      0.63x      0.63x         -
      1000     parallel        2      0.0006      0.0006      0.0006     13.653      0.63x      0.63x         -
      1000     parallel        4      0.0006      0.0006      0.0006     13.650      0.63x      0.63x         -
      1000     parallel        8      0.0006      0.0006      0.0006     13.653      0.63x      0.63x         -
      1000     parallel       12      0.0006      0.0006      0.0006     13.641      0.63x      0.63x         -
      1000     parallel       16      0.0006      0.0006      0.0006     13.647      0.63x      0.63x         -
      1000     parallel       24      0.0006      0.0006      0.0006     13.652      0.63x      0.63x         -

     10000       scalar        -      0.0016      0.0016      0.0016     49.322      1.00x      1.00x         -
     10000   sequential        -      0.0016      0.0016      0.0017     49.328      1.00x      1.00x         -
     10000     parallel        1      0.0027      0.0027      0.0027     29.635      0.60x      0.60x         -
     10000     parallel        2      0.0027      0.0027      0.0027     29.652      0.60x      0.60x         -
     10000     parallel        4      0.0027      0.0027      0.0027     29.638      0.60x      0.60x         -
     10000     parallel        8      0.0027      0.0027      0.0027     29.633      0.60x      0.60x         -
     10000     parallel       12      0.0027      0.0027      0.0027     29.650      0.60x      0.60x         -
     10000     parallel       16      0.0027      0.0027      0.0027     29.655      0.60x      0.60x         -
     10000     parallel       24      0.0027      0.0027      0.0027     29.633      0.60x      0.60x         -

     32768       scalar        -      0.0032      0.0032      0.0032     82.050      1.00x      0.94x         -
     32768   sequential        -      0.0030      0.0030      0.0031     86.838      1.06x      1.00x         -
     32768     parallel        1      0.0034      0.0034      0.0035     76.298      0.93x      0.88x         -
     32768     parallel        2      0.0034      0.0034      0.0036     76.367      0.93x      0.88x         -
     32768     parallel        4      0.0034      0.0034      0.0035     76.358      0.93x      0.88x         -
     32768     parallel        8      0.0034      0.0034      0.0035     76.283      0.93x      0.88x         -
     32768     parallel       12      0.0034      0.0034      0.0035     76.415      0.93x      0.88x         -
     32768     parallel       16      0.0034      0.0034      0.0035     76.311      0.93x      0.88x         -
     32768     parallel       24      0.0035      0.0034      0.0040     75.789      0.92x      0.87x         -

     32769       scalar        -      0.0024      0.0024      0.0024    108.696      1.00x      0.99x         -
     32769   sequential        -      0.0024      0.0024      0.0024    109.289      1.01x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0024    108.639      1.00x      0.99x         -
     32769     parallel        2      0.0158      0.0158      0.0161     16.636      0.15x      0.15x         -
     32769     parallel        4      0.0150      0.0150      0.0151     17.440      0.16x      0.16x         -
     32769     parallel        8      0.0221      0.0221      0.0227     11.837      0.11x      0.11x         -
     32769     parallel       12      0.0278      0.0276      0.0288      9.421      0.09x      0.09x         -
     32769     parallel       16      0.0300      0.0297      0.0305      8.736      0.08x      0.08x         -
     32769     parallel       24      0.0371      0.0359      0.0378      7.059      0.06x      0.06x         -

     40000       scalar        -      0.0039      0.0039      0.0039     82.499      1.00x      1.00x         -
     40000   sequential        -      0.0039      0.0039      0.0039     82.429      1.00x      1.00x         -
     40000     parallel        1      0.0049      0.0049      0.0050     64.790      0.79x      0.79x         -
     40000     parallel        2      0.0262      0.0254      0.0272     12.204      0.15x      0.15x         -
     40000     parallel        4      0.0216      0.0198      0.0260     14.839      0.18x      0.18x         -
     40000     parallel        8      0.0242      0.0235      0.0260     13.242      0.16x      0.16x         -
     40000     parallel       12      0.0287      0.0285      0.0289     11.166      0.14x      0.14x         -
     40000     parallel       16      0.0305      0.0304      0.0309     10.507      0.13x      0.13x         -
     40000     parallel       24      0.0399      0.0392      0.0429      8.021      0.10x      0.10x         -

     50000       scalar        -      0.0081      0.0081      0.0082     49.592      1.00x      1.00x         -
     50000   sequential        -      0.0081      0.0081      0.0082     49.592      1.00x      1.00x         -
     50000     parallel        1      0.0135      0.0134      0.0136     29.732      0.60x      0.60x         -
     50000     parallel        2      0.0293      0.0290      0.0299     13.674      0.28x      0.28x         -
     50000     parallel        4      0.0221      0.0219      0.0223     18.143      0.37x      0.37x         -
     50000     parallel        8      0.0258      0.0249      0.0272     15.496      0.31x      0.31x         -
     50000     parallel       12      0.0297      0.0294      0.0299     13.482      0.27x      0.27x         -
     50000     parallel       16      0.0315      0.0313      0.0323     12.709      0.26x      0.26x         -
     50000     parallel       24      0.0415      0.0412      0.0419      9.640      0.19x      0.19x         -

     75000       scalar        -      0.0272      0.0271      0.0276     22.066      1.00x      1.00x         -
     75000   sequential        -      0.0271      0.0270      0.0278     22.103      1.00x      1.00x         -
     75000     parallel        1      0.0433      0.0432      0.0443     13.867      0.63x      0.63x         -
     75000     parallel        2      0.0383      0.0380      0.0387     15.674      0.71x      0.71x         -
     75000     parallel        4      0.0270      0.0266      0.0276     22.202      1.01x      1.00x         -
     75000     parallel        8      0.0280      0.0269      0.0291     21.434      0.97x      0.97x         -
     75000     parallel       12      0.0323      0.0321      0.0329     18.562      0.84x      0.84x         -
     75000     parallel       16      0.0331      0.0329      0.0337     18.123      0.82x      0.82x         -
     75000     parallel       24      0.0433      0.0428      0.0445     13.844      0.63x      0.63x         -

    100000       scalar        -      0.0120      0.0119      0.0121     66.727      1.00x      1.00x         -
    100000   sequential        -      0.0120      0.0119      0.0121     66.666      1.00x      1.00x         -
    100000     parallel        1      0.0173      0.0173      0.0175     46.134      0.69x      0.69x         -
    100000     parallel        2      0.0484      0.0482      0.0493     16.546      0.25x      0.25x         -
    100000     parallel        4      0.0312      0.0311      0.0315     25.662      0.38x      0.38x         -
    100000     parallel        8      0.0294      0.0293      0.0300     27.217      0.41x      0.41x         -
    100000     parallel       12      0.0348      0.0348      0.0355     22.977      0.34x      0.34x         -
    100000     parallel       16      0.0344      0.0341      0.0348     23.256      0.35x      0.35x         -
    100000     parallel       24      0.0449      0.0443      0.0451     17.806      0.27x      0.27x         -

    200000       scalar        -      0.0208      0.0207      0.0210     77.028      1.00x      1.00x         -
    200000   sequential        -      0.0209      0.0208      0.0211     76.662      1.00x      1.00x         -
    200000     parallel        1      0.0252      0.0251      0.0256     63.475      0.82x      0.83x         -
    200000     parallel        2      0.0864      0.0862      0.0874     18.511      0.24x      0.24x         -
    200000     parallel        4      0.0510      0.0508      0.0516     31.372      0.41x      0.41x         -
    200000     parallel        8      0.0394      0.0392      0.0402     40.588      0.53x      0.53x         -
    200000     parallel       12      0.0435      0.0434      0.0445     36.749      0.48x      0.48x         -
    200000     parallel       16      0.0432      0.0425      0.0477     37.030      0.48x      0.48x         -
    200000     parallel       24      0.0538      0.0524      0.0559     29.724      0.39x      0.39x         -

    500000       scalar        -      0.0595      0.0594      0.0597     67.231      1.00x      1.01x         -
    500000   sequential        -      0.0602      0.0599      0.0607     66.496      0.99x      1.00x         -
    500000     parallel        1      0.0871      0.0870      0.1026     45.936      0.68x      0.69x         -
    500000     parallel        2      0.1969      0.1952      0.2002     20.317      0.30x      0.31x         -
    500000     parallel        4      0.1076      0.1071      0.1087     37.165      0.55x      0.56x         -
    500000     parallel        8      0.0680      0.0676      0.0696     58.855      0.88x      0.89x         -
    500000     parallel       12      0.0770      0.0667      0.0775     51.924      0.77x      0.78x         -
    500000     parallel       16      0.0686      0.0685      0.0710     58.324      0.87x      0.88x         -
    500000     parallel       24      0.0741      0.0724      0.0855     54.001      0.80x      0.81x         -

   1000000       scalar        -      0.1058      0.1056      0.1065     75.635      1.00x      1.01x         -
   1000000   sequential        -      0.1070      0.1047      0.1098     74.749      0.99x      1.00x         -
   1000000     parallel        1      0.1275      0.1271      0.1291     62.729      0.83x      0.84x         -
   1000000     parallel        2      0.3848      0.3810      0.3903     20.793      0.27x      0.28x         -
   1000000     parallel        4      0.2025      0.2019      0.2048     39.507      0.52x      0.53x         -
   1000000     parallel        8      0.1163      0.1157      0.1198     68.766      0.91x      0.92x         -
   1000000     parallel       12      0.1062      0.1023      0.1135     75.297      1.00x      1.01x         -
   1000000     parallel       16      0.1112      0.1110      0.1120     71.931      0.95x      0.96x         -
   1000000     parallel       24      0.1164      0.1116      0.1231     68.705      0.91x      0.92x         -

   2000000       scalar        -      0.2049      0.2024      0.2133     78.092      1.00x      1.01x         -
   2000000   sequential        -      0.2071      0.2063      0.2283     77.248      0.99x      1.00x         -
   2000000     parallel        1      0.2362      0.2338      0.2465     67.741      0.87x      0.88x         -
   2000000     parallel        2      0.7642      0.7573      0.7714     20.938      0.27x      0.27x         -
   2000000     parallel        4      0.3908      0.3896      0.4066     40.943      0.52x      0.53x         -
   2000000     parallel        8      0.2113      0.2101      0.2149     75.731      0.97x      0.98x         -
   2000000     parallel       12      0.1948      0.1873      0.2237     82.125      1.05x      1.06x         -
   2000000     parallel       16      0.1982      0.1972      0.2047     80.714      1.03x      1.04x         -
   2000000     parallel       24      0.2093      0.2049      0.2201     76.432      0.98x      0.99x         -

   5000000       scalar        -      1.0516      1.0313      1.1026     38.037      1.00x      0.99x         -
   5000000   sequential        -      1.0424      1.0274      1.0689     38.373      1.01x      1.00x         -
   5000000     parallel        1      1.0800      1.0501      1.1220     37.039      0.97x      0.97x         -
   5000000     parallel        2      1.9627      1.9424      2.0830     20.380      0.54x      0.53x         -
   5000000     parallel        4      1.0083      0.9969      1.4646     39.671      1.04x      1.03x         -
   5000000     parallel        8      0.6170      0.5576      0.6405     64.825      1.70x      1.69x         -
   5000000     parallel       12      0.5614      0.5439      0.6138     71.251      1.87x      1.86x         -
   5000000     parallel       16      0.4829      0.4554      0.8902     82.831      2.18x      2.16x         -
   5000000     parallel       24      0.5106      0.4572      0.6050     78.342      2.06x      2.04x         -

  10000000       scalar        -      2.5791      2.5035      2.6734     31.019      1.00x      1.00x         -
  10000000   sequential        -      2.5735      2.4854      2.9045     31.086      1.00x      1.00x         -
  10000000     parallel        1      2.5920      2.4943      2.8447     30.864      1.00x      0.99x         -
  10000000     parallel        2      4.0069      3.9569      4.2567     19.966      0.64x      0.64x         -
  10000000     parallel        4      2.2490      2.2048      2.4867     35.572      1.15x      1.14x         -
  10000000     parallel        8      1.4716      1.4314      1.5347     54.361      1.75x      1.75x         -
  10000000     parallel       12      2.0468      1.9173      2.4668     39.086      1.26x      1.26x         -
  10000000     parallel       16      2.0887      1.8138      2.2771     38.302      1.23x      1.23x         -
  10000000     parallel       24      2.3314      2.0716      2.6432     34.314      1.11x      1.10x         -

  33554432       scalar        -     11.2815     11.1487     11.7084     23.794      1.00x      0.95x      0.92
  33554432   sequential        -     10.7124     10.5603     14.0921     25.058      1.05x      1.00x      0.93
  33554432     parallel        1     11.1743     10.9964     11.5114     24.023      1.01x      0.96x      1.08
  33554432     parallel        2     16.2302     16.1269     16.3783     16.539      0.70x      0.66x      2.03
  33554432     parallel        4     10.3708     10.1478     10.6180     25.884      1.09x      1.03x         -
  33554432     parallel        8      9.2939      9.2477     10.0569     28.883      1.21x      1.15x         -
  33554432     parallel       12      9.8571      9.7230     10.0045     27.233      1.14x      1.09x         -
  33554432     parallel       16     10.2608      9.8508     10.6052     26.161      1.10x      1.04x         -
  33554432     parallel       24     10.1433      9.9361     11.2706     26.464      1.11x      1.06x         -


== dot ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0009      0.0009      0.0009      9.356      1.00x      1.00x         -
      1000   sequential        -      0.0009      0.0009      0.0009      9.358      1.00x      1.00x         -
      1000     parallel        1      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -
      1000     parallel        2      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -
      1000     parallel        4      0.0009      0.0009      0.0009      9.322      1.00x      1.00x         -
      1000     parallel        8      0.0009      0.0009      0.0009      9.335      1.00x      1.00x         -
      1000     parallel       12      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -
      1000     parallel       16      0.0009      0.0009      0.0009      9.335      1.00x      1.00x         -
      1000     parallel       24      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -

     10000       scalar        -      0.0086      0.0086      0.0087      9.313      1.00x      1.00x         -
     10000   sequential        -      0.0086      0.0086      0.0087      9.313      1.00x      1.00x         -
     10000     parallel        1      0.0086      0.0086      0.0087      9.311      1.00x      1.00x         -
     10000     parallel        2      0.0086      0.0086      0.0087      9.311      1.00x      1.00x         -
     10000     parallel        4      0.0086      0.0086      0.0087      9.306      1.00x      1.00x         -
     10000     parallel        8      0.0086      0.0086      0.0086      9.310      1.00x      1.00x         -
     10000     parallel       12      0.0086      0.0086      0.0086      9.310      1.00x      1.00x         -
     10000     parallel       16      0.0086      0.0086      0.0086      9.307      1.00x      1.00x         -
     10000     parallel       24      0.0086      0.0086      0.0086      9.311      1.00x      1.00x         -

     32768       scalar        -      0.0281      0.0281      0.0283      9.313      1.00x      1.00x         -
     32768   sequential        -      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32768     parallel        1      0.0282      0.0281      0.0283      9.312      1.00x      1.00x         -
     32768     parallel        2      0.0282      0.0281      0.0283      9.312      1.00x      1.00x         -
     32768     parallel        4      0.0282      0.0281      0.0283      9.311      1.00x      1.00x         -
     32768     parallel        8      0.0282      0.0281      0.0283      9.312      1.00x      1.00x         -
     32768     parallel       12      0.0282      0.0281      0.0283      9.312      1.00x      1.00x         -
     32768     parallel       16      0.0282      0.0281      0.0285      9.312      1.00x      1.00x         -
     32768     parallel       24      0.0282      0.0281      0.0283      9.312      1.00x      1.00x         -

     32769       scalar        -      0.0282      0.0281      0.0283      9.312      1.00x      1.00x         -
     32769   sequential        -      0.0282      0.0281      0.0283      9.313      1.00x      1.00x         -
     32769     parallel        1      0.0282      0.0281      0.0285      9.311      1.00x      1.00x         -
     32769     parallel        2      0.0248      0.0247      0.0251     10.570      1.14x      1.13x         -
     32769     parallel        4      0.0195      0.0194      0.0197     13.463      1.45x      1.45x         -
     32769     parallel        8      0.0237      0.0236      0.0242     11.053      1.19x      1.19x         -
     32769     parallel       12      0.0281      0.0281      0.0290      9.315      1.00x      1.00x         -
     32769     parallel       16      0.0305      0.0298      0.0309      8.604      0.92x      0.92x         -
     32769     parallel       24      0.0368      0.0362      0.0370      7.120      0.76x      0.76x         -

     40000       scalar        -      0.0344      0.0344      0.0348      9.311      1.00x      1.00x         -
     40000   sequential        -      0.0344      0.0344      0.0348      9.313      1.00x      1.00x         -
     40000     parallel        1      0.0344      0.0344      0.0348      9.312      1.00x      1.00x         -
     40000     parallel        2      0.0280      0.0279      0.0283     11.446      1.23x      1.23x         -
     40000     parallel        4      0.0211      0.0210      0.0214     15.179      1.63x      1.63x         -
     40000     parallel        8      0.0250      0.0247      0.0253     12.809      1.38x      1.38x         -
     40000     parallel       12      0.0289      0.0285      0.0305     11.068      1.19x      1.19x         -
     40000     parallel       16      0.0319      0.0307      0.0356     10.018      1.08x      1.08x         -
     40000     parallel       24      0.0405      0.0396      0.0414      7.906      0.85x      0.85x         -

     50000       scalar        -      0.0430      0.0429      0.0430      9.308      1.00x      1.00x         -
     50000   sequential        -      0.0430      0.0429      0.0431      9.309      1.00x      1.00x         -
     50000     parallel        1      0.0430      0.0429      0.0434      9.312      1.00x      1.00x         -
     50000     parallel        2      0.0320      0.0314      0.0341     12.488      1.34x      1.34x         -
     50000     parallel        4      0.0242      0.0240      0.0246     16.549      1.78x      1.78x         -
     50000     parallel        8      0.0255      0.0252      0.0276     15.674      1.68x      1.68x         -
     50000     parallel       12      0.0296      0.0295      0.0298     13.507      1.45x      1.45x         -
     50000     parallel       16      0.0318      0.0316      0.0321     12.591      1.35x      1.35x         -
     50000     parallel       24      0.0392      0.0388      0.0401     10.211      1.10x      1.10x         -

     75000       scalar        -      0.0645      0.0645      0.0655      9.296      1.00x      1.00x         -
     75000   sequential        -      0.0645      0.0645      0.0654      9.296      1.00x      1.00x         -
     75000     parallel        1      0.0645      0.0645      0.0654      9.296      1.00x      1.00x         -
     75000     parallel        2      0.0433      0.0433      0.0437     13.841      1.49x      1.49x         -
     75000     parallel        4      0.0288      0.0287      0.0290     20.856      2.24x      2.24x         -
     75000     parallel        8      0.0286      0.0286      0.0293     20.959      2.25x      2.25x         -
     75000     parallel       12      0.0315      0.0310      0.0327     19.077      2.05x      2.05x         -
     75000     parallel       16      0.0334      0.0331      0.0347     17.987      1.93x      1.93x         -
     75000     parallel       24      0.0417      0.0412      0.0421     14.396      1.55x      1.55x         -

    100000       scalar        -      0.0861      0.0860      0.0870      9.293      1.00x      1.00x         -
    100000   sequential        -      0.0861      0.0860      0.0873      9.293      1.00x      1.00x         -
    100000     parallel        1      0.0861      0.0860      0.0870      9.294      1.00x      1.00x         -
    100000     parallel        2      0.0547      0.0547      0.0555     14.613      1.57x      1.57x         -
    100000     parallel        4      0.0344      0.0343      0.0350     23.256      2.50x      2.50x         -
    100000     parallel        8      0.0322      0.0311      0.0323     24.874      2.68x      2.68x         -
    100000     parallel       12      0.0329      0.0326      0.0337     24.322      2.62x      2.62x         -
    100000     parallel       16      0.0337      0.0334      0.0341     23.715      2.55x      2.55x         -
    100000     parallel       24      0.0426      0.0422      0.0438     18.761      2.02x      2.02x         -

    200000       scalar        -      0.1720      0.1720      0.1741      9.300      1.00x      1.00x         -
    200000   sequential        -      0.1721      0.1720      0.1737      9.295      1.00x      1.00x         -
    200000     parallel        1      0.1720      0.1719      0.1723      9.303      1.00x      1.00x         -
    200000     parallel        2      0.0991      0.0984      0.1028     16.146      1.74x      1.74x         -
    200000     parallel        4      0.0564      0.0563      0.0569     28.363      3.05x      3.05x         -
    200000     parallel        8      0.0430      0.0427      0.0448     37.235      4.00x      4.01x         -
    200000     parallel       12      0.0414      0.0399      0.0447     38.618      4.15x      4.15x         -
    200000     parallel       16      0.0396      0.0393      0.0405     40.415      4.35x      4.35x         -
    200000     parallel       24      0.0463      0.0461      0.0516     34.527      3.71x      3.71x         -

    500000       scalar        -      0.4302      0.4298      0.4337      9.298      1.00x      1.00x         -
    500000   sequential        -      0.4301      0.4298      0.4336      9.300      1.00x      1.00x         -
    500000     parallel        1      0.4301      0.4301      0.4333      9.300      1.00x      1.00x         -
    500000     parallel        2      0.2273      0.2272      0.2293     17.597      1.89x      1.89x         -
    500000     parallel        4      0.1220      0.1219      0.1230     32.796      3.53x      3.53x         -
    500000     parallel        8      0.0756      0.0751      0.0781     52.893      5.69x      5.69x         -
    500000     parallel       12      0.0621      0.0616      0.0624     64.443      6.93x      6.93x         -
    500000     parallel       16      0.0564      0.0558      0.0566     70.977      7.63x      7.63x         -
    500000     parallel       24      0.0581      0.0577      0.0622     68.902      7.41x      7.41x         -

   1000000       scalar        -      0.8602      0.8596      0.8674      9.300      1.00x      1.00x         -
   1000000   sequential        -      0.8602      0.8594      0.8672      9.300      1.00x      1.00x         -
   1000000     parallel        1      0.8600      0.8594      0.8667      9.302      1.00x      1.00x         -
   1000000     parallel        2      0.4421      0.4420      0.4444     18.096      1.95x      1.95x         -
   1000000     parallel        4      0.2297      0.2294      0.2314     34.829      3.74x      3.75x         -
   1000000     parallel        8      0.1297      0.1294      0.1345     61.693      6.63x      6.63x         -
   1000000     parallel       12      0.0985      0.0982      0.0996     81.182      8.73x      8.73x         -
   1000000     parallel       16      0.0847      0.0843      0.0862     94.402     10.15x     10.15x         -
   1000000     parallel       24      0.0770      0.0767      0.0840    103.875     11.17x     11.17x         -

   2000000       scalar        -      1.7202      1.7190      1.7358      9.301      1.00x      1.00x         -
   2000000   sequential        -      1.7203      1.7189      1.7442      9.301      1.00x      1.00x         -
   2000000     parallel        1      1.7213      1.7200      1.7352      9.295      1.00x      1.00x         -
   2000000     parallel        2      0.8732      0.8711      0.8769     18.324      1.97x      1.97x         -
   2000000     parallel        4      0.4439      0.4437      0.4510     36.043      3.88x      3.88x         -
   2000000     parallel        8      0.2372      0.2365      0.2396     67.450      7.25x      7.25x         -
   2000000     parallel       12      0.1726      0.1708      0.1787     92.723      9.97x      9.97x         -
   2000000     parallel       16      0.1430      0.1421      0.1594    111.898     12.03x     12.03x         -
   2000000     parallel       24      0.1160      0.1154      0.1209    137.950     14.83x     14.83x         -

   5000000       scalar        -      4.4170      4.3963      4.4766      9.056      1.00x      1.00x         -
   5000000   sequential        -      4.4240      4.4169      4.4786      9.042      1.00x      1.00x         -
   5000000     parallel        1      4.4201      4.4151      4.4750      9.050      1.00x      1.00x         -
   5000000     parallel        2      2.2251      2.2169      2.2684     17.977      1.99x      1.99x         -
   5000000     parallel        4      1.1291      1.1258      1.1542     35.426      3.91x      3.92x         -
   5000000     parallel        8      0.5731      0.5675      0.5839     69.799      7.71x      7.72x         -
   5000000     parallel       12      0.4042      0.3994      0.4067     98.955     10.93x     10.94x         -
   5000000     parallel       16      0.3506      0.3455      0.3601    114.090     12.60x     12.62x         -
   5000000     parallel       24      0.2892      0.2640      0.3062    138.313     15.27x     15.30x         -

  10000000       scalar        -      8.9383      8.8868      8.9551      8.950      1.00x      1.00x         -
  10000000   sequential        -      8.9410      8.8969      8.9767      8.948      1.00x      1.00x         -
  10000000     parallel        1      8.9181      8.8959      8.9513      8.971      1.00x      1.00x         -
  10000000     parallel        2      4.5074      4.4896      4.5487     17.749      1.98x      1.98x         -
  10000000     parallel        4      2.2934      2.2886      2.3004     34.883      3.90x      3.90x         -
  10000000     parallel        8      1.1608      1.1538      1.6734     68.915      7.70x      7.70x         -
  10000000     parallel       12      1.3297      1.2516      1.4019     60.162      6.72x      6.72x         -
  10000000     parallel       16      1.4114      1.3235      1.4365     56.683      6.33x      6.34x         -
  10000000     parallel       24      1.2844      1.2281      1.6356     62.286      6.96x      6.96x         -

  33554432       scalar        -     29.9038     29.8636     31.0045      8.977      1.00x      1.00x      0.98
  33554432   sequential        -     29.8945     29.8509     29.9862      8.979      1.00x      1.00x      0.99
  33554432     parallel        1     29.8841     29.8445     29.9013      8.983      1.00x      1.00x      0.99
  33554432     parallel        2     15.0903     15.0505     15.1258     17.789      1.98x      1.98x      2.07
  33554432     parallel        4      7.8871      7.7914      8.8022     34.035      3.79x      3.79x         -
  33554432     parallel        8      6.5662      6.0928      7.3941     40.881      4.55x      4.55x         -
  33554432     parallel       12      6.8942      6.6891      7.1984     38.936      4.34x      4.34x         -
  33554432     parallel       16      6.9093      6.6531      7.1379     38.851      4.33x      4.33x         -
  33554432     parallel       24      6.8612      6.5801      7.1064     39.124      4.36x      4.36x         -
```

</details>

<details>
<summary>Full output: cpu_parallel_benchmark, run 2</summary>

```
=== Scalar vs sequential vs parallel on the CPU (float32) ===
cpu: AMD Ryzen 9 5900X 12-Core Processor | hardware threads: 24 | OpenMP threads: 24
compiler: msvc 1951 | OpenMP: yes (2.0) | AVX2: yes | build: Release
timing: 1 warm-up call(s), median of 9 samples (3 if a call takes over 0.2 s), each sample at least 2 ms
for comparable numbers: plugged in, High performance power plan, other programs closed, run twice
impl: scalar = sequential.cpp without auto-vectorization; sequential = the same code with the compiler's SIMD; parallel = the parallel kernel with N threads
vs scalar / vs seq = that baseline's median time / this row's median time
parallel cutoff: 32768 elements
axpy: alpha=1.250000, fused multiply-add
bias_add: features = largest power-of-two divisor of n up to 256; the first features entries of b are the bias; GB/s counts 8*n + 4*features bytes

== add ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0002    142.444      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    142.073      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    136.780      0.96x      0.96x         -
      1000     parallel        2      0.0001      0.0001      0.0001    136.604      0.96x      0.96x         -
      1000     parallel        4      0.0001      0.0001      0.0001    136.699      0.96x      0.96x         -
      1000     parallel        8      0.0001      0.0001      0.0001    136.519      0.96x      0.96x         -
      1000     parallel       12      0.0001      0.0001      0.0001    136.833      0.96x      0.96x         -
      1000     parallel       16      0.0001      0.0001      0.0001    136.311      0.96x      0.96x         -
      1000     parallel       24      0.0001      0.0001      0.0001    136.467      0.96x      0.96x         -

     10000       scalar        -      0.0011      0.0011      0.0011    110.355      1.00x      1.00x         -
     10000   sequential        -      0.0011      0.0011      0.0011    110.429      1.00x      1.00x         -
     10000     parallel        1      0.0011      0.0011      0.0011    109.719      0.99x      0.99x         -
     10000     parallel        2      0.0011      0.0011      0.0011    109.709      0.99x      0.99x         -
     10000     parallel        4      0.0011      0.0011      0.0011    109.690      0.99x      0.99x         -
     10000     parallel        8      0.0011      0.0011      0.0011    109.602      0.99x      0.99x         -
     10000     parallel       12      0.0011      0.0011      0.0011    109.729      0.99x      0.99x         -
     10000     parallel       16      0.0011      0.0011      0.0011    109.729      0.99x      0.99x         -
     10000     parallel       24      0.0011      0.0011      0.0011    109.641      0.99x      0.99x         -

     32768       scalar        -      0.0035      0.0035      0.0036    110.841      1.00x      1.02x         -
     32768   sequential        -      0.0036      0.0036      0.0036    108.866      0.98x      1.00x         -
     32768     parallel        1      0.0036      0.0036      0.0036    110.292      1.00x      1.01x         -
     32768     parallel        2      0.0036      0.0035      0.0036    110.561      1.00x      1.02x         -
     32768     parallel        4      0.0036      0.0036      0.0036    110.401      1.00x      1.01x         -
     32768     parallel        8      0.0036      0.0035      0.0036    110.601      1.00x      1.02x         -
     32768     parallel       12      0.0036      0.0035      0.0036    110.528      1.00x      1.02x         -
     32768     parallel       16      0.0036      0.0035      0.0036    110.501      1.00x      1.02x         -
     32768     parallel       24      0.0036      0.0035      0.0036    110.561      1.00x      1.02x         -

     32769       scalar        -      0.0036      0.0035      0.0036    110.117      1.00x      1.03x         -
     32769   sequential        -      0.0037      0.0037      0.0077    106.404      0.97x      1.00x         -
     32769     parallel        1      0.0036      0.0036      0.0036    110.262      1.00x      1.04x         -
     32769     parallel        2      0.0162      0.0162      0.0171     24.213      0.22x      0.23x         -
     32769     parallel        4      0.0150      0.0150      0.0152     26.148      0.24x      0.25x         -
     32769     parallel        8      0.0222      0.0221      0.0227     17.696      0.16x      0.17x         -
     32769     parallel       12      0.0280      0.0279      0.0284     14.034      0.13x      0.13x         -
     32769     parallel       16      0.0306      0.0303      0.0317     12.848      0.12x      0.12x         -
     32769     parallel       24      0.0381      0.0374      0.0398     10.322      0.09x      0.10x         -

     40000       scalar        -      0.0045      0.0044      0.0045    107.572      1.00x      1.02x         -
     40000   sequential        -      0.0045      0.0045      0.0046    105.513      0.98x      1.00x         -
     40000     parallel        1      0.0045      0.0045      0.0045    107.577      1.00x      1.02x         -
     40000     parallel        2      0.0166      0.0165      0.0168     28.842      0.27x      0.27x         -
     40000     parallel        4      0.0158      0.0158      0.0161     30.311      0.28x      0.29x         -
     40000     parallel        8      0.0229      0.0228      0.0232     20.971      0.19x      0.20x         -
     40000     parallel       12      0.0285      0.0284      0.0290     16.818      0.16x      0.16x         -
     40000     parallel       16      0.0307      0.0302      0.0317     15.613      0.15x      0.15x         -
     40000     parallel       24      0.0372      0.0363      0.0376     12.907      0.12x      0.12x         -

     50000       scalar        -      0.0061      0.0061      0.0061     98.696      1.00x      1.00x         -
     50000   sequential        -      0.0061      0.0061      0.0061     98.705      1.00x      1.00x         -
     50000     parallel        1      0.0061      0.0061      0.0062     98.197      0.99x      0.99x         -
     50000     parallel        2      0.0194      0.0194      0.0198     30.868      0.31x      0.31x         -
     50000     parallel        4      0.0170      0.0169      0.0177     35.397      0.36x      0.36x         -
     50000     parallel        8      0.0234      0.0233      0.0257     25.617      0.26x      0.26x         -
     50000     parallel       12      0.0295      0.0293      0.0302     20.316      0.21x      0.21x         -
     50000     parallel       16      0.0315      0.0311      0.0317     19.025      0.19x      0.19x         -
     50000     parallel       24      0.0394      0.0375      0.0410     15.247      0.15x      0.15x         -

     75000       scalar        -      0.0094      0.0094      0.0096     95.254      1.00x      0.99x         -
     75000   sequential        -      0.0094      0.0094      0.0094     95.980      1.01x      1.00x         -
     75000     parallel        1      0.0095      0.0094      0.0095     95.022      1.00x      0.99x         -
     75000     parallel        2      0.0244      0.0244      0.0255     36.811      0.39x      0.38x         -
     75000     parallel        4      0.0204      0.0192      0.0206     44.108      0.46x      0.46x         -
     75000     parallel        8      0.0281      0.0264      0.0283     32.081      0.34x      0.33x         -
     75000     parallel       12      0.0325      0.0310      0.0330     27.728      0.29x      0.29x         -
     75000     parallel       16      0.0326      0.0317      0.0336     27.618      0.29x      0.29x         -
     75000     parallel       24      0.0391      0.0388      0.0404     23.004      0.24x      0.24x         -

    100000       scalar        -      0.0127      0.0127      0.0127     94.450      1.00x      0.99x         -
    100000   sequential        -      0.0126      0.0126      0.0127     95.448      1.01x      1.00x         -
    100000     parallel        1      0.0127      0.0127      0.0129     94.622      1.00x      0.99x         -
    100000     parallel        2      0.0289      0.0288      0.0293     41.563      0.44x      0.44x         -
    100000     parallel        4      0.0220      0.0215      0.0229     54.532      0.58x      0.57x         -
    100000     parallel        8      0.0268      0.0267      0.0281     44.711      0.47x      0.47x         -
    100000     parallel       12      0.0341      0.0331      0.0344     35.166      0.37x      0.37x         -
    100000     parallel       16      0.0346      0.0341      0.0349     34.728      0.37x      0.36x         -
    100000     parallel       24      0.0414      0.0409      0.0421     29.014      0.31x      0.30x         -

    200000       scalar        -      0.0254      0.0252      0.0257     94.398      1.00x      1.00x         -
    200000   sequential        -      0.0255      0.0254      0.0257     94.277      1.00x      1.00x         -
    200000     parallel        1      0.0255      0.0254      0.0256     94.288      1.00x      1.00x         -
    200000     parallel        2      0.0459      0.0459      0.0465     52.263      0.55x      0.55x         -
    200000     parallel        4      0.0308      0.0307      0.0312     78.035      0.83x      0.83x         -
    200000     parallel        8      0.0345      0.0344      0.0352     69.654      0.74x      0.74x         -
    200000     parallel       12      0.0395      0.0389      0.0407     60.803      0.64x      0.64x         -
    200000     parallel       16      0.0427      0.0417      0.0446     56.151      0.59x      0.60x         -
    200000     parallel       24      0.0494      0.0483      0.0504     48.591      0.51x      0.52x         -

    500000       scalar        -      0.0630      0.0630      0.0639     95.233      1.00x      1.00x         -
    500000   sequential        -      0.0632      0.0632      0.0641     94.904      1.00x      1.00x         -
    500000     parallel        1      0.0628      0.0628      0.0630     95.537      1.00x      1.01x         -
    500000     parallel        2      0.0971      0.0950      0.0980     61.804      0.65x      0.65x         -
    500000     parallel        4      0.0563      0.0562      0.0585    106.554      1.12x      1.12x         -
    500000     parallel        8      0.0565      0.0556      0.0581    106.115      1.11x      1.12x         -
    500000     parallel       12      0.0623      0.0618      0.0645     96.231      1.01x      1.01x         -
    500000     parallel       16      0.0647      0.0641      0.0681     92.727      0.97x      0.98x         -
    500000     parallel       24      0.0729      0.0701      0.0760     82.305      0.86x      0.87x         -

   1000000       scalar        -      0.1262      0.1258      0.1279     95.073      1.00x      1.01x         -
   1000000   sequential        -      0.1273      0.1271      0.1406     94.275      0.99x      1.00x         -
   1000000     parallel        1      0.1268      0.1265      0.1405     94.633      1.00x      1.00x         -
   1000000     parallel        2      0.1811      0.1806      0.1891     66.280      0.70x      0.70x         -
   1000000     parallel        4      0.0987      0.0985      0.0997    121.581      1.28x      1.29x         -
   1000000     parallel        8      0.0980      0.0906      0.1033    122.398      1.29x      1.30x         -
   1000000     parallel       12      0.0984      0.0911      0.1007    121.920      1.28x      1.29x         -
   1000000     parallel       16      0.1098      0.0910      0.1126    109.283      1.15x      1.16x         -
   1000000     parallel       24      0.1181      0.1125      0.1198    101.633      1.07x      1.08x         -

   2000000       scalar        -      0.2723      0.2530      0.3185     88.146      1.00x      1.12x         -
   2000000   sequential        -      0.3044      0.2538      0.3720     78.834      0.89x      1.00x         -
   2000000     parallel        1      0.2548      0.2534      0.2968     94.196      1.07x      1.19x         -
   2000000     parallel        2      0.3568      0.3513      0.3786     67.267      0.76x      0.85x         -
   2000000     parallel        4      0.1857      0.1828      0.2162    129.236      1.47x      1.64x         -
   2000000     parallel        8      0.1812      0.1668      0.2014    132.482      1.50x      1.68x         -
   2000000     parallel       12      0.1742      0.1606      0.2007    137.788      1.56x      1.75x         -
   2000000     parallel       16      0.2179      0.1807      0.5151    110.145      1.25x      1.40x         -
   2000000     parallel       24      0.2972      0.2618      0.3335     80.750      0.92x      1.02x         -

   5000000       scalar        -      1.5154      1.4665      1.5855     39.592      1.00x      0.99x         -
   5000000   sequential        -      1.4949      1.4772      1.5788     40.138      1.01x      1.00x         -
   5000000     parallel        1      1.4994      1.4489      1.7077     40.015      1.01x      1.00x         -
   5000000     parallel        2      1.4176      1.4064      1.6782     42.325      1.07x      1.05x         -
   5000000     parallel        4      1.1528      1.1078      1.4449     52.045      1.31x      1.30x         -
   5000000     parallel        8      0.9729      0.8804      1.4204     61.671      1.56x      1.54x         -
   5000000     parallel       12      1.2239      0.9838      1.4346     49.024      1.24x      1.22x         -
   5000000     parallel       16      1.2111      0.7831      1.5331     49.542      1.25x      1.23x         -
   5000000     parallel       24      1.1849      1.1485      1.2230     50.635      1.28x      1.26x         -

  10000000       scalar        -      3.5820      3.4751      3.9191     33.501      1.00x      1.00x         -
  10000000   sequential        -      3.5872      3.4728      3.8352     33.452      1.00x      1.00x         -
  10000000     parallel        1      3.7195      3.5695      4.1785     32.262      0.96x      0.96x         -
  10000000     parallel        2      3.3595      3.3055      3.6096     35.720      1.07x      1.07x         -
  10000000     parallel        4      3.5290      3.4828      3.8565     34.004      1.02x      1.02x         -
  10000000     parallel        8      2.5829      2.4752      4.7100     46.459      1.39x      1.39x         -
  10000000     parallel       12      3.2101      2.9117      3.6636     37.382      1.12x      1.12x         -
  10000000     parallel       16      3.4576      3.1881      3.5630     34.706      1.04x      1.04x         -
  10000000     parallel       24      3.4024      3.2379      3.5632     35.269      1.05x      1.05x         -

  33554432       scalar        -     14.9025     14.7374     15.2411     27.019      1.00x      1.01x      1.05
  33554432   sequential        -     15.0210     14.8851     15.9207     26.806      0.99x      1.00x      0.92
  33554432     parallel        1     14.8181     14.7613     16.0213     27.173      1.01x      1.01x      1.04
  33554432     parallel        2     14.4777     13.7760     15.0485     27.812      1.03x      1.04x      1.80
  33554432     parallel        4     14.3750     14.2472     15.7894     28.011      1.04x      1.04x      3.81
  33554432     parallel        8     13.8077     13.4915     14.1523     29.161      1.08x      1.09x      6.04
  33554432     parallel       12     13.7857     13.3589     14.3944     29.208      1.08x      1.09x     11.84
  33554432     parallel       16     14.1912     13.9339     14.5223     28.373      1.05x      1.06x     13.54
  33554432     parallel       24     13.9862     13.8781     14.5682     28.789      1.07x      1.07x     23.05


== sub ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    142.630      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    142.635      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    134.640      0.94x      0.94x         -
      1000     parallel        2      0.0001      0.0001      0.0001    137.507      0.96x      0.96x         -
      1000     parallel        4      0.0001      0.0001      0.0001    137.565      0.96x      0.96x         -
      1000     parallel        8      0.0001      0.0001      0.0001    137.478      0.96x      0.96x         -
      1000     parallel       12      0.0001      0.0001      0.0001    137.531      0.96x      0.96x         -
      1000     parallel       16      0.0001      0.0001      0.0001    137.719      0.97x      0.97x         -
      1000     parallel       24      0.0001      0.0001      0.0001    137.522      0.96x      0.96x         -

     10000       scalar        -      0.0011      0.0011      0.0011    110.340      1.00x      1.00x         -
     10000   sequential        -      0.0011      0.0011      0.0011    110.399      1.00x      1.00x         -
     10000     parallel        1      0.0011      0.0011      0.0011    109.724      0.99x      0.99x         -
     10000     parallel        2      0.0011      0.0011      0.0011    109.763      0.99x      0.99x         -
     10000     parallel        4      0.0011      0.0011      0.0011    109.763      0.99x      0.99x         -
     10000     parallel        8      0.0011      0.0011      0.0011    109.758      0.99x      0.99x         -
     10000     parallel       12      0.0011      0.0011      0.0013    109.758      0.99x      0.99x         -
     10000     parallel       16      0.0011      0.0011      0.0011    109.744      0.99x      0.99x         -
     10000     parallel       24      0.0011      0.0011      0.0011    109.753      0.99x      0.99x         -

     32768       scalar        -      0.0037      0.0036      0.0037    107.563      1.00x      1.00x         -
     32768   sequential        -      0.0037      0.0036      0.0037    107.730      1.00x      1.00x         -
     32768     parallel        1      0.0037      0.0036      0.0041    105.656      0.98x      0.98x         -
     32768     parallel        2      0.0037      0.0036      0.0038    107.389      1.00x      1.00x         -
     32768     parallel        4      0.0037      0.0036      0.0037    107.566      1.00x      1.00x         -
     32768     parallel        8      0.0037      0.0037      0.0037    107.060      1.00x      0.99x         -
     32768     parallel       12      0.0036      0.0036      0.0037    107.794      1.00x      1.00x         -
     32768     parallel       16      0.0037      0.0036      0.0038    107.552      1.00x      1.00x         -
     32768     parallel       24      0.0036      0.0036      0.0037    107.733      1.00x      1.00x         -

     32769       scalar        -      0.0036      0.0036      0.0037    107.861      1.00x      1.00x         -
     32769   sequential        -      0.0036      0.0036      0.0037    108.226      1.00x      1.00x         -
     32769     parallel        1      0.0036      0.0036      0.0037    107.797      1.00x      1.00x         -
     32769     parallel        2      0.0163      0.0163      0.0165     24.173      0.22x      0.22x         -
     32769     parallel        4      0.0154      0.0152      0.0155     25.555      0.24x      0.24x         -
     32769     parallel        8      0.0229      0.0218      0.0243     17.201      0.16x      0.16x         -
     32769     parallel       12      0.0279      0.0276      0.0283     14.119      0.13x      0.13x         -
     32769     parallel       16      0.0302      0.0298      0.0322     13.019      0.12x      0.12x         -
     32769     parallel       24      0.0393      0.0386      0.0418     10.014      0.09x      0.09x         -

     40000       scalar        -      0.0046      0.0046      0.0046    104.472      1.00x      0.98x         -
     40000   sequential        -      0.0045      0.0045      0.0046    106.987      1.02x      1.00x         -
     40000     parallel        1      0.0047      0.0047      0.0049    102.760      0.98x      0.96x         -
     40000     parallel        2      0.0174      0.0174      0.0176     27.523      0.26x      0.26x         -
     40000     parallel        4      0.0166      0.0165      0.0169     28.857      0.28x      0.27x         -
     40000     parallel        8      0.0238      0.0228      0.0240     20.190      0.19x      0.19x         -
     40000     parallel       12      0.0285      0.0281      0.0288     16.856      0.16x      0.16x         -
     40000     parallel       16      0.0305      0.0302      0.0310     15.730      0.15x      0.15x         -
     40000     parallel       24      0.0377      0.0369      0.0381     12.734      0.12x      0.12x         -

     50000       scalar        -      0.0061      0.0060      0.0061     99.161      1.00x      0.99x         -
     50000   sequential        -      0.0060      0.0060      0.0061    100.098      1.01x      1.00x         -
     50000     parallel        1      0.0061      0.0061      0.0061     98.851      1.00x      0.99x         -
     50000     parallel        2      0.0195      0.0195      0.0201     30.698      0.31x      0.31x         -
     50000     parallel        4      0.0171      0.0171      0.0175     35.070      0.35x      0.35x         -
     50000     parallel        8      0.0234      0.0232      0.0245     25.620      0.26x      0.26x         -
     50000     parallel       12      0.0294      0.0293      0.0298     20.423      0.21x      0.20x         -
     50000     parallel       16      0.0311      0.0310      0.0317     19.302      0.19x      0.19x         -
     50000     parallel       24      0.0377      0.0371      0.0385     15.911      0.16x      0.16x         -

     75000       scalar        -      0.0095      0.0095      0.0096     94.717      1.00x      0.99x         -
     75000   sequential        -      0.0094      0.0094      0.0095     95.665      1.01x      1.00x         -
     75000     parallel        1      0.0095      0.0095      0.0095     94.846      1.00x      0.99x         -
     75000     parallel        2      0.0249      0.0246      0.0253     36.181      0.38x      0.38x         -
     75000     parallel        4      0.0195      0.0192      0.0215     46.206      0.49x      0.48x         -
     75000     parallel        8      0.0253      0.0252      0.0255     35.507      0.37x      0.37x         -
     75000     parallel       12      0.0315      0.0309      0.0318     28.588      0.30x      0.30x         -
     75000     parallel       16      0.0323      0.0317      0.0328     27.838      0.29x      0.29x         -
     75000     parallel       24      0.0389      0.0383      0.0401     23.162      0.24x      0.24x         -

    100000       scalar        -      0.0127      0.0127      0.0128     94.216      1.00x      0.99x         -
    100000   sequential        -      0.0126      0.0126      0.0126     95.282      1.01x      1.00x         -
    100000     parallel        1      0.0127      0.0127      0.0128     94.167      1.00x      0.99x         -
    100000     parallel        2      0.0292      0.0291      0.0294     41.142      0.44x      0.43x         -
    100000     parallel        4      0.0221      0.0216      0.0233     54.387      0.58x      0.57x         -
    100000     parallel        8      0.0274      0.0269      0.0325     43.864      0.47x      0.46x         -
    100000     parallel       12      0.0343      0.0337      0.0350     34.974      0.37x      0.37x         -
    100000     parallel       16      0.0343      0.0341      0.0355     34.938      0.37x      0.37x         -
    100000     parallel       24      0.0413      0.0408      0.0429     29.040      0.31x      0.30x         -

    200000       scalar        -      0.0253      0.0252      0.0256     95.017      1.00x      1.00x         -
    200000   sequential        -      0.0253      0.0253      0.0254     94.750      1.00x      1.00x         -
    200000     parallel        1      0.0253      0.0252      0.0254     94.997      1.00x      1.00x         -
    200000     parallel        2      0.0460      0.0458      0.0474     52.193      0.55x      0.55x         -
    200000     parallel        4      0.0306      0.0304      0.0309     78.373      0.82x      0.83x         -
    200000     parallel        8      0.0344      0.0343      0.0350     69.863      0.74x      0.74x         -
    200000     parallel       12      0.0400      0.0390      0.0428     59.963      0.63x      0.63x         -
    200000     parallel       16      0.0430      0.0427      0.0439     55.759      0.59x      0.59x         -
    200000     parallel       24      0.0491      0.0484      0.0495     48.902      0.51x      0.52x         -

    500000       scalar        -      0.0630      0.0629      0.0635     95.300      1.00x      1.01x         -
    500000   sequential        -      0.0633      0.0633      0.0636     94.777      0.99x      1.00x         -
    500000     parallel        1      0.0629      0.0628      0.0630     95.394      1.00x      1.01x         -
    500000     parallel        2      0.0977      0.0963      0.0989     61.428      0.64x      0.65x         -
    500000     parallel        4      0.0582      0.0577      0.0584    103.173      1.08x      1.09x         -
    500000     parallel        8      0.0581      0.0561      0.0627    103.215      1.08x      1.09x         -
    500000     parallel       12      0.0608      0.0602      0.0619     98.687      1.04x      1.04x         -
    500000     parallel       16      0.0691      0.0681      0.0709     86.842      0.91x      0.92x         -
    500000     parallel       24      0.0729      0.0710      0.0755     82.312      0.86x      0.87x         -

   1000000       scalar        -      0.1258      0.1258      0.1287     95.356      1.00x      1.02x         -
   1000000   sequential        -      0.1278      0.1273      0.1299     93.910      0.98x      1.00x         -
   1000000     parallel        1      0.1266      0.1265      0.1285     94.754      0.99x      1.01x         -
   1000000     parallel        2      0.1815      0.1808      0.2122     66.120      0.69x      0.70x         -
   1000000     parallel        4      0.0991      0.0984      0.1003    121.067      1.27x      1.29x         -
   1000000     parallel        8      0.0918      0.0912      0.0956    130.657      1.37x      1.39x         -
   1000000     parallel       12      0.1021      0.0994      0.1106    117.507      1.23x      1.25x         -
   1000000     parallel       16      0.1110      0.0914      0.1144    108.084      1.13x      1.15x         -
   1000000     parallel       24      0.1127      0.1093      0.1183    106.466      1.12x      1.13x         -

   2000000       scalar        -      0.2921      0.2594      0.3336     82.167      1.00x      0.87x         -
   2000000   sequential        -      0.2555      0.2531      0.2767     93.947      1.14x      1.00x         -
   2000000     parallel        1      0.2653      0.2580      0.3081     90.476      1.10x      0.96x         -
   2000000     parallel        2      0.3570      0.3492      0.3865     67.220      0.82x      0.72x         -
   2000000     parallel        4      0.1880      0.1838      0.2004    127.668      1.55x      1.36x         -
   2000000     parallel        8      0.1628      0.1569      0.1795    147.437      1.79x      1.57x         -
   2000000     parallel       12      0.2012      0.1824      0.2116    119.288      1.45x      1.27x         -
   2000000     parallel       16      0.2909      0.2493      0.4463     82.488      1.00x      0.88x         -
   2000000     parallel       24      0.3137      0.2699      0.4749     76.497      0.93x      0.81x         -

   5000000       scalar        -      1.5486      1.5121      1.6087     38.743      1.00x      0.97x         -
   5000000   sequential        -      1.5095      1.4647      2.2803     39.750      1.03x      1.00x         -
   5000000     parallel        1      1.4923      1.4680      1.5527     40.208      1.04x      1.01x         -
   5000000     parallel        2      1.4058      1.3610      1.5572     42.680      1.10x      1.07x         -
   5000000     parallel        4      1.1407      1.0996      1.2310     52.599      1.36x      1.32x         -
   5000000     parallel        8      0.8625      0.7718      1.4055     69.567      1.80x      1.75x         -
   5000000     parallel       12      1.1545      1.0329      1.2496     51.971      1.34x      1.31x         -
   5000000     parallel       16      1.1978      0.9466      1.3057     50.092      1.29x      1.26x         -
   5000000     parallel       24      1.2766      1.1223      1.3699     47.000      1.21x      1.18x         -

  10000000       scalar        -      3.7100      3.5408      4.0405     32.345      1.00x      0.97x         -
  10000000   sequential        -      3.5982      3.4754      3.8464     33.350      1.03x      1.00x         -
  10000000     parallel        1      3.5726      3.4959      3.7824     33.589      1.04x      1.01x         -
  10000000     parallel        2      3.2803      3.2433      3.3355     36.582      1.13x      1.10x         -
  10000000     parallel        4      3.3038      3.2398      4.4495     36.322      1.12x      1.09x         -
  10000000     parallel        8      2.5133      2.4802      2.7198     47.746      1.48x      1.43x         -
  10000000     parallel       12      3.6111      3.1292      3.9634     33.231      1.03x      1.00x         -
  10000000     parallel       16      3.3895      3.2986      3.5975     35.403      1.09x      1.06x         -
  10000000     parallel       24      3.3730      3.2565      3.4562     35.577      1.10x      1.07x         -

  33554432       scalar        -     15.2583     14.8115     16.2639     26.389      1.00x      0.98x      1.02
  33554432   sequential        -     14.9683     14.9008     15.1749     26.900      1.02x      1.00x      0.93
  33554432     parallel        1     15.1903     14.8470     21.8542     26.507      1.00x      0.99x      0.97
  33554432     parallel        2     13.8716     13.7425     14.5208     29.027      1.10x      1.08x      1.87
  33554432     parallel        4     14.9234     14.5390     15.9565     26.981      1.02x      1.00x      3.71
  33554432     parallel        8     13.5081     13.3607     14.2032     29.808      1.13x      1.11x      5.73
  33554432     parallel       12     13.9593     13.7941     15.1563     28.845      1.09x      1.07x     10.35
  33554432     parallel       16     14.4212     14.2172     15.0144     27.921      1.06x      1.04x     15.85
  33554432     parallel       24     14.3102     14.1892     14.8885     28.137      1.07x      1.05x     19.57


== mul ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    142.889      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    143.243      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    137.325      0.96x      0.96x         -
      1000     parallel        2      0.0001      0.0001      0.0001    137.296      0.96x      0.96x         -
      1000     parallel        4      0.0001      0.0001      0.0001    137.291      0.96x      0.96x         -
      1000     parallel        8      0.0001      0.0001      0.0001    137.085      0.96x      0.96x         -
      1000     parallel       12      0.0001      0.0001      0.0001    137.339      0.96x      0.96x         -
      1000     parallel       16      0.0001      0.0001      0.0001    137.100      0.96x      0.96x         -
      1000     parallel       24      0.0001      0.0001      0.0001    137.310      0.96x      0.96x         -

     10000       scalar        -      0.0011      0.0011      0.0011    110.152      1.00x      1.00x         -
     10000   sequential        -      0.0011      0.0011      0.0011    110.399      1.00x      1.00x         -
     10000     parallel        1      0.0011      0.0011      0.0011    109.538      0.99x      0.99x         -
     10000     parallel        2      0.0011      0.0011      0.0011    109.675      1.00x      0.99x         -
     10000     parallel        4      0.0011      0.0011      0.0011    109.690      1.00x      0.99x         -
     10000     parallel        8      0.0011      0.0011      0.0011    109.695      1.00x      0.99x         -
     10000     parallel       12      0.0011      0.0011      0.0011    109.700      1.00x      0.99x         -
     10000     parallel       16      0.0011      0.0011      0.0011    109.670      1.00x      0.99x         -
     10000     parallel       24      0.0011      0.0011      0.0011    109.680      1.00x      0.99x         -

     32768       scalar        -      0.0036      0.0036      0.0080    107.941      1.00x      1.00x         -
     32768   sequential        -      0.0037      0.0037      0.0037    107.426      1.00x      1.00x         -
     32768     parallel        1      0.0036      0.0036      0.0037    108.104      1.00x      1.01x         -
     32768     parallel        2      0.0036      0.0036      0.0037    108.031      1.00x      1.01x         -
     32768     parallel        4      0.0036      0.0036      0.0037    108.083      1.00x      1.01x         -
     32768     parallel        8      0.0037      0.0036      0.0037    107.664      1.00x      1.00x         -
     32768     parallel       12      0.0036      0.0036      0.0037    108.167      1.00x      1.01x         -
     32768     parallel       16      0.0036      0.0036      0.0037    108.170      1.00x      1.01x         -
     32768     parallel       24      0.0036      0.0036      0.0037    108.159      1.00x      1.01x         -

     32769       scalar        -      0.0036      0.0036      0.0037    108.720      1.00x      1.00x         -
     32769   sequential        -      0.0036      0.0036      0.0037    108.206      1.00x      1.00x         -
     32769     parallel        1      0.0037      0.0036      0.0037    107.656      0.99x      0.99x         -
     32769     parallel        2      0.0165      0.0164      0.0166     23.889      0.22x      0.22x         -
     32769     parallel        4      0.0154      0.0153      0.0160     25.576      0.24x      0.24x         -
     32769     parallel        8      0.0222      0.0221      0.0228     17.745      0.16x      0.16x         -
     32769     parallel       12      0.0279      0.0277      0.0280     14.104      0.13x      0.13x         -
     32769     parallel       16      0.0302      0.0298      0.0306     13.008      0.12x      0.12x         -
     32769     parallel       24      0.0369      0.0364      0.0379     10.652      0.10x      0.10x         -

     40000       scalar        -      0.0046      0.0046      0.0046    104.494      1.00x      0.97x         -
     40000   sequential        -      0.0045      0.0045      0.0045    107.667      1.03x      1.00x         -
     40000     parallel        1      0.0045      0.0045      0.0046    105.558      1.01x      0.98x         -
     40000     parallel        2      0.0176      0.0175      0.0177     27.297      0.26x      0.25x         -
     40000     parallel        4      0.0164      0.0160      0.0174     29.274      0.28x      0.27x         -
     40000     parallel        8      0.0254      0.0246      0.0259     18.919      0.18x      0.18x         -
     40000     parallel       12      0.0285      0.0281      0.0291     16.818      0.16x      0.16x         -
     40000     parallel       16      0.0306      0.0299      0.0319     15.683      0.15x      0.15x         -
     40000     parallel       24      0.0366      0.0361      0.0375     13.129      0.13x      0.12x         -

     50000       scalar        -      0.0060      0.0060      0.0064     99.199      1.00x      1.00x         -
     50000   sequential        -      0.0061      0.0060      0.0062     99.100      1.00x      1.00x         -
     50000     parallel        1      0.0061      0.0060      0.0062     98.553      0.99x      0.99x         -
     50000     parallel        2      0.0194      0.0194      0.0198     30.984      0.31x      0.31x         -
     50000     parallel        4      0.0169      0.0169      0.0173     35.501      0.36x      0.36x         -
     50000     parallel        8      0.0235      0.0233      0.0241     25.559      0.26x      0.26x         -
     50000     parallel       12      0.0296      0.0294      0.0297     20.290      0.20x      0.20x         -
     50000     parallel       16      0.0309      0.0306      0.0315     19.423      0.20x      0.20x         -
     50000     parallel       24      0.0378      0.0362      0.0401     15.880      0.16x      0.16x         -

     75000       scalar        -      0.0095      0.0095      0.0096     94.955      1.00x      0.99x         -
     75000   sequential        -      0.0094      0.0094      0.0095     95.661      1.01x      1.00x         -
     75000     parallel        1      0.0095      0.0094      0.0095     95.069      1.00x      0.99x         -
     75000     parallel        2      0.0240      0.0239      0.0242     37.556      0.40x      0.39x         -
     75000     parallel        4      0.0192      0.0191      0.0196     46.972      0.49x      0.49x         -
     75000     parallel        8      0.0254      0.0252      0.0257     35.487      0.37x      0.37x         -
     75000     parallel       12      0.0316      0.0310      0.0318     28.447      0.30x      0.30x         -
     75000     parallel       16      0.0323      0.0320      0.0332     27.831      0.29x      0.29x         -
     75000     parallel       24      0.0394      0.0389      0.0402     22.857      0.24x      0.24x         -

    100000       scalar        -      0.0127      0.0126      0.0130     94.291      1.00x      0.99x         -
    100000   sequential        -      0.0126      0.0125      0.0127     95.182      1.01x      1.00x         -
    100000     parallel        1      0.0127      0.0126      0.0128     94.552      1.00x      0.99x         -
    100000     parallel        2      0.0289      0.0287      0.0304     41.499      0.44x      0.44x         -
    100000     parallel        4      0.0215      0.0215      0.0220     55.779      0.59x      0.59x         -
    100000     parallel        8      0.0269      0.0267      0.0279     44.632      0.47x      0.47x         -
    100000     parallel       12      0.0341      0.0335      0.0346     35.160      0.37x      0.37x         -
    100000     parallel       16      0.0345      0.0340      0.0356     34.828      0.37x      0.37x         -
    100000     parallel       24      0.0428      0.0425      0.0443     28.024      0.30x      0.29x         -

    200000       scalar        -      0.0253      0.0252      0.0256     94.935      1.00x      1.00x         -
    200000   sequential        -      0.0254      0.0253      0.0257     94.610      1.00x      1.00x         -
    200000     parallel        1      0.0252      0.0252      0.0255     95.282      1.00x      1.01x         -
    200000     parallel        2      0.0456      0.0455      0.0463     52.617      0.55x      0.56x         -
    200000     parallel        4      0.0305      0.0302      0.0308     78.588      0.83x      0.83x         -
    200000     parallel        8      0.0344      0.0342      0.0359     69.793      0.74x      0.74x         -
    200000     parallel       12      0.0401      0.0388      0.0438     59.825      0.63x      0.63x         -
    200000     parallel       16      0.0433      0.0420      0.0438     55.389      0.58x      0.59x         -
    200000     parallel       24      0.0483      0.0479      0.0489     49.669      0.52x      0.52x         -

    500000       scalar        -      0.0629      0.0627      0.0642     95.418      1.00x      1.01x         -
    500000   sequential        -      0.0634      0.0634      0.0649     94.586      0.99x      1.00x         -
    500000     parallel        1      0.0629      0.0628      0.0649     95.371      1.00x      1.01x         -
    500000     parallel        2      0.0968      0.0962      0.0980     61.951      0.65x      0.65x         -
    500000     parallel        4      0.0563      0.0561      0.0571    106.584      1.12x      1.13x         -
    500000     parallel        8      0.0570      0.0559      0.0575    105.301      1.10x      1.11x         -
    500000     parallel       12      0.0598      0.0587      0.0619    100.264      1.05x      1.06x         -
    500000     parallel       16      0.0687      0.0570      0.0703     87.281      0.91x      0.92x         -
    500000     parallel       24      0.0724      0.0714      0.0744     82.866      0.87x      0.88x         -

   1000000       scalar        -      0.1797      0.1473      0.2445     66.771      1.00x      0.71x         -
   1000000   sequential        -      0.1275      0.1273      0.1322     94.131      1.41x      1.00x         -
   1000000     parallel        1      0.1262      0.1258      0.1290     95.073      1.42x      1.01x         -
   1000000     parallel        2      0.1813      0.1800      0.1874     66.177      0.99x      0.70x         -
   1000000     parallel        4      0.0986      0.0979      0.1001    121.688      1.82x      1.29x         -
   1000000     parallel        8      0.0948      0.0906      0.1053    126.607      1.90x      1.35x         -
   1000000     parallel       12      0.0847      0.0797      0.0935    141.682      2.12x      1.51x         -
   1000000     parallel       16      0.1006      0.0955      0.1071    119.325      1.79x      1.27x         -
   1000000     parallel       24      0.1151      0.1072      0.1190    104.232      1.56x      1.11x         -

   2000000       scalar        -      0.2535      0.2518      0.3004     94.675      1.00x      1.00x         -
   2000000   sequential        -      0.2545      0.2534      0.3004     94.303      1.00x      1.00x         -
   2000000     parallel        1      0.2551      0.2525      0.2865     94.090      0.99x      1.00x         -
   2000000     parallel        2      0.3658      0.3517      0.3871     65.603      0.69x      0.70x         -
   2000000     parallel        4      0.1833      0.1827      0.2171    130.906      1.38x      1.39x         -
   2000000     parallel        8      0.1674      0.1608      0.1867    143.396      1.51x      1.52x         -
   2000000     parallel       12      0.1900      0.1657      0.2095    126.328      1.33x      1.34x         -
   2000000     parallel       16      0.3413      0.3019      0.4328     70.314      0.74x      0.75x         -
   2000000     parallel       24      0.3183      0.2777      0.4982     75.398      0.80x      0.80x         -

   5000000       scalar        -      1.4628      1.4410      1.5906     41.016      1.00x      1.03x         -
   5000000   sequential        -      1.5030      1.4521      1.6946     39.919      0.97x      1.00x         -
   5000000     parallel        1      1.5444      1.5153      1.6293     38.851      0.95x      0.97x         -
   5000000     parallel        2      1.4362      1.4070      1.5594     41.777      1.02x      1.05x         -
   5000000     parallel        4      1.1489      1.0931      1.3266     52.226      1.27x      1.31x         -
   5000000     parallel        8      0.8563      0.8027      1.3363     70.071      1.71x      1.76x         -
   5000000     parallel       12      1.1654      0.9584      1.2612     51.483      1.26x      1.29x         -
   5000000     parallel       16      1.1752      1.0715      1.2938     51.056      1.24x      1.28x         -
   5000000     parallel       24      1.2295      1.1523      1.2729     48.800      1.19x      1.22x         -

  10000000       scalar        -      3.9803      3.6853      4.4411     30.148      1.00x      0.95x         -
  10000000   sequential        -      3.7724      3.5422      6.0183     31.810      1.06x      1.00x         -
  10000000     parallel        1      3.5986      3.5088      3.8340     33.346      1.11x      1.05x         -
  10000000     parallel        2      3.3434      3.2727      3.5072     35.892      1.19x      1.13x         -
  10000000     parallel        4      3.3509      3.2362      3.9082     35.811      1.19x      1.13x         -
  10000000     parallel        8      2.5462      2.4968      2.6103     47.129      1.56x      1.48x         -
  10000000     parallel       12      3.2026      3.0299      3.4309     37.470      1.24x      1.18x         -
  10000000     parallel       16      3.7139      3.5754      4.1105     32.311      1.07x      1.02x         -
  10000000     parallel       24      3.4425      3.3061      3.5003     34.858      1.16x      1.10x         -

  33554432       scalar        -     15.0890     14.9485     16.3904     26.685      1.00x      0.99x      1.02
  33554432   sequential        -     14.8990     14.8587     14.9651     27.026      1.01x      1.00x      0.93
  33554432     parallel        1     15.1818     14.9701     16.3503     26.522      0.99x      0.98x      0.91
  33554432     parallel        2     13.9898     13.7798     14.2563     28.782      1.08x      1.06x      2.11
  33554432     parallel        4     14.7203     14.3500     15.2730     27.354      1.03x      1.01x      4.11
  33554432     parallel        8     13.3891     13.3176     13.5477     30.073      1.13x      1.11x      7.37
  33554432     parallel       12     14.0569     13.4349     14.3939     28.645      1.07x      1.06x     11.81
  33554432     parallel       16     14.0549     13.8954     14.1911     28.649      1.07x      1.06x     13.84
  33554432     parallel       24     14.0102     13.8380     14.3600     28.740      1.08x      1.06x     23.41


== div ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0003      0.0003      0.0003     46.798      1.00x      1.00x         -
      1000   sequential        -      0.0003      0.0003      0.0003     46.814      1.00x      1.00x         -
      1000     parallel        1      0.0003      0.0003      0.0003     47.004      1.00x      1.00x         -
      1000     parallel        2      0.0003      0.0002      0.0003     47.223      1.01x      1.01x         -
      1000     parallel        4      0.0003      0.0002      0.0003     47.040      1.01x      1.00x         -
      1000     parallel        8      0.0003      0.0003      0.0003     47.071      1.01x      1.01x         -
      1000     parallel       12      0.0003      0.0003      0.0003     47.144      1.01x      1.01x         -
      1000     parallel       16      0.0003      0.0003      0.0003     47.198      1.01x      1.01x         -
      1000     parallel       24      0.0003      0.0002      0.0003     47.155      1.01x      1.01x         -

     10000       scalar        -      0.0025      0.0025      0.0025     47.953      1.00x      1.00x         -
     10000   sequential        -      0.0025      0.0025      0.0025     47.929      1.00x      1.00x         -
     10000     parallel        1      0.0023      0.0022      0.0023     53.002      1.11x      1.11x         -
     10000     parallel        2      0.0023      0.0022      0.0023     52.936      1.10x      1.10x         -
     10000     parallel        4      0.0023      0.0022      0.0023     52.877      1.10x      1.10x         -
     10000     parallel        8      0.0023      0.0022      0.0023     53.110      1.11x      1.11x         -
     10000     parallel       12      0.0022      0.0022      0.0023     53.438      1.11x      1.11x         -
     10000     parallel       16      0.0023      0.0022      0.0023     52.852      1.10x      1.10x         -
     10000     parallel       24      0.0023      0.0022      0.0023     53.262      1.11x      1.11x         -

     32768       scalar        -      0.0082      0.0082      0.0083     48.015      1.00x      1.00x         -
     32768   sequential        -      0.0082      0.0082      0.0083     47.937      1.00x      1.00x         -
     32768     parallel        1      0.0078      0.0077      0.0078     50.734      1.06x      1.06x         -
     32768     parallel        2      0.0077      0.0076      0.0078     50.798      1.06x      1.06x         -
     32768     parallel        4      0.0077      0.0076      0.0078     50.794      1.06x      1.06x         -
     32768     parallel        8      0.0077      0.0077      0.0079     50.882      1.06x      1.06x         -
     32768     parallel       12      0.0078      0.0076      0.0078     50.624      1.05x      1.06x         -
     32768     parallel       16      0.0077      0.0077      0.0078     50.998      1.06x      1.06x         -
     32768     parallel       24      0.0077      0.0076      0.0078     50.917      1.06x      1.06x         -

     32769       scalar        -      0.0082      0.0082      0.0083     47.923      1.00x      0.97x         -
     32769   sequential        -      0.0080      0.0079      0.0080     49.349      1.03x      1.00x         -
     32769     parallel        1      0.0080      0.0079      0.0081     49.421      1.03x      1.00x         -
     32769     parallel        2      0.0269      0.0268      0.0273     14.634      0.31x      0.30x         -
     32769     parallel        4      0.0206      0.0204      0.0214     19.077      0.40x      0.39x         -
     32769     parallel        8      0.0241      0.0239      0.0245     16.305      0.34x      0.33x         -
     32769     parallel       12      0.0288      0.0286      0.0290     13.674      0.29x      0.28x         -
     32769     parallel       16      0.0307      0.0304      0.0323     12.819      0.27x      0.26x         -
     32769     parallel       24      0.0394      0.0391      0.0399      9.990      0.21x      0.20x         -

     40000       scalar        -      0.0095      0.0094      0.0097     50.303      1.00x      0.98x         -
     40000   sequential        -      0.0094      0.0093      0.0095     51.187      1.02x      1.00x         -
     40000     parallel        1      0.0100      0.0099      0.0100     48.213      0.96x      0.94x         -
     40000     parallel        2      0.0305      0.0295      0.0308     15.750      0.31x      0.31x         -
     40000     parallel        4      0.0222      0.0222      0.0227     21.597      0.43x      0.42x         -
     40000     parallel        8      0.0253      0.0250      0.0254     18.989      0.38x      0.37x         -
     40000     parallel       12      0.0294      0.0294      0.0297     16.316      0.32x      0.32x         -
     40000     parallel       16      0.0308      0.0305      0.0313     15.563      0.31x      0.30x         -
     40000     parallel       24      0.0403      0.0399      0.0411     11.905      0.24x      0.23x         -

     50000       scalar        -      0.0115      0.0114      0.0116     52.105      1.00x      1.01x         -
     50000   sequential        -      0.0116      0.0116      0.0117     51.516      0.99x      1.00x         -
     50000     parallel        1      0.0120      0.0119      0.0122     50.033      0.96x      0.97x         -
     50000     parallel        2      0.0358      0.0354      0.0369     16.780      0.32x      0.33x         -
     50000     parallel        4      0.0259      0.0246      0.0266     23.144      0.44x      0.45x         -
     50000     parallel        8      0.0276      0.0272      0.0290     21.756      0.42x      0.42x         -
     50000     parallel       12      0.0305      0.0303      0.0307     19.702      0.38x      0.38x         -
     50000     parallel       16      0.0316      0.0314      0.0323     19.000      0.36x      0.37x         -
     50000     parallel       24      0.0418      0.0413      0.0437     14.347      0.28x      0.28x         -

     75000       scalar        -      0.0175      0.0173      0.0176     51.532      1.00x      0.99x         -
     75000   sequential        -      0.0173      0.0172      0.0177     51.943      1.01x      1.00x         -
     75000     parallel        1      0.0178      0.0175      0.0179     50.698      0.98x      0.98x         -
     75000     parallel        2      0.0491      0.0489      0.0494     18.342      0.36x      0.35x         -
     75000     parallel        4      0.0312      0.0311      0.0319     28.850      0.56x      0.56x         -
     75000     parallel        8      0.0296      0.0293      0.0298     30.365      0.59x      0.58x         -
     75000     parallel       12      0.0328      0.0324      0.0336     27.478      0.53x      0.53x         -
     75000     parallel       16      0.0349      0.0344      0.0369     25.816      0.50x      0.50x         -
     75000     parallel       24      0.0449      0.0442      0.0453     20.041      0.39x      0.39x         -

    100000       scalar        -      0.0232      0.0230      0.0236     51.766      1.00x      1.00x         -
    100000   sequential        -      0.0233      0.0229      0.0235     51.532      1.00x      1.00x         -
    100000     parallel        1      0.0235      0.0231      0.0237     51.062      0.99x      0.99x         -
    100000     parallel        2      0.0612      0.0610      0.0616     19.599      0.38x      0.38x         -
    100000     parallel        4      0.0377      0.0375      0.0385     31.863      0.62x      0.62x         -
    100000     parallel        8      0.0323      0.0322      0.0331     37.100      0.72x      0.72x         -
    100000     parallel       12      0.0359      0.0355      0.0362     33.413      0.65x      0.65x         -
    100000     parallel       16      0.0374      0.0372      0.0385     32.057      0.62x      0.62x         -
    100000     parallel       24      0.0457      0.0453      0.0463     26.256      0.51x      0.51x         -

    200000       scalar        -      0.0461      0.0454      0.0466     52.066      1.00x      1.00x         -
    200000   sequential        -      0.0462      0.0458      0.0467     51.999      1.00x      1.00x         -
    200000     parallel        1      0.0466      0.0464      0.0468     51.523      0.99x      0.99x         -
    200000     parallel        2      0.1102      0.1086      0.1114     21.787      0.42x      0.42x         -
    200000     parallel        4      0.0638      0.0637      0.0641     37.614      0.72x      0.72x         -
    200000     parallel        8      0.0459      0.0457      0.0466     52.263      1.00x      1.01x         -
    200000     parallel       12      0.0462      0.0455      0.0533     51.996      1.00x      1.00x         -
    200000     parallel       16      0.0517      0.0508      0.0533     46.451      0.89x      0.89x         -
    200000     parallel       24      0.0492      0.0483      0.0553     48.745      0.94x      0.94x         -

    500000       scalar        -      0.1151      0.1141      0.1158     52.143      1.00x      1.00x         -
    500000   sequential        -      0.1149      0.1137      0.1301     52.229      1.00x      1.00x         -
    500000     parallel        1      0.1160      0.1155      0.1175     51.723      0.99x      0.99x         -
    500000     parallel        2      0.2478      0.2454      0.2583     24.209      0.46x      0.46x         -
    500000     parallel        4      0.1377      0.1372      0.1397     43.577      0.84x      0.83x         -
    500000     parallel        8      0.0834      0.0826      0.0869     71.926      1.38x      1.38x         -
    500000     parallel       12      0.0751      0.0728      0.0802     79.860      1.53x      1.53x         -
    500000     parallel       16      0.0884      0.0880      0.0886     67.861      1.30x      1.30x         -
    500000     parallel       24      0.0799      0.0762      0.0899     75.106      1.44x      1.44x         -

   1000000       scalar        -      0.2304      0.2273      0.2340     52.089      1.00x      0.99x         -
   1000000   sequential        -      0.2291      0.2280      0.2304     52.372      1.01x      1.00x         -
   1000000     parallel        1      0.2327      0.2309      0.2344     51.578      0.99x      0.98x         -
   1000000     parallel        2      0.4945      0.4920      0.5229     24.265      0.47x      0.46x         -
   1000000     parallel        4      0.2577      0.2567      0.2619     46.566      0.89x      0.89x         -
   1000000     parallel        8      0.1437      0.1429      0.1504     83.529      1.60x      1.59x         -
   1000000     parallel       12      0.1193      0.1172      0.1238    100.618      1.93x      1.92x         -
   1000000     parallel       16      0.1477      0.1464      0.1484     81.232      1.56x      1.55x         -
   1000000     parallel       24      0.1184      0.1168      0.1229    101.309      1.94x      1.93x         -

   2000000       scalar        -      0.4622      0.4549      0.4647     51.930      1.00x      1.08x         -
   2000000   sequential        -      0.4969      0.4739      0.5242     48.297      0.93x      1.00x         -
   2000000     parallel        1      0.5076      0.4795      0.5170     47.285      0.91x      0.98x         -
   2000000     parallel        2      0.9851      0.9683      0.9996     24.362      0.47x      0.50x         -
   2000000     parallel        4      0.5007      0.4957      0.5070     47.935      0.92x      0.99x         -
   2000000     parallel        8      0.2660      0.2647      0.2715     90.230      1.74x      1.87x         -
   2000000     parallel       12      0.2142      0.2083      0.2402    112.042      2.16x      2.32x         -
   2000000     parallel       16      0.2647      0.2620      0.2742     90.682      1.75x      1.88x         -
   2000000     parallel       24      0.2063      0.2028      0.2188    116.335      2.24x      2.41x         -

   5000000       scalar        -      1.5410      1.5117      1.6144     38.936      1.00x      1.01x         -
   5000000   sequential        -      1.5603      1.5368      1.6008     38.454      0.99x      1.00x         -
   5000000     parallel        1      1.6044      1.5499      1.7672     37.397      0.96x      0.97x         -
   5000000     parallel        2      2.6503      2.5448      2.9456     22.639      0.58x      0.59x         -
   5000000     parallel        4      1.3349      1.3263      1.3888     44.947      1.15x      1.17x         -
   5000000     parallel        8      0.8439      0.6949      0.9765     71.103      1.83x      1.85x         -
   5000000     parallel       12      1.1519      0.9954      1.2709     52.090      1.34x      1.35x         -
   5000000     parallel       16      1.4581      1.2926      1.5270     41.149      1.06x      1.07x         -
   5000000     parallel       24      1.2280      1.1094      1.2671     48.860      1.25x      1.27x         -

  10000000       scalar        -      3.6164      3.5232      3.7658     33.182      1.00x      1.01x         -
  10000000   sequential        -      3.6553      3.5653      3.8945     32.829      0.99x      1.00x         -
  10000000     parallel        1      3.6268      3.5047      4.1311     33.087      1.00x      1.01x         -
  10000000     parallel        2      5.1613      5.1116      5.2683     23.250      0.70x      0.71x         -
  10000000     parallel        4      3.5289      3.3696      3.8865     34.005      1.02x      1.04x         -
  10000000     parallel        8      2.8790      2.5081      3.0603     41.681      1.26x      1.27x         -
  10000000     parallel       12      3.2984      3.0744      3.5567     36.381      1.10x      1.11x         -
  10000000     parallel       16      3.4514      3.2883      3.7158     34.768      1.05x      1.06x         -
  10000000     parallel       24      3.4304      3.2475      3.5819     34.981      1.05x      1.07x         -

  33554432       scalar        -     15.0750     14.9488     16.2967     26.710      1.00x      0.99x      1.02
  33554432   sequential        -     14.8620     14.7221     15.4663     27.093      1.01x      1.00x      0.93
  33554432     parallel        1     15.1076     14.9925     15.6933     26.652      1.00x      0.98x      1.03
  33554432     parallel        2     18.2332     18.1024     18.6151     22.084      0.83x      0.82x      1.99
  33554432     parallel        4     14.5651     14.4720     15.6571     27.645      1.04x      1.02x      3.90
  33554432     parallel        8     13.7017     13.3551     14.4448     29.387      1.10x      1.08x      8.17
  33554432     parallel       12     13.5776     13.4004     13.9610     29.656      1.11x      1.09x     13.10
  33554432     parallel       16     14.0806     13.9567     14.3658     28.596      1.07x      1.06x     13.13
  33554432     parallel       24     14.0329     13.8162     14.3491     28.694      1.07x      1.06x     23.80


== neg ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    110.815      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    110.881      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    110.810      1.00x      1.00x         -
      1000     parallel        2      0.0001      0.0001      0.0001    110.507      1.00x      1.00x         -
      1000     parallel        4      0.0001      0.0001      0.0001    110.595      1.00x      1.00x         -
      1000     parallel        8      0.0001      0.0001      0.0001    110.745      1.00x      1.00x         -
      1000     parallel       12      0.0001      0.0001      0.0001    110.801      1.00x      1.00x         -
      1000     parallel       16      0.0001      0.0001      0.0001    110.754      1.00x      1.00x         -
      1000     parallel       24      0.0001      0.0001      0.0001    110.796      1.00x      1.00x         -

     10000       scalar        -      0.0007      0.0007      0.0007    109.431      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    109.362      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0007    108.467      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0007    108.485      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0007    108.467      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0007    108.457      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0007    108.453      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0007    108.464      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0007    108.317      0.99x      0.99x         -

     32768       scalar        -      0.0024      0.0024      0.0024    110.445      1.00x      1.00x         -
     32768   sequential        -      0.0024      0.0024      0.0024    110.572      1.00x      1.00x         -
     32768     parallel        1      0.0024      0.0024      0.0024    110.227      1.00x      1.00x         -
     32768     parallel        2      0.0024      0.0024      0.0024    110.209      1.00x      1.00x         -
     32768     parallel        4      0.0024      0.0024      0.0024    110.218      1.00x      1.00x         -
     32768     parallel        8      0.0024      0.0024      0.0024    110.213      1.00x      1.00x         -
     32768     parallel       12      0.0024      0.0024      0.0024    110.209      1.00x      1.00x         -
     32768     parallel       16      0.0024      0.0024      0.0024    110.195      1.00x      1.00x         -
     32768     parallel       24      0.0024      0.0024      0.0024    110.213      1.00x      1.00x         -

     32769       scalar        -      0.0024      0.0024      0.0024    110.466      1.00x      1.00x         -
     32769   sequential        -      0.0024      0.0024      0.0024    110.452      1.00x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0024    110.149      1.00x      1.00x         -
     32769     parallel        2      0.0155      0.0155      0.0165     16.937      0.15x      0.15x         -
     32769     parallel        4      0.0149      0.0149      0.0150     17.561      0.16x      0.16x         -
     32769     parallel        8      0.0236      0.0219      0.0242     11.108      0.10x      0.10x         -
     32769     parallel       12      0.0280      0.0272      0.0285      9.354      0.08x      0.08x         -
     32769     parallel       16      0.0300      0.0300      0.0307      8.725      0.08x      0.08x         -
     32769     parallel       24      0.0380      0.0364      0.0404      6.898      0.06x      0.06x         -

     40000       scalar        -      0.0029      0.0029      0.0029    110.419      1.00x      1.00x         -
     40000   sequential        -      0.0029      0.0029      0.0029    110.535      1.00x      1.00x         -
     40000     parallel        1      0.0029      0.0029      0.0029    110.159      1.00x      1.00x         -
     40000     parallel        2      0.0164      0.0164      0.0165     19.527      0.18x      0.18x         -
     40000     parallel        4      0.0153      0.0152      0.0154     20.867      0.19x      0.19x         -
     40000     parallel        8      0.0229      0.0227      0.0232     13.995      0.13x      0.13x         -
     40000     parallel       12      0.0283      0.0281      0.0285     11.300      0.10x      0.10x         -
     40000     parallel       16      0.0307      0.0303      0.0312     10.437      0.09x      0.09x         -
     40000     parallel       24      0.0370      0.0365      0.0382      8.654      0.08x      0.08x         -

     50000       scalar        -      0.0037      0.0037      0.0037    109.431      1.00x      1.01x         -
     50000   sequential        -      0.0037      0.0037      0.0037    108.194      0.99x      1.00x         -
     50000     parallel        1      0.0037      0.0037      0.0037    108.552      0.99x      1.00x         -
     50000     parallel        2      0.0184      0.0183      0.0186     21.688      0.20x      0.20x         -
     50000     parallel        4      0.0161      0.0161      0.0164     24.810      0.23x      0.23x         -
     50000     parallel        8      0.0232      0.0231      0.0236     17.275      0.16x      0.16x         -
     50000     parallel       12      0.0295      0.0293      0.0296     13.552      0.12x      0.13x         -
     50000     parallel       16      0.0316      0.0312      0.0319     12.654      0.12x      0.12x         -
     50000     parallel       24      0.0374      0.0367      0.0382     10.701      0.10x      0.10x         -

     75000       scalar        -      0.0061      0.0061      0.0061     98.677      1.00x      0.99x         -
     75000   sequential        -      0.0060      0.0060      0.0061     99.353      1.01x      1.00x         -
     75000     parallel        1      0.0062      0.0062      0.0062     96.945      0.98x      0.98x         -
     75000     parallel        2      0.0219      0.0218      0.0223     27.394      0.28x      0.28x         -
     75000     parallel        4      0.0179      0.0179      0.0183     33.447      0.34x      0.34x         -
     75000     parallel        8      0.0249      0.0249      0.0254     24.058      0.24x      0.24x         -
     75000     parallel       12      0.0316      0.0310      0.0322     18.967      0.19x      0.19x         -
     75000     parallel       16      0.0336      0.0328      0.0339     17.884      0.18x      0.18x         -
     75000     parallel       24      0.0413      0.0404      0.0426     14.541      0.15x      0.15x         -

    100000       scalar        -      0.0083      0.0083      0.0084     96.572      1.00x      1.01x         -
    100000   sequential        -      0.0084      0.0083      0.0087     95.723      0.99x      1.00x         -
    100000     parallel        1      0.0084      0.0084      0.0085     94.973      0.98x      0.99x         -
    100000     parallel        2      0.0259      0.0258      0.0261     30.938      0.32x      0.32x         -
    100000     parallel        4      0.0198      0.0197      0.0246     40.353      0.42x      0.42x         -
    100000     parallel        8      0.0263      0.0260      0.0273     30.421      0.32x      0.32x         -
    100000     parallel       12      0.0338      0.0329      0.0345     23.653      0.24x      0.25x         -
    100000     parallel       16      0.0346      0.0337      0.0353     23.133      0.24x      0.24x         -
    100000     parallel       24      0.0397      0.0391      0.0403     20.174      0.21x      0.21x         -

    200000       scalar        -      0.0168      0.0168      0.0171     95.238      1.00x      1.00x         -
    200000   sequential        -      0.0168      0.0168      0.0168     95.322      1.00x      1.00x         -
    200000     parallel        1      0.0168      0.0168      0.0168     95.066      1.00x      1.00x         -
    200000     parallel        2      0.0410      0.0409      0.0413     39.050      0.41x      0.41x         -
    200000     parallel        4      0.0278      0.0277      0.0281     57.645      0.61x      0.60x         -
    200000     parallel        8      0.0339      0.0337      0.0346     47.191      0.50x      0.50x         -
    200000     parallel       12      0.0389      0.0384      0.0409     41.110      0.43x      0.43x         -
    200000     parallel       16      0.0427      0.0422      0.0435     37.434      0.39x      0.39x         -
    200000     parallel       24      0.0474      0.0468      0.0479     33.730      0.35x      0.35x         -

    500000       scalar        -      0.0416      0.0415      0.0417     96.114      1.00x      1.01x         -
    500000   sequential        -      0.0421      0.0421      0.0427     94.966      0.99x      1.00x         -
    500000     parallel        1      0.0417      0.0417      0.0422     95.808      1.00x      1.01x         -
    500000     parallel        2      0.0841      0.0840      0.0865     47.584      0.50x      0.50x         -
    500000     parallel        4      0.0504      0.0501      0.0508     79.414      0.83x      0.84x         -
    500000     parallel        8      0.0551      0.0548      0.0554     72.599      0.76x      0.76x         -
    500000     parallel       12      0.0608      0.0595      0.0624     65.768      0.68x      0.69x         -
    500000     parallel       16      0.0685      0.0650      0.0721     58.402      0.61x      0.61x         -
    500000     parallel       24      0.0726      0.0714      0.0741     55.094      0.57x      0.58x         -

   1000000       scalar        -      0.0834      0.0832      0.0874     95.938      1.00x      1.02x         -
   1000000   sequential        -      0.0851      0.0848      0.0889     93.952      0.98x      1.00x         -
   1000000     parallel        1      0.0834      0.0833      0.0857     95.891      1.00x      1.02x         -
   1000000     parallel        2      0.1567      0.1541      0.1687     51.039      0.53x      0.54x         -
   1000000     parallel        4      0.0873      0.0859      0.0970     91.612      0.95x      0.98x         -
   1000000     parallel        8      0.0900      0.0897      0.0930     88.846      0.93x      0.95x         -
   1000000     parallel       12      0.0964      0.0954      0.1043     82.950      0.86x      0.88x         -
   1000000     parallel       16      0.1122      0.0800      0.1143     71.305      0.74x      0.76x         -
   1000000     parallel       24      0.1127      0.1092      0.1192     70.995      0.74x      0.76x         -

   2000000       scalar        -      0.1677      0.1673      0.1917     95.437      1.00x      1.00x         -
   2000000   sequential        -      0.1680      0.1672      0.1915     95.245      1.00x      1.00x         -
   2000000     parallel        1      0.1712      0.1695      0.1750     93.451      0.98x      0.98x         -
   2000000     parallel        2      0.3012      0.3003      0.3046     53.127      0.56x      0.56x         -
   2000000     parallel        4      0.1598      0.1577      0.1882    100.106      1.05x      1.05x         -
   2000000     parallel        8      0.1613      0.1587      0.1731     99.206      1.04x      1.04x         -
   2000000     parallel       12      0.1890      0.1772      0.2225     84.673      0.89x      0.89x         -
   2000000     parallel       16      0.2282      0.1838      0.3943     70.116      0.73x      0.74x         -
   2000000     parallel       24      0.2392      0.2079      0.3695     66.881      0.70x      0.70x         -

   5000000       scalar        -      0.9934      0.9768      1.0964     40.268      1.00x      1.01x         -
   5000000   sequential        -      0.9988      0.9499      1.0882     40.046      0.99x      1.00x         -
   5000000     parallel        1      0.9899      0.9623      1.1027     40.410      1.00x      1.01x         -
   5000000     parallel        2      0.9109      0.8894      1.0069     43.914      1.09x      1.10x         -
   5000000     parallel        4      0.6282      0.5886      0.9215     63.674      1.58x      1.59x         -
   5000000     parallel        8      0.6058      0.5320      0.9892     66.031      1.64x      1.65x         -
   5000000     parallel       12      0.4405      0.3964      0.6253     90.808      2.26x      2.27x         -
   5000000     parallel       16      0.5759      0.4968      0.7407     69.453      1.72x      1.73x         -
   5000000     parallel       24      0.5209      0.4209      0.6969     76.798      1.91x      1.92x         -

  10000000       scalar        -      2.5350      2.4275      2.8266     31.558      1.00x      0.99x         -
  10000000   sequential        -      2.5070      2.4423      2.8053     31.911      1.01x      1.00x         -
  10000000     parallel        1      2.4685      2.3901      2.7542     32.408      1.03x      1.02x         -
  10000000     parallel        2      2.1484      2.1094      2.2303     37.237      1.18x      1.17x         -
  10000000     parallel        4      2.0564      1.9866      2.8808     38.903      1.23x      1.22x         -
  10000000     parallel        8      1.3956      1.3337      1.9105     57.325      1.82x      1.80x         -
  10000000     parallel       12      2.1523      1.8031      2.2900     37.170      1.18x      1.16x         -
  10000000     parallel       16      1.9879      1.9320      2.2839     40.243      1.28x      1.26x         -
  10000000     parallel       24      2.1409      2.0530      2.3752     37.367      1.18x      1.17x         -

  33554432       scalar        -     11.3178     11.0275     12.3536     23.718      1.00x      0.98x      1.06
  33554432   sequential        -     11.1424     10.5796     12.8660     24.091      1.02x      1.00x      1.08
  33554432     parallel        1     11.4036     11.0247     12.0719     23.540      0.99x      0.98x      0.91
  33554432     parallel        2     11.6936     10.3897     13.5988     22.956      0.97x      0.95x      2.08
  33554432     parallel        4     10.6859     10.2577     17.8392     25.121      1.06x      1.04x      3.02
  33554432     parallel        8     10.0975      9.4114     10.4080     26.584      1.12x      1.10x         -
  33554432     parallel       12     10.0508      9.8201     10.7320     26.708      1.13x      1.11x         -
  33554432     parallel       16     10.1535     10.0486     10.2164     26.438      1.11x      1.10x         -
  33554432     parallel       24     10.3493     10.1124     10.9440     25.938      1.09x      1.08x         -


== abs ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    110.904      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    110.904      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    110.876      1.00x      1.00x         -
      1000     parallel        2      0.0001      0.0001      0.0001    110.885      1.00x      1.00x         -
      1000     parallel        4      0.0001      0.0001      0.0001    110.890      1.00x      1.00x         -
      1000     parallel        8      0.0001      0.0001      0.0001    110.829      1.00x      1.00x         -
      1000     parallel       12      0.0001      0.0001      0.0001    110.904      1.00x      1.00x         -
      1000     parallel       16      0.0001      0.0001      0.0001    110.876      1.00x      1.00x         -
      1000     parallel       24      0.0001      0.0001      0.0001    110.899      1.00x      1.00x         -

     10000       scalar        -      0.0007      0.0007      0.0007    109.464      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    109.391      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0008    108.604      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0007    108.554      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0007    108.543      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0007    108.561      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0007    108.586      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0007    108.561      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0007    108.593      0.99x      0.99x         -

     32768       scalar        -      0.0024      0.0024      0.0024    110.508      1.00x      1.00x         -
     32768   sequential        -      0.0024      0.0024      0.0024    110.549      1.00x      1.00x         -
     32768     parallel        1      0.0024      0.0024      0.0024    110.186      1.00x      1.00x         -
     32768     parallel        2      0.0024      0.0024      0.0024    110.245      1.00x      1.00x         -
     32768     parallel        4      0.0024      0.0024      0.0024    110.240      1.00x      1.00x         -
     32768     parallel        8      0.0024      0.0024      0.0024    110.245      1.00x      1.00x         -
     32768     parallel       12      0.0024      0.0024      0.0024    110.173      1.00x      1.00x         -
     32768     parallel       16      0.0024      0.0024      0.0024    110.231      1.00x      1.00x         -
     32768     parallel       24      0.0024      0.0024      0.0024    110.245      1.00x      1.00x         -

     32769       scalar        -      0.0024      0.0024      0.0024    110.448      1.00x      1.00x         -
     32769   sequential        -      0.0024      0.0024      0.0024    110.275      1.00x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0024    110.217      1.00x      1.00x         -
     32769     parallel        2      0.0153      0.0152      0.0156     17.171      0.16x      0.16x         -
     32769     parallel        4      0.0147      0.0146      0.0284     17.871      0.16x      0.16x         -
     32769     parallel        8      0.0220      0.0218      0.0224     11.891      0.11x      0.11x         -
     32769     parallel       12      0.0279      0.0271      0.0282      9.384      0.08x      0.09x         -
     32769     parallel       16      0.0303      0.0299      0.0318      8.659      0.08x      0.08x         -
     32769     parallel       24      0.0363      0.0358      0.0372      7.231      0.07x      0.07x         -

     40000       scalar        -      0.0029      0.0029      0.0029    110.497      1.00x      1.00x         -
     40000   sequential        -      0.0029      0.0029      0.0029    110.456      1.00x      1.00x         -
     40000     parallel        1      0.0029      0.0029      0.0029    110.222      1.00x      1.00x         -
     40000     parallel        2      0.0164      0.0164      0.0166     19.514      0.18x      0.18x         -
     40000     parallel        4      0.0153      0.0153      0.0159     20.876      0.19x      0.19x         -
     40000     parallel        8      0.0229      0.0228      0.0233     13.979      0.13x      0.13x         -
     40000     parallel       12      0.0284      0.0280      0.0286     11.276      0.10x      0.10x         -
     40000     parallel       16      0.0309      0.0306      0.0311     10.367      0.09x      0.09x         -
     40000     parallel       24      0.0369      0.0364      0.0379      8.674      0.08x      0.08x         -

     50000       scalar        -      0.0037      0.0037      0.0037    108.020      1.00x      0.98x         -
     50000   sequential        -      0.0036      0.0036      0.0036    110.640      1.02x      1.00x         -
     50000     parallel        1      0.0037      0.0037      0.0037    107.764      1.00x      0.97x         -
     50000     parallel        2      0.0178      0.0177      0.0183     22.493      0.21x      0.20x         -
     50000     parallel        4      0.0172      0.0171      0.0173     23.301      0.22x      0.21x         -
     50000     parallel        8      0.0243      0.0232      0.0244     16.470      0.15x      0.15x         -
     50000     parallel       12      0.0295      0.0293      0.0297     13.558      0.13x      0.12x         -
     50000     parallel       16      0.0312      0.0311      0.0325     12.810      0.12x      0.12x         -
     50000     parallel       24      0.0376      0.0374      0.0396     10.627      0.10x      0.10x         -

     75000       scalar        -      0.0062      0.0061      0.0062     97.524      1.00x      1.00x         -
     75000   sequential        -      0.0061      0.0061      0.0062     97.716      1.00x      1.00x         -
     75000     parallel        1      0.0062      0.0062      0.0063     96.504      0.99x      0.99x         -
     75000     parallel        2      0.0221      0.0219      0.0228     27.162      0.28x      0.28x         -
     75000     parallel        4      0.0178      0.0177      0.0184     33.721      0.35x      0.35x         -
     75000     parallel        8      0.0251      0.0249      0.0286     23.943      0.25x      0.25x         -
     75000     parallel       12      0.0314      0.0309      0.0344     19.082      0.20x      0.20x         -
     75000     parallel       16      0.0331      0.0326      0.0337     18.132      0.19x      0.19x         -
     75000     parallel       24      0.0384      0.0375      0.0393     15.624      0.16x      0.16x         -

    100000       scalar        -      0.0083      0.0083      0.0083     96.490      1.00x      1.00x         -
    100000   sequential        -      0.0083      0.0083      0.0083     96.823      1.00x      1.00x         -
    100000     parallel        1      0.0083      0.0083      0.0083     95.993      0.99x      0.99x         -
    100000     parallel        2      0.0259      0.0257      0.0284     30.934      0.32x      0.32x         -
    100000     parallel        4      0.0197      0.0196      0.0200     40.572      0.42x      0.42x         -
    100000     parallel        8      0.0265      0.0264      0.0269     30.209      0.31x      0.31x         -
    100000     parallel       12      0.0337      0.0334      0.0347     23.719      0.25x      0.24x         -
    100000     parallel       16      0.0344      0.0341      0.0351     23.274      0.24x      0.24x         -
    100000     parallel       24      0.0398      0.0392      0.0407     20.113      0.21x      0.21x         -

    200000       scalar        -      0.0167      0.0167      0.0168     95.746      1.00x      1.00x         -
    200000   sequential        -      0.0167      0.0167      0.0171     95.540      1.00x      1.00x         -
    200000     parallel        1      0.0168      0.0167      0.0170     95.416      1.00x      1.00x         -
    200000     parallel        2      0.0409      0.0409      0.0413     39.100      0.41x      0.41x         -
    200000     parallel        4      0.0279      0.0277      0.0291     57.300      0.60x      0.60x         -
    200000     parallel        8      0.0338      0.0336      0.0342     47.377      0.49x      0.50x         -
    200000     parallel       12      0.0392      0.0384      0.0402     40.779      0.43x      0.43x         -
    200000     parallel       16      0.0427      0.0419      0.0430     37.463      0.39x      0.39x         -
    200000     parallel       24      0.0463      0.0454      0.0475     34.557      0.36x      0.36x         -

    500000       scalar        -      0.0416      0.0416      0.0419     96.223      1.00x      1.01x         -
    500000   sequential        -      0.0421      0.0421      0.0424     94.952      0.99x      1.00x         -
    500000     parallel        1      0.0417      0.0416      0.0418     95.848      1.00x      1.01x         -
    500000     parallel        2      0.0856      0.0840      0.0873     46.712      0.49x      0.49x         -
    500000     parallel        4      0.0502      0.0501      0.0521     79.691      0.83x      0.84x         -
    500000     parallel        8      0.0634      0.0597      0.0682     63.116      0.66x      0.66x         -
    500000     parallel       12      0.0540      0.0506      0.0614     74.076      0.77x      0.78x         -
    500000     parallel       16      0.0677      0.0667      0.0736     59.084      0.61x      0.62x         -
    500000     parallel       24      0.0712      0.0688      0.0726     56.195      0.58x      0.59x         -

   1000000       scalar        -      0.0834      0.0832      0.1244     95.891      1.00x      1.01x         -
   1000000   sequential        -      0.0846      0.0845      0.0896     94.577      0.99x      1.00x         -
   1000000     parallel        1      0.0838      0.0835      0.0894     95.426      1.00x      1.01x         -
   1000000     parallel        2      0.1556      0.1555      0.1566     51.416      0.54x      0.54x         -
   1000000     parallel        4      0.0858      0.0856      0.0919     93.277      0.97x      0.99x         -
   1000000     parallel        8      0.0905      0.0903      0.0915     88.389      0.92x      0.93x         -
   1000000     parallel       12      0.0966      0.0950      0.1009     82.840      0.86x      0.88x         -
   1000000     parallel       16      0.1106      0.1024      0.1128     72.304      0.75x      0.76x         -
   1000000     parallel       24      0.1120      0.1086      0.1150     71.413      0.74x      0.76x         -

   2000000       scalar        -      0.1692      0.1674      0.1879     94.570      1.00x      0.99x         -
   2000000   sequential        -      0.1671      0.1664      0.1803     95.758      1.01x      1.00x         -
   2000000     parallel        1      0.1695      0.1689      0.1930     94.406      1.00x      0.99x         -
   2000000     parallel        2      0.2999      0.2995      0.3043     53.358      0.56x      0.56x         -
   2000000     parallel        4      0.1593      0.1579      0.1869    100.435      1.06x      1.05x         -
   2000000     parallel        8      0.1613      0.1600      0.1841     99.167      1.05x      1.04x         -
   2000000     parallel       12      0.1984      0.1782      0.2695     80.630      0.85x      0.84x         -
   2000000     parallel       16      0.1787      0.1678      0.1841     89.539      0.95x      0.94x         -
   2000000     parallel       24      0.2063      0.1894      0.2189     77.573      0.82x      0.81x         -

   5000000       scalar        -      0.9635      0.9520      1.0523     41.515      1.00x      1.03x         -
   5000000   sequential        -      0.9910      0.9637      1.0227     40.363      0.97x      1.00x         -
   5000000     parallel        1      0.9667      0.9507      1.0560     41.376      1.00x      1.03x         -
   5000000     parallel        2      0.9002      0.8828      1.0175     44.432      1.07x      1.10x         -
   5000000     parallel        4      0.6177      0.5505      0.7225     64.759      1.56x      1.60x         -
   5000000     parallel        8      0.5686      0.5242      0.7686     70.345      1.69x      1.74x         -
   5000000     parallel       12      0.6108      0.5627      0.6722     65.491      1.58x      1.62x         -
   5000000     parallel       16      0.6454      0.4058      0.7633     61.975      1.49x      1.54x         -
   5000000     parallel       24      0.6160      0.4710      0.8300     64.930      1.56x      1.61x         -

  10000000       scalar        -      2.4663      2.4253      2.7568     32.437      1.00x      1.03x         -
  10000000   sequential        -      2.5323      2.4368      2.8251     31.592      0.97x      1.00x         -
  10000000     parallel        1      2.4873      2.4059      2.7162     32.163      0.99x      1.02x         -
  10000000     parallel        2      2.1795      2.1379      2.4756     36.706      1.13x      1.16x         -
  10000000     parallel        4      2.2797      1.9513      2.6634     35.092      1.08x      1.11x         -
  10000000     parallel        8      1.2229      1.1699      1.7316     65.416      2.02x      2.07x         -
  10000000     parallel       12      2.1778      1.8655      2.4604     36.734      1.13x      1.16x         -
  10000000     parallel       16      2.0694      1.9228      2.3932     38.659      1.19x      1.22x         -
  10000000     parallel       24      2.2009      2.1170      2.2761     36.349      1.12x      1.15x         -

  33554432       scalar        -     11.1591     11.0624     11.8685     24.055      1.00x      0.97x      0.92
  33554432   sequential        -     10.8079     10.7574     11.1302     24.837      1.03x      1.00x         -
  33554432     parallel        1     11.2578     11.0759     12.1680     23.844      0.99x      0.96x      0.92
  33554432     parallel        2     10.4950     10.3662     11.3401     25.577      1.06x      1.03x         -
  33554432     parallel        4     10.8502     10.0749     11.4956     24.740      1.03x      1.00x         -
  33554432     parallel        8      9.6021      9.2773     10.3203     27.956      1.16x      1.13x         -
  33554432     parallel       12     10.1138      9.6919     10.5319     26.542      1.10x      1.07x         -
  33554432     parallel       16     10.1700      9.9794     10.8262     26.395      1.10x      1.06x         -
  33554432     parallel       24     10.3395     10.1528     10.8353     25.962      1.08x      1.05x         -


== square ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    109.168      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    109.168      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    108.963      1.00x      1.00x         -
      1000     parallel        2      0.0001      0.0001      0.0001    109.158      1.00x      1.00x         -
      1000     parallel        4      0.0001      0.0001      0.0001    109.158      1.00x      1.00x         -
      1000     parallel        8      0.0001      0.0001      0.0001    109.090      1.00x      1.00x         -
      1000     parallel       12      0.0001      0.0001      0.0001    109.163      1.00x      1.00x         -
      1000     parallel       16      0.0001      0.0001      0.0001    109.040      1.00x      1.00x         -
      1000     parallel       24      0.0001      0.0001      0.0001    109.049      1.00x      1.00x         -

     10000       scalar        -      0.0007      0.0007      0.0007    109.270      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    109.252      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0007    108.320      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0007    108.349      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0007    108.353      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0007    108.345      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0007    108.335      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0008    108.317      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0007    108.363      0.99x      0.99x         -

     32768       scalar        -      0.0024      0.0024      0.0024    110.268      1.00x      1.01x         -
     32768   sequential        -      0.0024      0.0024      0.0024    109.485      0.99x      1.00x         -
     32768     parallel        1      0.0024      0.0024      0.0024    110.024      1.00x      1.00x         -
     32768     parallel        2      0.0024      0.0024      0.0024    110.006      1.00x      1.00x         -
     32768     parallel        4      0.0024      0.0024      0.0024    110.033      1.00x      1.01x         -
     32768     parallel        8      0.0024      0.0024      0.0024    110.055      1.00x      1.01x         -
     32768     parallel       12      0.0024      0.0024      0.0024    110.015      1.00x      1.00x         -
     32768     parallel       16      0.0024      0.0024      0.0024    109.992      1.00x      1.00x         -
     32768     parallel       24      0.0024      0.0024      0.0025    110.037      1.00x      1.01x         -

     32769       scalar        -      0.0024      0.0024      0.0024    110.262      1.00x      1.01x         -
     32769   sequential        -      0.0024      0.0024      0.0024    109.506      0.99x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0025    109.977      1.00x      1.00x         -
     32769     parallel        2      0.0153      0.0152      0.0243     17.090      0.15x      0.16x         -
     32769     parallel        4      0.0150      0.0146      0.0250     17.477      0.16x      0.16x         -
     32769     parallel        8      0.0221      0.0220      0.0229     11.850      0.11x      0.11x         -
     32769     parallel       12      0.0279      0.0278      0.0281      9.382      0.09x      0.09x         -
     32769     parallel       16      0.0302      0.0299      0.0305      8.685      0.08x      0.08x         -
     32769     parallel       24      0.0367      0.0363      0.0372      7.149      0.06x      0.07x         -

     40000       scalar        -      0.0029      0.0029      0.0029    110.056      1.00x      1.00x         -
     40000   sequential        -      0.0029      0.0029      0.0029    110.093      1.00x      1.00x         -
     40000     parallel        1      0.0029      0.0029      0.0029    109.919      1.00x      1.00x         -
     40000     parallel        2      0.0163      0.0163      0.0164     19.621      0.18x      0.18x         -
     40000     parallel        4      0.0153      0.0153      0.0157     20.860      0.19x      0.19x         -
     40000     parallel        8      0.0229      0.0226      0.0237     13.950      0.13x      0.13x         -
     40000     parallel       12      0.0285      0.0283      0.0291     11.239      0.10x      0.10x         -
     40000     parallel       16      0.0311      0.0309      0.0320     10.281      0.09x      0.09x         -
     40000     parallel       24      0.0366      0.0364      0.0374      8.732      0.08x      0.08x         -

     50000       scalar        -      0.0037      0.0037      0.0037    107.772      1.00x      1.00x         -
     50000   sequential        -      0.0037      0.0037      0.0037    107.696      1.00x      1.00x         -
     50000     parallel        1      0.0037      0.0037      0.0038    107.191      0.99x      1.00x         -
     50000     parallel        2      0.0179      0.0179      0.0181     22.328      0.21x      0.21x         -
     50000     parallel        4      0.0161      0.0160      0.0165     24.857      0.23x      0.23x         -
     50000     parallel        8      0.0231      0.0229      0.0235     17.287      0.16x      0.16x         -
     50000     parallel       12      0.0295      0.0293      0.0297     13.582      0.13x      0.13x         -
     50000     parallel       16      0.0313      0.0310      0.0316     12.769      0.12x      0.12x         -
     50000     parallel       24      0.0401      0.0385      0.0408      9.965      0.09x      0.09x         -

     75000       scalar        -      0.0062      0.0062      0.0062     96.829      1.00x      0.99x         -
     75000   sequential        -      0.0062      0.0061      0.0062     97.539      1.01x      1.00x         -
     75000     parallel        1      0.0062      0.0062      0.0063     96.495      1.00x      0.99x         -
     75000     parallel        2      0.0208      0.0207      0.0227     28.845      0.30x      0.30x         -
     75000     parallel        4      0.0179      0.0178      0.0187     33.543      0.35x      0.34x         -
     75000     parallel        8      0.0251      0.0249      0.0255     23.898      0.25x      0.25x         -
     75000     parallel       12      0.0313      0.0308      0.0318     19.150      0.20x      0.20x         -
     75000     parallel       16      0.0327      0.0322      0.0335     18.351      0.19x      0.19x         -
     75000     parallel       24      0.0386      0.0376      0.0397     15.538      0.16x      0.16x         -

    100000       scalar        -      0.0084      0.0084      0.0084     95.723      1.00x      1.00x         -
    100000   sequential        -      0.0083      0.0083      0.0084     96.056      1.00x      1.00x         -
    100000     parallel        1      0.0084      0.0084      0.0084     95.531      1.00x      0.99x         -
    100000     parallel        2      0.0257      0.0256      0.0262     31.188      0.33x      0.32x         -
    100000     parallel        4      0.0196      0.0196      0.0198     40.790      0.43x      0.42x         -
    100000     parallel        8      0.0263      0.0262      0.0267     30.378      0.32x      0.32x         -
    100000     parallel       12      0.0336      0.0331      0.0343     23.817      0.25x      0.25x         -
    100000     parallel       16      0.0342      0.0336      0.0354     23.413      0.24x      0.24x         -
    100000     parallel       24      0.0394      0.0390      0.0422     20.287      0.21x      0.21x         -

    200000       scalar        -      0.0168      0.0168      0.0169     95.119      1.00x      1.00x         -
    200000   sequential        -      0.0169      0.0169      0.0169     94.898      1.00x      1.00x         -
    200000     parallel        1      0.0168      0.0168      0.0169     94.964      1.00x      1.00x         -
    200000     parallel        2      0.0428      0.0409      0.0442     37.419      0.39x      0.39x         -
    200000     parallel        4      0.0286      0.0279      0.0294     55.872      0.59x      0.59x         -
    200000     parallel        8      0.0356      0.0339      0.0384     45.003      0.47x      0.47x         -
    200000     parallel       12      0.0391      0.0380      0.0419     40.922      0.43x      0.43x         -
    200000     parallel       16      0.0433      0.0423      0.0461     36.970      0.39x      0.39x         -
    200000     parallel       24      0.0479      0.0463      0.0493     33.382      0.35x      0.35x         -

    500000       scalar        -      0.0418      0.0418      0.0419     95.676      1.00x      1.01x         -
    500000   sequential        -      0.0424      0.0423      0.0428     94.305      0.99x      1.00x         -
    500000     parallel        1      0.0417      0.0416      0.0419     95.877      1.00x      1.02x         -
    500000     parallel        2      0.0857      0.0840      0.0870     46.668      0.49x      0.49x         -
    500000     parallel        4      0.0513      0.0502      0.0520     77.992      0.82x      0.83x         -
    500000     parallel        8      0.0583      0.0550      0.0636     68.614      0.72x      0.73x         -
    500000     parallel       12      0.0624      0.0594      0.0637     64.148      0.67x      0.68x         -
    500000     parallel       16      0.0689      0.0681      0.0699     58.076      0.61x      0.62x         -
    500000     parallel       24      0.0730      0.0717      0.0845     54.806      0.57x      0.58x         -

   1000000       scalar        -      0.0835      0.0835      0.0889     95.762      1.00x      1.02x         -
   1000000   sequential        -      0.0853      0.0851      0.0901     93.818      0.98x      1.00x         -
   1000000     parallel        1      0.0836      0.0832      0.0907     95.654      1.00x      1.02x         -
   1000000     parallel        2      0.1560      0.1557      0.1574     51.268      0.54x      0.55x         -
   1000000     parallel        4      0.0868      0.0858      0.0950     92.126      0.96x      0.98x         -
   1000000     parallel        8      0.0910      0.0882      0.0926     87.918      0.92x      0.94x         -
   1000000     parallel       12      0.0975      0.0934      0.1093     82.067      0.86x      0.87x         -
   1000000     parallel       16      0.1048      0.1009      0.1055     76.338      0.80x      0.81x         -
   1000000     parallel       24      0.1102      0.1065      0.1139     72.595      0.76x      0.77x         -

   2000000       scalar        -      0.1679      0.1678      0.1928     95.288      1.00x      1.00x         -
   2000000   sequential        -      0.1685      0.1676      0.1923     94.959      1.00x      1.00x         -
   2000000     parallel        1      0.1694      0.1683      0.1928     94.475      0.99x      0.99x         -
   2000000     parallel        2      0.2995      0.2993      0.3022     53.427      0.56x      0.56x         -
   2000000     parallel        4      0.1590      0.1579      0.1934    100.633      1.06x      1.06x         -
   2000000     parallel        8      0.1792      0.1626      0.2145     89.283      0.94x      0.94x         -
   2000000     parallel       12      0.1784      0.1711      0.1914     89.689      0.94x      0.94x         -
   2000000     parallel       16      0.1789      0.1661      0.2003     89.460      0.94x      0.94x         -
   2000000     parallel       24      0.2212      0.2094      0.2842     72.345      0.76x      0.76x         -

   5000000       scalar        -      0.9814      0.9755      1.0309     40.759      1.00x      0.99x         -
   5000000   sequential        -      0.9759      0.9667      0.9938     40.986      1.01x      1.00x         -
   5000000     parallel        1      0.9942      0.9647      1.0413     40.231      0.99x      0.98x         -
   5000000     parallel        2      0.8947      0.8876      0.9641     44.709      1.10x      1.09x         -
   5000000     parallel        4      0.5648      0.5313      0.5866     70.828      1.74x      1.73x         -
   5000000     parallel        8      0.4121      0.3453      0.8931     97.058      2.38x      2.37x         -
   5000000     parallel       12      0.6256      0.5248      0.7006     63.936      1.57x      1.56x         -
   5000000     parallel       16      0.5139      0.4751      0.5938     77.840      1.91x      1.90x         -
   5000000     parallel       24      0.4778      0.3719      0.6939     83.715      2.05x      2.04x         -

  10000000       scalar        -      2.5208      2.4092      2.8424     31.736      1.00x      0.99x         -
  10000000   sequential        -      2.4900      2.4509      2.7822     32.129      1.01x      1.00x         -
  10000000     parallel        1      2.4918      2.4182      2.8339     32.105      1.01x      1.00x         -
  10000000     parallel        2      2.2000      2.1341      2.8056     36.364      1.15x      1.13x         -
  10000000     parallel        4      2.4095      2.1545      3.1610     33.202      1.05x      1.03x         -
  10000000     parallel        8      1.4366      1.4083      1.9154     55.685      1.75x      1.73x         -
  10000000     parallel       12      1.8473      1.8085      2.0316     43.306      1.36x      1.35x         -
  10000000     parallel       16      2.0932      2.0484      2.2800     38.219      1.20x      1.19x         -
  10000000     parallel       24      2.2505      2.1512      2.3486     35.548      1.12x      1.11x         -

  33554432       scalar        -     11.2548     11.1001     16.8361     23.851      1.00x      0.96x      1.17
  33554432   sequential        -     10.8129     10.6981     11.4458     24.825      1.04x      1.00x         -
  33554432     parallel        1     11.0823     11.0418     12.0218     24.222      1.02x      0.98x      1.08
  33554432     parallel        2     10.6225     10.3768     11.2192     25.270      1.06x      1.02x         -
  33554432     parallel        4     10.1607     10.0326     11.2653     26.419      1.11x      1.06x         -
  33554432     parallel        8      9.5995      9.2419     10.0455     27.963      1.17x      1.13x         -
  33554432     parallel       12     10.0791      9.6714     10.4553     26.633      1.12x      1.07x         -
  33554432     parallel       16     10.2000     10.0181     10.3238     26.317      1.10x      1.06x         -
  33554432     parallel       24     10.3642     10.0292     10.7768     25.900      1.09x      1.04x         -


== relu ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0001      0.0001      0.0001    109.149      1.00x      1.00x         -
      1000   sequential        -      0.0001      0.0001      0.0001    109.154      1.00x      1.00x         -
      1000     parallel        1      0.0001      0.0001      0.0001    108.991      1.00x      1.00x         -
      1000     parallel        2      0.0001      0.0001      0.0001    109.022      1.00x      1.00x         -
      1000     parallel        4      0.0001      0.0001      0.0001    108.968      1.00x      1.00x         -
      1000     parallel        8      0.0001      0.0001      0.0001    108.991      1.00x      1.00x         -
      1000     parallel       12      0.0001      0.0001      0.0001    108.968      1.00x      1.00x         -
      1000     parallel       16      0.0001      0.0001      0.0001    109.013      1.00x      1.00x         -
      1000     parallel       24      0.0001      0.0001      0.0001    108.995      1.00x      1.00x         -

     10000       scalar        -      0.0007      0.0007      0.0007    107.864      1.00x      1.00x         -
     10000   sequential        -      0.0007      0.0007      0.0007    108.063      1.00x      1.00x         -
     10000     parallel        1      0.0007      0.0007      0.0008    107.050      0.99x      0.99x         -
     10000     parallel        2      0.0007      0.0007      0.0007    107.057      0.99x      0.99x         -
     10000     parallel        4      0.0007      0.0007      0.0008    107.064      0.99x      0.99x         -
     10000     parallel        8      0.0007      0.0007      0.0008    107.074      0.99x      0.99x         -
     10000     parallel       12      0.0007      0.0007      0.0008    107.046      0.99x      0.99x         -
     10000     parallel       16      0.0007      0.0007      0.0008    107.088      0.99x      0.99x         -
     10000     parallel       24      0.0007      0.0007      0.0008    106.865      0.99x      0.99x         -

     32768       scalar        -      0.0024      0.0024      0.0024    108.987      1.00x      1.00x         -
     32768   sequential        -      0.0024      0.0024      0.0024    109.151      1.00x      1.00x         -
     32768     parallel        1      0.0024      0.0024      0.0024    108.617      1.00x      1.00x         -
     32768     parallel        2      0.0024      0.0024      0.0024    108.525      1.00x      0.99x         -
     32768     parallel        4      0.0024      0.0024      0.0025    108.573      1.00x      0.99x         -
     32768     parallel        8      0.0024      0.0024      0.0024    108.498      1.00x      0.99x         -
     32768     parallel       12      0.0024      0.0024      0.0024    108.428      0.99x      0.99x         -
     32768     parallel       16      0.0024      0.0024      0.0024    108.634      1.00x      1.00x         -
     32768     parallel       24      0.0024      0.0024      0.0025    108.529      1.00x      0.99x         -

     32769       scalar        -      0.0024      0.0024      0.0024    108.924      1.00x      1.00x         -
     32769   sequential        -      0.0024      0.0024      0.0024    109.128      1.00x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0024    108.682      1.00x      1.00x         -
     32769     parallel        2      0.0153      0.0152      0.0155     17.182      0.16x      0.16x         -
     32769     parallel        4      0.0147      0.0146      0.0148     17.833      0.16x      0.16x         -
     32769     parallel        8      0.0220      0.0219      0.0224     11.911      0.11x      0.11x         -
     32769     parallel       12      0.0279      0.0277      0.0281      9.382      0.09x      0.09x         -
     32769     parallel       16      0.0302      0.0300      0.0307      8.684      0.08x      0.08x         -
     32769     parallel       24      0.0363      0.0359      0.0370      7.214      0.07x      0.07x         -

     40000       scalar        -      0.0029      0.0029      0.0030    108.500      1.00x      1.00x         -
     40000   sequential        -      0.0029      0.0029      0.0030    108.799      1.00x      1.00x         -
     40000     parallel        1      0.0029      0.0029      0.0030    108.561      1.00x      1.00x         -
     40000     parallel        2      0.0164      0.0163      0.0166     19.561      0.18x      0.18x         -
     40000     parallel        4      0.0153      0.0152      0.0156     20.873      0.19x      0.19x         -
     40000     parallel        8      0.0227      0.0225      0.0230     14.094      0.13x      0.13x         -
     40000     parallel       12      0.0285      0.0284      0.0288     11.212      0.10x      0.10x         -
     40000     parallel       16      0.0311      0.0306      0.0323     10.280      0.09x      0.09x         -
     40000     parallel       24      0.0372      0.0365      0.0385      8.599      0.08x      0.08x         -

     50000       scalar        -      0.0037      0.0037      0.0037    108.616      1.00x      1.04x         -
     50000   sequential        -      0.0038      0.0038      0.0040    104.813      0.96x      1.00x         -
     50000     parallel        1      0.0037      0.0037      0.0037    108.558      1.00x      1.04x         -
     50000     parallel        2      0.0180      0.0178      0.0183     22.217      0.20x      0.21x         -
     50000     parallel        4      0.0160      0.0158      0.0164     25.062      0.23x      0.24x         -
     50000     parallel        8      0.0241      0.0232      0.0260     16.620      0.15x      0.16x         -
     50000     parallel       12      0.0292      0.0289      0.0297     13.713      0.13x      0.13x         -
     50000     parallel       16      0.0318      0.0309      0.0327     12.573      0.12x      0.12x         -
     50000     parallel       24      0.0400      0.0382      0.0411      9.994      0.09x      0.10x         -

     75000       scalar        -      0.0062      0.0062      0.0062     96.586      1.00x      1.01x         -
     75000   sequential        -      0.0063      0.0063      0.0063     95.422      0.99x      1.00x         -
     75000     parallel        1      0.0062      0.0062      0.0064     96.970      1.00x      1.02x         -
     75000     parallel        2      0.0221      0.0219      0.0223     27.177      0.28x      0.28x         -
     75000     parallel        4      0.0190      0.0186      0.0192     31.658      0.33x      0.33x         -
     75000     parallel        8      0.0252      0.0249      0.0272     23.824      0.25x      0.25x         -
     75000     parallel       12      0.0314      0.0311      0.0331     19.123      0.20x      0.20x         -
     75000     parallel       16      0.0328      0.0325      0.0333     18.266      0.19x      0.19x         -
     75000     parallel       24      0.0386      0.0371      0.0412     15.533      0.16x      0.16x         -

    100000       scalar        -      0.0085      0.0085      0.0085     94.587      1.00x      0.99x         -
    100000   sequential        -      0.0084      0.0084      0.0084     95.167      1.01x      1.00x         -
    100000     parallel        1      0.0085      0.0084      0.0085     94.583      1.00x      0.99x         -
    100000     parallel        2      0.0257      0.0257      0.0259     31.128      0.33x      0.33x         -
    100000     parallel        4      0.0198      0.0197      0.0202     40.404      0.43x      0.42x         -
    100000     parallel        8      0.0264      0.0263      0.0267     30.328      0.32x      0.32x         -
    100000     parallel       12      0.0339      0.0330      0.0350     23.633      0.25x      0.25x         -
    100000     parallel       16      0.0346      0.0336      0.0348     23.106      0.24x      0.24x         -
    100000     parallel       24      0.0400      0.0393      0.0406     19.995      0.21x      0.21x         -

    200000       scalar        -      0.0169      0.0169      0.0170     94.509      1.00x      1.00x         -
    200000   sequential        -      0.0170      0.0170      0.0170     94.339      1.00x      1.00x         -
    200000     parallel        1      0.0169      0.0169      0.0172     94.434      1.00x      1.00x         -
    200000     parallel        2      0.0411      0.0410      0.0415     38.969      0.41x      0.41x         -
    200000     parallel        4      0.0280      0.0279      0.0297     57.097      0.60x      0.61x         -
    200000     parallel        8      0.0341      0.0336      0.0346     46.929      0.50x      0.50x         -
    200000     parallel       12      0.0391      0.0388      0.0404     40.890      0.43x      0.43x         -
    200000     parallel       16      0.0424      0.0412      0.0441     37.729      0.40x      0.40x         -
    200000     parallel       24      0.0476      0.0468      0.0492     33.611      0.36x      0.36x         -

    500000       scalar        -      0.0420      0.0420      0.0422     95.196      1.00x      1.01x         -
    500000   sequential        -      0.0426      0.0426      0.0427     93.790      0.99x      1.00x         -
    500000     parallel        1      0.0420      0.0419      0.0421     95.295      1.00x      1.02x         -
    500000     parallel        2      0.0840      0.0824      0.0846     47.603      0.50x      0.51x         -
    500000     parallel        4      0.0498      0.0497      0.0517     80.357      0.84x      0.86x         -
    500000     parallel        8      0.0551      0.0549      0.0662     72.614      0.76x      0.77x         -
    500000     parallel       12      0.0601      0.0586      0.0618     66.561      0.70x      0.71x         -
    500000     parallel       16      0.0691      0.0673      0.0705     57.908      0.61x      0.62x         -
    500000     parallel       24      0.0736      0.0712      0.0744     54.327      0.57x      0.58x         -

   1000000       scalar        -      0.0842      0.0841      0.0894     95.026      1.00x      1.02x         -
   1000000   sequential        -      0.0858      0.0856      0.0923     93.203      0.98x      1.00x         -
   1000000     parallel        1      0.0845      0.0844      0.0883     94.643      1.00x      1.02x         -
   1000000     parallel        2      0.1560      0.1557      0.1601     51.268      0.54x      0.55x         -
   1000000     parallel        4      0.0859      0.0857      0.0943     93.159      0.98x      1.00x         -
   1000000     parallel        8      0.0903      0.0900      0.0996     88.615      0.93x      0.95x         -
   1000000     parallel       12      0.0987      0.0934      0.1069     81.036      0.85x      0.87x         -
   1000000     parallel       16      0.1103      0.1073      0.1124     72.501      0.76x      0.78x         -
   1000000     parallel       24      0.1096      0.1026      0.1179     72.999      0.77x      0.78x         -

   2000000       scalar        -      0.1709      0.1695      0.1876     93.649      1.00x      0.99x         -
   2000000   sequential        -      0.1693      0.1683      0.1923     94.524      1.01x      1.00x         -
   2000000     parallel        1      0.1704      0.1695      0.1931     93.924      1.00x      0.99x         -
   2000000     parallel        2      0.3011      0.2996      0.3133     53.147      0.57x      0.56x         -
   2000000     parallel        4      0.1582      0.1575      0.1878    101.162      1.08x      1.07x         -
   2000000     parallel        8      0.1722      0.1606      0.2117     92.939      0.99x      0.98x         -
   2000000     parallel       12      0.2098      0.1962      0.2310     76.270      0.81x      0.81x         -
   2000000     parallel       16      0.2268      0.1987      0.3372     70.553      0.75x      0.75x         -
   2000000     parallel       24      0.2664      0.2324      0.3486     60.060      0.64x      0.64x         -

   5000000       scalar        -      1.0704      1.0513      1.1041     37.367      1.00x      1.02x         -
   5000000   sequential        -      1.0939      1.0641      1.1569     36.565      0.98x      1.00x         -
   5000000     parallel        1      1.0579      1.0435      1.0787     37.809      1.01x      1.03x         -
   5000000     parallel        2      0.8854      0.8787      0.8962     45.176      1.21x      1.24x         -
   5000000     parallel        4      0.6700      0.5925      0.7423     59.701      1.60x      1.63x         -
   5000000     parallel        8      0.5407      0.4279      0.7039     73.971      1.98x      2.02x         -
   5000000     parallel       12      0.3863      0.3589      0.7505    103.540      2.77x      2.83x         -
   5000000     parallel       16      0.4926      0.4407      0.8599     81.208      2.17x      2.22x         -
   5000000     parallel       24      0.6119      0.4866      0.8122     65.376      1.75x      1.79x         -

  10000000       scalar        -      2.6438      2.5678      2.8522     30.259      1.00x      1.00x         -
  10000000   sequential        -      2.6322      2.5546      2.9249     30.393      1.00x      1.00x         -
  10000000     parallel        1      2.7990      2.6704      3.1028     28.582      0.94x      0.94x         -
  10000000     parallel        2      2.2740      2.1256      3.4264     35.180      1.16x      1.16x         -
  10000000     parallel        4      1.9827      1.9395      2.0347     40.349      1.33x      1.33x         -
  10000000     parallel        8      1.4343      1.3821      1.5145     55.776      1.84x      1.84x         -
  10000000     parallel       12      1.8538      1.8273      1.8955     43.155      1.43x      1.42x         -
  10000000     parallel       16      2.2117      2.1251      2.3302     36.172      1.20x      1.19x         -
  10000000     parallel       24      2.2043      2.0647      2.3295     36.293      1.20x      1.19x         -

  33554432       scalar        -     11.6778     11.5281     11.8831     22.987      1.00x      0.95x      0.89
  33554432   sequential        -     11.0497     10.9191     11.9953     24.293      1.06x      1.00x      0.93
  33554432     parallel        1     11.6152     11.4893     11.8450     23.111      1.01x      0.95x      0.90
  33554432     parallel        2     10.6384     10.4219     11.8596     25.233      1.10x      1.04x         -
  33554432     parallel        4     10.2186     10.0922     11.0134     26.269      1.14x      1.08x         -
  33554432     parallel        8      9.5154      9.3347     10.0041     28.211      1.23x      1.16x         -
  33554432     parallel       12      9.8029      9.4309     10.0243     27.383      1.19x      1.13x         -
  33554432     parallel       16     10.0954      9.9342     10.6779     26.590      1.16x      1.09x         -
  33554432     parallel       24     10.2504      9.9900     11.0719     26.188      1.14x      1.08x         -


== sqrt ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0004      0.0004      0.0004     22.359      1.00x      1.00x         -
      1000   sequential        -      0.0004      0.0004      0.0004     22.357      1.00x      1.00x         -
      1000     parallel        1      0.0004      0.0004      0.0004     22.357      1.00x      1.00x         -
      1000     parallel        2      0.0004      0.0004      0.0004     22.356      1.00x      1.00x         -
      1000     parallel        4      0.0004      0.0004      0.0004     22.363      1.00x      1.00x         -
      1000     parallel        8      0.0004      0.0004      0.0004     22.353      1.00x      1.00x         -
      1000     parallel       12      0.0004      0.0004      0.0004     22.349      1.00x      1.00x         -
      1000     parallel       16      0.0004      0.0004      0.0004     22.356      1.00x      1.00x         -
      1000     parallel       24      0.0004      0.0004      0.0004     22.357      1.00x      1.00x         -

     10000       scalar        -      0.0036      0.0036      0.0036     22.355      1.00x      1.00x         -
     10000   sequential        -      0.0036      0.0036      0.0036     22.337      1.00x      1.00x         -
     10000     parallel        1      0.0036      0.0036      0.0036     22.345      1.00x      1.00x         -
     10000     parallel        2      0.0036      0.0036      0.0036     22.352      1.00x      1.00x         -
     10000     parallel        4      0.0036      0.0036      0.0039     22.306      1.00x      1.00x         -
     10000     parallel        8      0.0036      0.0036      0.0036     22.351      1.00x      1.00x         -
     10000     parallel       12      0.0036      0.0036      0.0036     22.356      1.00x      1.00x         -
     10000     parallel       16      0.0036      0.0036      0.0036     22.354      1.00x      1.00x         -
     10000     parallel       24      0.0036      0.0036      0.0036     22.338      1.00x      1.00x         -

     32768       scalar        -      0.0117      0.0117      0.0117     22.347      1.00x      1.00x         -
     32768   sequential        -      0.0117      0.0117      0.0117     22.352      1.00x      1.00x         -
     32768     parallel        1      0.0117      0.0117      0.0117     22.353      1.00x      1.00x         -
     32768     parallel        2      0.0117      0.0117      0.0117     22.354      1.00x      1.00x         -
     32768     parallel        4      0.0117      0.0117      0.0118     22.355      1.00x      1.00x         -
     32768     parallel        8      0.0117      0.0117      0.0118     22.351      1.00x      1.00x         -
     32768     parallel       12      0.0117      0.0117      0.0118     22.353      1.00x      1.00x         -
     32768     parallel       16      0.0117      0.0117      0.0117     22.354      1.00x      1.00x         -
     32768     parallel       24      0.0117      0.0117      0.0118     22.354      1.00x      1.00x         -

     32769       scalar        -      0.0117      0.0117      0.0119     22.354      1.00x      1.00x         -
     32769   sequential        -      0.0117      0.0117      0.0118     22.352      1.00x      1.00x         -
     32769     parallel        1      0.0117      0.0117      0.0118     22.347      1.00x      1.00x         -
     32769     parallel        2      0.0340      0.0339      0.0347      7.709      0.34x      0.34x         -
     32769     parallel        4      0.0241      0.0240      0.0245     10.897      0.49x      0.49x         -
     32769     parallel        8      0.0256      0.0255      0.0260     10.243      0.46x      0.46x         -
     32769     parallel       12      0.0297      0.0296      0.0305      8.815      0.39x      0.39x         -
     32769     parallel       16      0.0314      0.0311      0.0333      8.347      0.37x      0.37x         -
     32769     parallel       24      0.0417      0.0412      0.0421      6.293      0.28x      0.28x         -

     40000       scalar        -      0.0143      0.0143      0.0143     22.330      1.00x      1.00x         -
     40000   sequential        -      0.0143      0.0143      0.0143     22.328      1.00x      1.00x         -
     40000     parallel        1      0.0143      0.0143      0.0144     22.315      1.00x      1.00x         -
     40000     parallel        2      0.0393      0.0392      0.0403      8.137      0.36x      0.36x         -
     40000     parallel        4      0.0268      0.0267      0.0273     11.927      0.53x      0.53x         -
     40000     parallel        8      0.0274      0.0272      0.0304     11.663      0.52x      0.52x         -
     40000     parallel       12      0.0302      0.0300      0.0310     10.591      0.47x      0.47x         -
     40000     parallel       16      0.0326      0.0325      0.0329      9.824      0.44x      0.44x         -
     40000     parallel       24      0.0422      0.0420      0.0427      7.587      0.34x      0.34x         -

     50000       scalar        -      0.0180      0.0180      0.0180     22.236      1.00x      1.00x         -
     50000   sequential        -      0.0180      0.0180      0.0182     22.233      1.00x      1.00x         -
     50000     parallel        1      0.0179      0.0179      0.0181     22.352      1.01x      1.01x         -
     50000     parallel        2      0.0480      0.0476      0.0487      8.334      0.37x      0.37x         -
     50000     parallel        4      0.0306      0.0299      0.0318     13.093      0.59x      0.59x         -
     50000     parallel        8      0.0286      0.0284      0.0295     14.007      0.63x      0.63x         -
     50000     parallel       12      0.0315      0.0313      0.0320     12.683      0.57x      0.57x         -
     50000     parallel       16      0.0343      0.0342      0.0357     11.673      0.52x      0.53x         -
     50000     parallel       24      0.0437      0.0434      0.0444      9.147      0.41x      0.41x         -

     75000       scalar        -      0.0269      0.0269      0.0273     22.269      1.00x      1.00x         -
     75000   sequential        -      0.0269      0.0269      0.0270     22.269      1.00x      1.00x         -
     75000     parallel        1      0.0270      0.0269      0.0272     22.262      1.00x      1.00x         -
     75000     parallel        2      0.0645      0.0645      0.0658      9.302      0.42x      0.42x         -
     75000     parallel        4      0.0395      0.0392      0.0400     15.204      0.68x      0.68x         -
     75000     parallel        8      0.0337      0.0336      0.0349     17.807      0.80x      0.80x         -
     75000     parallel       12      0.0344      0.0342      0.0351     17.450      0.78x      0.78x         -
     75000     parallel       16      0.0388      0.0387      0.0394     15.453      0.69x      0.69x         -
     75000     parallel       24      0.0445      0.0435      0.0452     13.494      0.61x      0.61x         -

    100000       scalar        -      0.0359      0.0359      0.0360     22.279      1.00x      1.00x         -
    100000   sequential        -      0.0359      0.0359      0.0360     22.280      1.00x      1.00x         -
    100000     parallel        1      0.0359      0.0359      0.0360     22.272      1.00x      1.00x         -
    100000     parallel        2      0.0835      0.0831      0.0845      9.586      0.43x      0.43x         -
    100000     parallel        4      0.0491      0.0484      0.0511     16.283      0.73x      0.73x         -
    100000     parallel        8      0.0386      0.0379      0.0390     20.742      0.93x      0.93x         -
    100000     parallel       12      0.0408      0.0407      0.0413     19.585      0.88x      0.88x         -
    100000     parallel       16      0.0432      0.0430      0.0436     18.519      0.83x      0.83x         -
    100000     parallel       24      0.0453      0.0445      0.0463     17.641      0.79x      0.79x         -

    200000       scalar        -      0.0718      0.0718      0.0729     22.294      1.00x      1.00x         -
    200000   sequential        -      0.0717      0.0717      0.0719     22.300      1.00x      1.00x         -
    200000     parallel        1      0.0718      0.0718      0.0723     22.291      1.00x      1.00x         -
    200000     parallel        2      0.1566      0.1565      0.1569     10.215      0.46x      0.46x         -
    200000     parallel        4      0.0862      0.0862      0.0872     18.551      0.83x      0.83x         -
    200000     parallel        8      0.0561      0.0555      0.0580     28.501      1.28x      1.28x         -
    200000     parallel       12      0.0493      0.0491      0.0497     32.460      1.46x      1.46x         -
    200000     parallel       16      0.0611      0.0604      0.0631     26.187      1.17x      1.17x         -
    200000     parallel       24      0.0546      0.0541      0.0603     29.302      1.31x      1.31x         -

    500000       scalar        -      0.1794      0.1793      0.1797     22.301      1.00x      1.00x         -
    500000   sequential        -      0.1795      0.1793      0.1799     22.284      1.00x      1.00x         -
    500000     parallel        1      0.1795      0.1793      0.1797     22.287      1.00x      1.00x         -
    500000     parallel        2      0.3715      0.3715      0.3720     10.766      0.48x      0.48x         -
    500000     parallel        4      0.1948      0.1946      0.1961     20.531      0.92x      0.92x         -
    500000     parallel        8      0.1103      0.1100      0.1133     36.250      1.63x      1.63x         -
    500000     parallel       12      0.0862      0.0859      0.0914     46.412      2.08x      2.08x         -
    500000     parallel       16      0.1138      0.1136      0.1143     35.162      1.58x      1.58x         -
    500000     parallel       24      0.0907      0.0903      0.0935     44.101      1.98x      1.98x         -

   1000000       scalar        -      0.3587      0.3586      0.3625     22.304      1.00x      1.00x         -
   1000000   sequential        -      0.3593      0.3589      0.3632     22.267      1.00x      1.00x         -
   1000000     parallel        1      0.3596      0.3593      0.3597     22.246      1.00x      1.00x         -
   1000000     parallel        2      0.7308      0.7272      0.7966     10.947      0.49x      0.49x         -
   1000000     parallel        4      0.3726      0.3722      0.3735     21.471      0.96x      0.96x         -
   1000000     parallel        8      0.2017      0.2003      0.2050     39.664      1.78x      1.78x         -
   1000000     parallel       12      0.1477      0.1471      0.2114     54.164      2.43x      2.43x         -
   1000000     parallel       16      0.2053      0.2048      0.2299     38.960      1.75x      1.75x         -
   1000000     parallel       24      0.1530      0.1524      0.2088     52.281      2.34x      2.35x         -

   2000000       scalar        -      0.7177      0.7170      0.7180     22.293      1.00x      1.00x         -
   2000000   sequential        -      0.7179      0.7173      0.7207     22.286      1.00x      1.00x         -
   2000000     parallel        1      0.7192      0.7185      0.7201     22.245      1.00x      1.00x         -
   2000000     parallel        2      1.4468      1.4402      1.4485     11.059      0.50x      0.50x         -
   2000000     parallel        4      0.7414      0.7312      0.8077     21.582      0.97x      0.97x         -
   2000000     parallel        8      0.3801      0.3798      0.3812     42.091      1.89x      1.89x         -
   2000000     parallel       12      0.2736      0.2731      0.2805     58.477      2.62x      2.62x         -
   2000000     parallel       16      0.3845      0.3834      0.3863     41.618      1.87x      1.87x         -
   2000000     parallel       24      0.2796      0.2790      0.2848     57.219      2.57x      2.57x         -

   5000000       scalar        -      1.9558      1.9409      1.9642     20.452      1.00x      0.99x         -
   5000000   sequential        -      1.9384      1.9047      1.9941     20.636      1.01x      1.00x         -
   5000000     parallel        1      1.9720      1.9601      1.9918     20.284      0.99x      0.98x         -
   5000000     parallel        2      3.5979      3.5958      3.7285     11.118      0.54x      0.54x         -
   5000000     parallel        4      1.8194      1.8157      1.8403     21.986      1.07x      1.07x         -
   5000000     parallel        8      0.9209      0.9193      0.9325     43.436      2.12x      2.10x         -
   5000000     parallel       12      0.9255      0.6334      1.2151     43.219      2.11x      2.09x         -
   5000000     parallel       16      0.9222      0.9213      0.9250     43.375      2.12x      2.10x         -
   5000000     parallel       24      0.6408      0.6348      0.6500     62.427      3.05x      3.03x         -

  10000000       scalar        -      3.9798      3.9666      3.9926     20.102      1.00x      1.01x         -
  10000000   sequential        -      4.0028      3.9989      4.0403     19.986      0.99x      1.00x         -
  10000000     parallel        1      3.9786      3.9733      4.1192     20.108      1.00x      1.01x         -
  10000000     parallel        2      7.2664      7.2215      7.4888     11.010      0.55x      0.55x         -
  10000000     parallel        4      3.6759      3.6457      3.7233     21.763      1.08x      1.09x         -
  10000000     parallel        8      1.8472      1.8430      1.9552     43.309      2.15x      2.17x         -
  10000000     parallel       12      2.0727      1.8089      2.5116     38.597      1.92x      1.93x         -
  10000000     parallel       16      2.4066      2.1725      2.5738     33.242      1.65x      1.66x         -
  10000000     parallel       24      2.3426      2.2247      2.6728     34.150      1.70x      1.71x         -

  33554432       scalar        -     14.4865     14.3778     14.7418     18.530      1.00x      0.94x      1.08
  33554432   sequential        -     13.5571     13.4831     13.5937     19.800      1.07x      1.00x      1.03
  33554432     parallel        1     14.4748     14.4164     14.7727     18.545      1.00x      0.94x      0.84
  33554432     parallel        2     24.9562     24.8694     25.3726     10.756      0.58x      0.54x      1.94
  33554432     parallel        4     12.8538     12.8250     22.6689     20.884      1.13x      1.05x      3.84
  33554432     parallel        8      9.4432      9.3530     10.3291     28.426      1.53x      1.44x         -
  33554432     parallel       12     10.1774      9.8698     10.5267     26.376      1.42x      1.33x         -
  33554432     parallel       16     10.3167     10.1367     10.4858     26.020      1.40x      1.31x         -
  33554432     parallel       24     10.9048     10.2365     11.7777     24.616      1.33x      1.24x         -


== exp ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0026      0.0026      0.0026      3.094      1.00x      1.00x         -
      1000   sequential        -      0.0026      0.0026      0.0026      3.093      1.00x      1.00x         -
      1000     parallel        1      0.0026      0.0026      0.0026      3.092      1.00x      1.00x         -
      1000     parallel        2      0.0026      0.0026      0.0026      3.093      1.00x      1.00x         -
      1000     parallel        4      0.0026      0.0026      0.0026      3.094      1.00x      1.00x         -
      1000     parallel        8      0.0026      0.0026      0.0026      3.092      1.00x      1.00x         -
      1000     parallel       12      0.0026      0.0026      0.0027      3.092      1.00x      1.00x         -
      1000     parallel       16      0.0026      0.0026      0.0026      3.094      1.00x      1.00x         -
      1000     parallel       24      0.0026      0.0026      0.0026      3.092      1.00x      1.00x         -

     10000       scalar        -      0.0258      0.0258      0.0261      3.101      1.00x      1.00x         -
     10000   sequential        -      0.0258      0.0258      0.0264      3.102      1.00x      1.00x         -
     10000     parallel        1      0.0258      0.0258      0.0260      3.102      1.00x      1.00x         -
     10000     parallel        2      0.0258      0.0258      0.0262      3.101      1.00x      1.00x         -
     10000     parallel        4      0.0258      0.0258      0.0262      3.102      1.00x      1.00x         -
     10000     parallel        8      0.0258      0.0258      0.0271      3.102      1.00x      1.00x         -
     10000     parallel       12      0.0258      0.0258      0.0259      3.097      1.00x      1.00x         -
     10000     parallel       16      0.0258      0.0258      0.0262      3.102      1.00x      1.00x         -
     10000     parallel       24      0.0258      0.0258      0.0260      3.101      1.00x      1.00x         -

     32768       scalar        -      0.0846      0.0845      0.0900      3.100      1.00x      1.00x         -
     32768   sequential        -      0.0845      0.0845      0.0854      3.102      1.00x      1.00x         -
     32768     parallel        1      0.0845      0.0845      0.0855      3.102      1.00x      1.00x         -
     32768     parallel        2      0.0846      0.0845      0.0890      3.099      1.00x      1.00x         -
     32768     parallel        4      0.0846      0.0845      0.0848      3.099      1.00x      1.00x         -
     32768     parallel        8      0.0846      0.0845      0.0855      3.100      1.00x      1.00x         -
     32768     parallel       12      0.0846      0.0845      0.0853      3.100      1.00x      1.00x         -
     32768     parallel       16      0.0845      0.0845      0.0857      3.102      1.00x      1.00x         -
     32768     parallel       24      0.0845      0.0845      0.0900      3.102      1.00x      1.00x         -

     32769       scalar        -      0.0846      0.0845      0.0871      3.100      1.00x      1.00x         -
     32769   sequential        -      0.0845      0.0844      0.0852      3.102      1.00x      1.00x         -
     32769     parallel        1      0.0845      0.0845      0.0851      3.102      1.00x      1.00x         -
     32769     parallel        2      0.0528      0.0520      0.0569      4.963      1.60x      1.60x         -
     32769     parallel        4      0.0335      0.0334      0.0342      7.835      2.53x      2.53x         -
     32769     parallel        8      0.0312      0.0312      0.0321      8.394      2.71x      2.71x         -
     32769     parallel       12      0.0333      0.0331      0.0337      7.884      2.54x      2.54x         -
     32769     parallel       16      0.0369      0.0368      0.0378      7.097      2.29x      2.29x         -
     32769     parallel       24      0.0467      0.0463      0.0470      5.615      1.81x      1.81x         -

     40000       scalar        -      0.1032      0.1031      0.1041      3.102      1.00x      1.00x         -
     40000   sequential        -      0.1032      0.1031      0.1042      3.102      1.00x      1.00x         -
     40000     parallel        1      0.1032      0.1031      0.1042      3.102      1.00x      1.00x         -
     40000     parallel        2      0.0623      0.0621      0.0700      5.135      1.66x      1.66x         -
     40000     parallel        4      0.0383      0.0379      0.0397      8.350      2.69x      2.69x         -
     40000     parallel        8      0.0337      0.0333      0.0347      9.489      3.06x      3.06x         -
     40000     parallel       12      0.0345      0.0343      0.0350      9.267      2.99x      2.99x         -
     40000     parallel       16      0.0393      0.0391      0.0397      8.151      2.63x      2.63x         -
     40000     parallel       24      0.0472      0.0471      0.0479      6.778      2.19x      2.18x         -

     50000       scalar        -      0.1290      0.1290      0.1316      3.101      1.00x      1.00x         -
     50000   sequential        -      0.1289      0.1289      0.1291      3.102      1.00x      1.00x         -
     50000     parallel        1      0.1290      0.1290      0.1292      3.101      1.00x      1.00x         -
     50000     parallel        2      0.0753      0.0751      0.0757      5.314      1.71x      1.71x         -
     50000     parallel        4      0.0452      0.0447      0.0463      8.850      2.85x      2.85x         -
     50000     parallel        8      0.0373      0.0368      0.0375     10.728      3.46x      3.46x         -
     50000     parallel       12      0.0370      0.0367      0.0378     10.809      3.49x      3.48x         -
     50000     parallel       16      0.0424      0.0421      0.0429      9.429      3.04x      3.04x         -
     50000     parallel       24      0.0492      0.0489      0.0506      8.136      2.62x      2.62x         -

     75000       scalar        -      0.1938      0.1936      0.1940      3.097      1.00x      1.00x         -
     75000   sequential        -      0.1941      0.1938      0.1942      3.092      1.00x      1.00x         -
     75000     parallel        1      0.1938      0.1937      0.1961      3.095      1.00x      1.00x         -
     75000     parallel        2      0.1082      0.1076      0.1126      5.543      1.79x      1.79x         -
     75000     parallel        4      0.0609      0.0607      0.0620      9.849      3.18x      3.19x         -
     75000     parallel        8      0.0449      0.0444      0.0455     13.377      4.32x      4.33x         -
     75000     parallel       12      0.0419      0.0417      0.0544     14.319      4.62x      4.63x         -
     75000     parallel       16      0.0502      0.0500      0.0505     11.952      3.86x      3.87x         -
     75000     parallel       24      0.0537      0.0535      0.0596     11.174      3.61x      3.61x         -

    100000       scalar        -      0.2585      0.2585      0.2609      3.095      1.00x      1.00x         -
    100000   sequential        -      0.2583      0.2583      0.2587      3.097      1.00x      1.00x         -
    100000     parallel        1      0.2585      0.2585      0.2589      3.095      1.00x      1.00x         -
    100000     parallel        2      0.1407      0.1405      0.1426      5.687      1.84x      1.84x         -
    100000     parallel        4      0.0774      0.0772      0.0803     10.341      3.34x      3.34x         -
    100000     parallel        8      0.0531      0.0527      0.0537     15.067      4.87x      4.86x         -
    100000     parallel       12      0.0478      0.0475      0.0492     16.733      5.41x      5.40x         -
    100000     parallel       16      0.0590      0.0587      0.0609     13.559      4.38x      4.38x         -
    100000     parallel       24      0.0593      0.0592      0.0660     13.490      4.36x      4.36x         -

    200000       scalar        -      0.5167      0.5167      0.5208      3.096      1.00x      1.00x         -
    200000   sequential        -      0.5167      0.5167      0.5176      3.096      1.00x      1.00x         -
    200000     parallel        1      0.5168      0.5167      0.5187      3.096      1.00x      1.00x         -
    200000     parallel        2      0.2705      0.2703      0.2709      5.915      1.91x      1.91x         -
    200000     parallel        4      0.1426      0.1424      0.1476     11.217      3.62x      3.62x         -
    200000     parallel        8      0.0874      0.0857      0.0879     18.311      5.91x      5.91x         -
    200000     parallel       12      0.0699      0.0695      0.0707     22.904      7.40x      7.40x         -
    200000     parallel       16      0.0911      0.0909      0.0924     17.555      5.67x      5.67x         -
    200000     parallel       24      0.1019      0.0808      0.1029     15.707      5.07x      5.07x         -

    500000       scalar        -      1.2912      1.2911      1.2996      3.098      1.00x      1.00x         -
    500000   sequential        -      1.2913      1.2911      1.3060      3.098      1.00x      1.00x         -
    500000     parallel        1      1.2913      1.2912      1.2933      3.098      1.00x      1.00x         -
    500000     parallel        2      0.6617      0.6576      0.6715      6.045      1.95x      1.95x         -
    500000     parallel        4      0.3369      0.3365      0.3461     11.873      3.83x      3.83x         -
    500000     parallel        8      0.1875      0.1872      0.2016     21.330      6.89x      6.89x         -
    500000     parallel       12      0.1370      0.1365      0.1807     29.189      9.42x      9.42x         -
    500000     parallel       16      0.1882      0.1878      0.1950     21.252      6.86x      6.86x         -
    500000     parallel       24      0.1483      0.1469      0.1495     26.976      8.71x      8.71x         -

   1000000       scalar        -      2.5824      2.5820      2.5860      3.098      1.00x      1.00x         -
   1000000   sequential        -      2.5872      2.5823      2.6280      3.092      1.00x      1.00x         -
   1000000     parallel        1      2.5868      2.5843      2.6193      3.093      1.00x      1.00x         -
   1000000     parallel        2      1.3012      1.3005      1.3099      6.148      1.98x      1.99x         -
   1000000     parallel        4      0.6602      0.6597      0.6714     12.118      3.91x      3.92x         -
   1000000     parallel        8      0.3598      0.3591      0.3739     22.234      7.18x      7.19x         -
   1000000     parallel       12      0.2538      0.2525      0.2657     31.526     10.18x     10.20x         -
   1000000     parallel       16      0.3496      0.3488      0.3529     22.885      7.39x      7.40x         -
   1000000     parallel       24      0.2607      0.2603      0.4108     30.687      9.91x      9.92x         -

   2000000       scalar        -      5.1669      5.1664      5.1704      3.097      1.00x      1.00x         -
   2000000   sequential        -      5.1727      5.1693      5.2000      3.093      1.00x      1.00x         -
   2000000     parallel        1      5.1724      5.1698      5.3577      3.093      1.00x      1.00x         -
   2000000     parallel        2      2.6066      2.5975      2.6287      6.138      1.98x      1.98x         -
   2000000     parallel        4      1.3075      1.3063      1.3255     12.237      3.95x      3.96x         -
   2000000     parallel        8      0.7365      0.7261      0.7506     21.724      7.02x      7.02x         -
   2000000     parallel       12      0.8836      0.5001      0.8923     18.107      5.85x      5.85x         -
   2000000     parallel       16      0.6731      0.6718      0.6758     23.769      7.68x      7.68x         -
   2000000     parallel       24      0.4930      0.4829      0.5598     32.454     10.48x     10.49x         -

   5000000       scalar        -     12.9544     12.9471     13.0282      3.088      1.00x      1.00x      0.94
   5000000   sequential        -     13.0150     12.9431     13.8688      3.073      1.00x      1.00x      1.06
   5000000     parallel        1     12.9884     12.9463     13.1815      3.080      1.00x      1.00x      0.93
   5000000     parallel        2      6.5190      6.4924      6.6933      6.136      1.99x      2.00x         -
   5000000     parallel        4      3.2924      3.2546      3.3188     12.149      3.93x      3.95x         -
   5000000     parallel        8      2.1462      2.0939      2.2677     18.638      6.04x      6.06x         -
   5000000     parallel       12      1.3144      1.3018      1.3366     30.432      9.86x      9.90x         -
   5000000     parallel       16      1.6504      1.6413      1.7287     24.236      7.85x      7.89x         -
   5000000     parallel       24      1.3584      1.2553      1.4010     29.445      9.54x      9.58x         -

  10000000       scalar        -     25.9789     25.9476     26.1674      3.079      1.00x      1.00x      1.07
  10000000   sequential        -     25.9861     25.9240     26.0662      3.079      1.00x      1.00x      1.00
  10000000     parallel        1     25.9954     25.9385     26.0633      3.077      1.00x      1.00x      0.93
  10000000     parallel        2     13.0127     12.9847     13.0732      6.148      2.00x      2.00x      1.87
  10000000     parallel        4      6.5638      6.5325      6.8512     12.188      3.96x      3.96x         -
  10000000     parallel        8      4.1918      3.9881      4.2519     19.085      6.20x      6.20x         -
  10000000     parallel       12      2.8289      2.7974      3.3592     28.280      9.18x      9.19x         -
  10000000     parallel       16      3.2686      3.2612      5.7384     24.475      7.95x      7.95x         -
  10000000     parallel       24      2.6701      2.5518      4.2001     29.961      9.73x      9.73x         -

  33554432       scalar        -     88.0703     87.8768     88.3905      3.048      1.00x      0.99x      0.95
  33554432   sequential        -     87.2834     87.0947     87.4048      3.075      1.01x      1.00x      0.99
  33554432     parallel        1     88.0390     87.8106     89.3378      3.049      1.00x      0.99x      0.98
  33554432     parallel        2     44.2349     44.1426     44.9773      6.068      1.99x      1.97x      2.08
  33554432     parallel        4     22.2676     22.1787     23.9387     12.055      3.96x      3.92x      4.01
  33554432     parallel        8     12.4430     11.9508     13.1050     21.573      7.08x      7.01x      6.67
  33554432     parallel       12     11.8240     10.6109     14.5847     22.703      7.45x      7.38x     11.18
  33554432     parallel       16     11.4850     11.0223     11.6024     23.373      7.67x      7.60x     12.89
  33554432     parallel       24     11.2617     10.6498     15.2416     23.836      7.82x      7.75x     23.26


== sigmoid ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0032      0.0032      0.0032      2.500      1.00x      1.00x         -
      1000   sequential        -      0.0032      0.0032      0.0032      2.501      1.00x      1.00x         -
      1000     parallel        1      0.0032      0.0032      0.0033      2.476      0.99x      0.99x         -
      1000     parallel        2      0.0032      0.0032      0.0033      2.475      0.99x      0.99x         -
      1000     parallel        4      0.0032      0.0032      0.0033      2.476      0.99x      0.99x         -
      1000     parallel        8      0.0032      0.0032      0.0033      2.480      0.99x      0.99x         -
      1000     parallel       12      0.0032      0.0032      0.0033      2.476      0.99x      0.99x         -
      1000     parallel       16      0.0032      0.0032      0.0033      2.475      0.99x      0.99x         -
      1000     parallel       24      0.0032      0.0032      0.0033      2.479      0.99x      0.99x         -

     10000       scalar        -      0.0320      0.0320      0.0321      2.502      1.00x      1.00x         -
     10000   sequential        -      0.0320      0.0320      0.0321      2.502      1.00x      1.00x         -
     10000     parallel        1      0.0322      0.0322      0.0327      2.484      0.99x      0.99x         -
     10000     parallel        2      0.0323      0.0322      0.0327      2.480      0.99x      0.99x         -
     10000     parallel        4      0.0323      0.0322      0.0329      2.480      0.99x      0.99x         -
     10000     parallel        8      0.0323      0.0322      0.0327      2.479      0.99x      0.99x         -
     10000     parallel       12      0.0323      0.0322      0.0327      2.480      0.99x      0.99x         -
     10000     parallel       16      0.0323      0.0322      0.0327      2.480      0.99x      0.99x         -
     10000     parallel       24      0.0323      0.0322      0.0328      2.480      0.99x      0.99x         -

     32768       scalar        -      0.1048      0.1047      0.1049      2.501      1.00x      1.00x         -
     32768   sequential        -      0.1048      0.1047      0.1054      2.502      1.00x      1.00x         -
     32768     parallel        1      0.1057      0.1055      0.1065      2.480      0.99x      0.99x         -
     32768     parallel        2      0.1058      0.1056      0.1072      2.479      0.99x      0.99x         -
     32768     parallel        4      0.1058      0.1057      0.1067      2.478      0.99x      0.99x         -
     32768     parallel        8      0.1058      0.1058      0.1074      2.478      0.99x      0.99x         -
     32768     parallel       12      0.1058      0.1055      0.1074      2.477      0.99x      0.99x         -
     32768     parallel       16      0.1058      0.1056      0.1073      2.478      0.99x      0.99x         -
     32768     parallel       24      0.1059      0.1056      0.1076      2.476      0.99x      0.99x         -

     32769       scalar        -      0.1050      0.1048      0.1061      2.497      1.00x      1.00x         -
     32769   sequential        -      0.1049      0.1048      0.1052      2.500      1.00x      1.00x         -
     32769     parallel        1      0.1058      0.1056      0.1063      2.478      0.99x      0.99x         -
     32769     parallel        2      0.0630      0.0629      0.0663      4.163      1.67x      1.67x         -
     32769     parallel        4      0.0388      0.0387      0.0420      6.757      2.71x      2.70x         -
     32769     parallel        8      0.0335      0.0332      0.0362      7.824      3.13x      3.13x         -
     32769     parallel       12      0.0344      0.0340      0.0359      7.630      3.06x      3.05x         -
     32769     parallel       16      0.0383      0.0380      0.0384      6.843      2.74x      2.74x         -
     32769     parallel       24      0.0466      0.0464      0.0473      5.624      2.25x      2.25x         -

     40000       scalar        -      0.1279      0.1278      0.1282      2.502      1.00x      1.00x         -
     40000   sequential        -      0.1279      0.1278      0.1299      2.502      1.00x      1.00x         -
     40000     parallel        1      0.1290      0.1289      0.1304      2.480      0.99x      0.99x         -
     40000     parallel        2      0.0746      0.0745      0.0767      4.291      1.71x      1.71x         -
     40000     parallel        4      0.0443      0.0442      0.0448      7.224      2.89x      2.89x         -
     40000     parallel        8      0.0368      0.0366      0.0381      8.697      3.48x      3.48x         -
     40000     parallel       12      0.0363      0.0361      0.0368      8.820      3.52x      3.52x         -
     40000     parallel       16      0.0413      0.0410      0.0417      7.740      3.09x      3.09x         -
     40000     parallel       24      0.0482      0.0480      0.0495      6.636      2.65x      2.65x         -

     50000       scalar        -      0.1600      0.1599      0.1619      2.499      1.00x      1.00x         -
     50000   sequential        -      0.1599      0.1598      0.1617      2.501      1.00x      1.00x         -
     50000     parallel        1      0.1613      0.1610      0.1615      2.479      0.99x      0.99x         -
     50000     parallel        2      0.0908      0.0908      0.0916      4.403      1.76x      1.76x         -
     50000     parallel        4      0.0525      0.0524      0.0564      7.618      3.05x      3.05x         -
     50000     parallel        8      0.0424      0.0403      0.0442      9.439      3.78x      3.77x         -
     50000     parallel       12      0.0472      0.0462      0.0502      8.477      3.39x      3.39x         -
     50000     parallel       16      0.0447      0.0442      0.0450      8.941      3.58x      3.57x         -
     50000     parallel       24      0.0528      0.0523      0.0547      7.581      3.03x      3.03x         -

     75000       scalar        -      0.2401      0.2400      0.2414      2.499      1.00x      1.00x         -
     75000   sequential        -      0.2404      0.2401      0.2414      2.496      1.00x      1.00x         -
     75000     parallel        1      0.2421      0.2417      0.2431      2.479      0.99x      0.99x         -
     75000     parallel        2      0.1310      0.1310      0.1313      4.580      1.83x      1.84x         -
     75000     parallel        4      0.0733      0.0730      0.0747      8.182      3.27x      3.28x         -
     75000     parallel        8      0.0511      0.0507      0.0521     11.749      4.70x      4.71x         -
     75000     parallel       12      0.0535      0.0453      0.0557     11.219      4.49x      4.49x         -
     75000     parallel       16      0.0546      0.0541      0.0562     10.994      4.40x      4.40x         -
     75000     parallel       24      0.0572      0.0569      0.0579     10.493      4.20x      4.20x         -

    100000       scalar        -      0.3204      0.3202      0.3210      2.497      1.00x      1.00x         -
    100000   sequential        -      0.3204      0.3200      0.3235      2.497      1.00x      1.00x         -
    100000     parallel        1      0.3227      0.3226      0.3236      2.479      0.99x      0.99x         -
    100000     parallel        2      0.1714      0.1713      0.1715      4.669      1.87x      1.87x         -
    100000     parallel        4      0.0928      0.0926      0.0938      8.624      3.45x      3.45x         -
    100000     parallel        8      0.0605      0.0600      0.0615     13.229      5.30x      5.30x         -
    100000     parallel       12      0.0525      0.0521      0.0542     15.253      6.11x      6.11x         -
    100000     parallel       16      0.0642      0.0637      0.0651     12.453      4.99x      4.99x         -
    100000     parallel       24      0.0634      0.0628      0.0643     12.616      5.05x      5.05x         -

    200000       scalar        -      0.6407      0.6403      0.6486      2.497      1.00x      1.00x         -
    200000   sequential        -      0.6404      0.6404      0.6417      2.499      1.00x      1.00x         -
    200000     parallel        1      0.6462      0.6456      0.6480      2.476      0.99x      0.99x         -
    200000     parallel        2      0.3325      0.3324      0.3367      4.813      1.93x      1.93x         -
    200000     parallel        4      0.1737      0.1735      0.1758      9.213      3.69x      3.69x         -
    200000     parallel        8      0.1018      0.1009      0.1026     15.721      6.30x      6.29x         -
    200000     parallel       12      0.0795      0.0790      0.1009     20.115      8.05x      8.05x         -
    200000     parallel       16      0.1047      0.1039      0.1079     15.288      6.12x      6.12x         -
    200000     parallel       24      0.1124      0.1102      0.1168     14.232      5.70x      5.70x         -

    500000       scalar        -      1.6012      1.5999      1.6601      2.498      1.00x      1.00x         -
    500000   sequential        -      1.6013      1.6000      1.6122      2.498      1.00x      1.00x         -
    500000     parallel        1      1.6165      1.6140      1.6399      2.474      0.99x      0.99x         -
    500000     parallel        2      0.8132      0.8125      0.8174      4.919      1.97x      1.97x         -
    500000     parallel        4      0.4153      0.4150      0.4194      9.632      3.86x      3.86x         -
    500000     parallel        8      0.2243      0.2229      0.2256     17.835      7.14x      7.14x         -
    500000     parallel       12      0.1616      0.1606      0.2765     24.758      9.91x      9.91x         -
    500000     parallel       16      0.2172      0.2152      0.2184     18.420      7.37x      7.37x         -
    500000     parallel       24      0.1658      0.1653      0.1739     24.126      9.66x      9.66x         -

   1000000       scalar        -      3.2030      3.2000      3.2342      2.498      1.00x      1.00x         -
   1000000   sequential        -      3.2062      3.2021      3.2111      2.495      1.00x      1.00x         -
   1000000     parallel        1      3.2323      3.2283      3.3182      2.475      0.99x      0.99x         -
   1000000     parallel        2      1.6140      1.6096      1.6186      4.956      1.98x      1.99x         -
   1000000     parallel        4      0.8166      0.8159      0.8593      9.797      3.92x      3.93x         -
   1000000     parallel        8      0.4317      0.4293      0.4391     18.530      7.42x      7.43x         -
   1000000     parallel       12      0.3006      0.2997      0.3106     26.610     10.65x     10.66x         -
   1000000     parallel       16      0.4083      0.4078      0.4090     19.592      7.84x      7.85x         -
   1000000     parallel       24      0.2970      0.2965      0.3082     26.937     10.78x     10.80x         -

   2000000       scalar        -      6.4079      6.4017      6.4403      2.497      1.00x      1.00x         -
   2000000   sequential        -      6.4080      6.4016      6.4369      2.497      1.00x      1.00x         -
   2000000     parallel        1      6.4680      6.4595      6.5038      2.474      0.99x      0.99x         -
   2000000     parallel        2      3.2228      3.2198      3.2497      4.965      1.99x      1.99x         -
   2000000     parallel        4      1.6200      1.6185      1.6361      9.876      3.96x      3.96x         -
   2000000     parallel        8      0.8535      0.8499      0.8866     18.746      7.51x      7.51x         -
   2000000     parallel       12      0.6036      0.6015      1.0215     26.510     10.62x     10.62x         -
   2000000     parallel       16      0.8233      0.8224      0.8381     19.434      7.78x      7.78x         -
   2000000     parallel       24      0.6641      0.5635      0.8932     24.091      9.65x      9.65x         -

   5000000       scalar        -     16.0682     16.0380     16.1149      2.489      1.00x      1.00x      1.19
   5000000   sequential        -     16.0749     16.0408     16.1286      2.488      1.00x      1.00x      0.97
   5000000     parallel        1     16.2135     16.1697     16.3188      2.467      0.99x      0.99x      0.96
   5000000     parallel        2      8.0485      8.0358      8.0917      4.970      2.00x      2.00x         -
   5000000     parallel        4      4.0387      4.0338      4.2582      9.904      3.98x      3.98x         -
   5000000     parallel        8      2.3200      2.2965      2.3477     17.241      6.93x      6.93x         -
   5000000     parallel       12      1.4754      1.4626      2.0733     27.110     10.89x     10.89x         -
   5000000     parallel       16      1.9432      1.9394      1.9531     20.585      8.27x      8.27x         -
   5000000     parallel       24      1.9807      1.7227      2.0067     20.194      8.11x      8.12x         -

  10000000       scalar        -     32.2053     32.0988     32.2771      2.484      1.00x      1.00x      0.97
  10000000   sequential        -     32.1918     32.1327     32.3534      2.485      1.00x      1.00x      0.97
  10000000     parallel        1     32.4704     32.4109     32.6229      2.464      0.99x      0.99x      1.02
  10000000     parallel        2     16.1548     16.1117     17.8151      4.952      1.99x      1.99x      1.91
  10000000     parallel        4      9.2205      9.1966      9.2360      8.676      3.49x      3.49x         -
  10000000     parallel        8      4.5114      4.2850      4.6018     17.733      7.14x      7.14x         -
  10000000     parallel       12      3.0756      2.9570      4.8522     26.011     10.47x     10.47x         -
  10000000     parallel       16      4.0510      3.9347      4.0701     19.748      7.95x      7.95x         -
  10000000     parallel       24      2.8698      2.7110      3.3529     27.877     11.22x     11.22x         -

  33554432       scalar        -    110.8812    110.4160    112.5407      2.421      1.00x      0.98x      0.98
  33554432   sequential        -    108.1140    107.9232    109.3090      2.483      1.03x      1.00x      0.99
  33554432     parallel        1    112.1766    111.5549    112.5229      2.393      0.99x      0.96x      1.02
  33554432     parallel        2     55.6033     55.3477     56.7292      4.828      1.99x      1.94x      1.96
  33554432     parallel        4     27.9079     27.8244     34.7897      9.619      3.97x      3.87x      4.04
  33554432     parallel        8     14.5362     14.3858     14.6866     18.467      7.63x      7.44x      8.60
  33554432     parallel       12     11.2242     10.7241     17.4402     23.916      9.88x      9.63x     11.56
  33554432     parallel       16     12.9344     12.9093     13.1478     20.754      8.57x      8.36x     13.13
  33554432     parallel       24     10.9371     10.6006     16.5691     24.544     10.14x      9.89x     15.01


== axpy ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0017      0.0017      0.0017      7.251      1.00x      1.00x         -
      1000   sequential        -      0.0017      0.0017      0.0017      7.254      1.00x      1.00x         -
      1000     parallel        1      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -
      1000     parallel        2      0.0017      0.0017      0.0017      7.238      1.00x      1.00x         -
      1000     parallel        4      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -
      1000     parallel        8      0.0017      0.0017      0.0017      7.238      1.00x      1.00x         -
      1000     parallel       12      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -
      1000     parallel       16      0.0017      0.0017      0.0017      7.238      1.00x      1.00x         -
      1000     parallel       24      0.0017      0.0017      0.0017      7.237      1.00x      1.00x         -

     10000       scalar        -      0.0165      0.0165      0.0167      7.287      1.00x      1.00x         -
     10000   sequential        -      0.0165      0.0165      0.0165      7.287      1.00x      1.00x         -
     10000     parallel        1      0.0165      0.0165      0.0165      7.284      1.00x      1.00x         -
     10000     parallel        2      0.0165      0.0165      0.0166      7.285      1.00x      1.00x         -
     10000     parallel        4      0.0165      0.0165      0.0166      7.286      1.00x      1.00x         -
     10000     parallel        8      0.0165      0.0165      0.0167      7.273      1.00x      1.00x         -
     10000     parallel       12      0.0165      0.0165      0.0165      7.284      1.00x      1.00x         -
     10000     parallel       16      0.0165      0.0165      0.0165      7.285      1.00x      1.00x         -
     10000     parallel       24      0.0165      0.0165      0.0167      7.282      1.00x      1.00x         -

     32768       scalar        -      0.0540      0.0539      0.0541      7.286      1.00x      1.00x         -
     32768   sequential        -      0.0540      0.0539      0.0543      7.286      1.00x      1.00x         -
     32768     parallel        1      0.0540      0.0540      0.0544      7.285      1.00x      1.00x         -
     32768     parallel        2      0.0540      0.0539      0.0543      7.285      1.00x      1.00x         -
     32768     parallel        4      0.0540      0.0540      0.0544      7.284      1.00x      1.00x         -
     32768     parallel        8      0.0540      0.0539      0.0551      7.284      1.00x      1.00x         -
     32768     parallel       12      0.0542      0.0539      0.0546      7.260      1.00x      1.00x         -
     32768     parallel       16      0.0540      0.0539      0.0543      7.286      1.00x      1.00x         -
     32768     parallel       24      0.0540      0.0539      0.0545      7.286      1.00x      1.00x         -

     32769       scalar        -      0.0540      0.0539      0.0547      7.287      1.00x      1.00x         -
     32769   sequential        -      0.0540      0.0539      0.0545      7.288      1.00x      1.00x         -
     32769     parallel        1      0.0540      0.0539      0.0545      7.285      1.00x      1.00x         -
     32769     parallel        2      0.0365      0.0363      0.0374     10.786      1.48x      1.48x         -
     32769     parallel        4      0.0254      0.0252      0.0261     15.496      2.13x      2.13x         -
     32769     parallel        8      0.0270      0.0269      0.0274     14.576      2.00x      2.00x         -
     32769     parallel       12      0.0305      0.0304      0.0309     12.872      1.77x      1.77x         -
     32769     parallel       16      0.0322      0.0320      0.0326     12.203      1.67x      1.67x         -
     32769     parallel       24      0.0426      0.0418      0.0442      9.238      1.27x      1.27x         -

     40000       scalar        -      0.0659      0.0659      0.0663      7.284      1.00x      1.00x         -
     40000   sequential        -      0.0659      0.0659      0.0670      7.281      1.00x      1.00x         -
     40000     parallel        1      0.0660      0.0660      0.0661      7.275      1.00x      1.00x         -
     40000     parallel        2      0.0424      0.0423      0.0426     11.323      1.55x      1.56x         -
     40000     parallel        4      0.0282      0.0282      0.0285     17.002      2.33x      2.34x         -
     40000     parallel        8      0.0288      0.0284      0.0291     16.684      2.29x      2.29x         -
     40000     parallel       12      0.0312      0.0309      0.0317     15.407      2.12x      2.12x         -
     40000     parallel       16      0.0331      0.0330      0.0339     14.493      1.99x      1.99x         -
     40000     parallel       24      0.0429      0.0426      0.0441     11.197      1.54x      1.54x         -

     50000       scalar        -      0.0827      0.0825      0.0854      7.251      1.00x      1.00x         -
     50000   sequential        -      0.0825      0.0825      0.0830      7.270      1.00x      1.00x         -
     50000     parallel        1      0.0826      0.0825      0.0829      7.264      1.00x      1.00x         -
     50000     parallel        2      0.0507      0.0506      0.0512     11.830      1.63x      1.63x         -
     50000     parallel        4      0.0320      0.0319      0.0327     18.728      2.58x      2.58x         -
     50000     parallel        8      0.0306      0.0305      0.0310     19.630      2.71x      2.70x         -
     50000     parallel       12      0.0329      0.0327      0.0331     18.218      2.51x      2.51x         -
     50000     parallel       16      0.0342      0.0339      0.0350     17.550      2.42x      2.41x         -
     50000     parallel       24      0.0443      0.0437      0.0446     13.555      1.87x      1.86x         -

     75000       scalar        -      0.1237      0.1237      0.1242      7.276      1.00x      1.00x         -
     75000   sequential        -      0.1240      0.1239      0.1241      7.258      1.00x      1.00x         -
     75000     parallel        1      0.1238      0.1238      0.1240      7.267      1.00x      1.00x         -
     75000     parallel        2      0.0711      0.0711      0.0716     12.652      1.74x      1.74x         -
     75000     parallel        4      0.0433      0.0422      0.0441     20.786      2.86x      2.86x         -
     75000     parallel        8      0.0358      0.0355      0.0366     25.134      3.45x      3.46x         -
     75000     parallel       12      0.0360      0.0359      0.0376     25.023      3.44x      3.45x         -
     75000     parallel       16      0.0374      0.0371      0.0383     24.084      3.31x      3.32x         -
     75000     parallel       24      0.0472      0.0466      0.0501     19.072      2.62x      2.63x         -

    100000       scalar        -      0.1651      0.1650      0.1653      7.268      1.00x      1.00x         -
    100000   sequential        -      0.1649      0.1648      0.1653      7.279      1.00x      1.00x         -
    100000     parallel        1      0.1651      0.1650      0.1673      7.270      1.00x      1.00x         -
    100000     parallel        2      0.0907      0.0907      0.0911     13.224      1.82x      1.82x         -
    100000     parallel        4      0.0522      0.0521      0.0532     22.971      3.16x      3.16x         -
    100000     parallel        8      0.0402      0.0402      0.0416     29.826      4.10x      4.10x         -
    100000     parallel       12      0.0391      0.0390      0.0396     30.694      4.22x      4.22x         -
    100000     parallel       16      0.0401      0.0400      0.0405     29.916      4.12x      4.11x         -
    100000     parallel       24      0.0491      0.0482      0.0504     24.459      3.37x      3.36x         -

    200000       scalar        -      0.3301      0.3298      0.3314      7.271      1.00x      1.00x         -
    200000   sequential        -      0.3302      0.3298      0.3305      7.269      1.00x      1.00x         -
    200000     parallel        1      0.3303      0.3302      0.3348      7.265      1.00x      1.00x         -
    200000     parallel        2      0.1702      0.1701      0.1743     14.100      1.94x      1.94x         -
    200000     parallel        4      0.0935      0.0927      0.0979     25.675      3.53x      3.53x         -
    200000     parallel        8      0.0656      0.0621      0.0665     36.592      5.03x      5.03x         -
    200000     parallel       12      0.0592      0.0558      0.0646     40.520      5.57x      5.57x         -
    200000     parallel       16      0.0558      0.0528      0.0571     43.026      5.92x      5.92x         -
    200000     parallel       24      0.0585      0.0580      0.0588     41.040      5.64x      5.65x         -

    500000       scalar        -      0.8248      0.8243      0.8287      7.274      1.00x      1.00x         -
    500000   sequential        -      0.8257      0.8256      0.8351      7.267      1.00x      1.00x         -
    500000     parallel        1      0.8259      0.8254      0.8295      7.264      1.00x      1.00x         -
    500000     parallel        2      0.4063      0.4062      0.4068     14.766      2.03x      2.03x         -
    500000     parallel        4      0.2115      0.2114      0.2140     28.365      3.90x      3.90x         -
    500000     parallel        8      0.1222      0.1219      0.1257     49.117      6.75x      6.76x         -
    500000     parallel       12      0.0939      0.0933      0.0972     63.877      8.78x      8.79x         -
    500000     parallel       16      0.0932      0.0927      0.0945     64.393      8.85x      8.86x         -
    500000     parallel       24      0.0876      0.0865      0.0886     68.518      9.42x      9.43x         -

   1000000       scalar        -      1.6498      1.6486      1.6554      7.274      1.00x      1.00x         -
   1000000   sequential        -      1.6511      1.6500      1.6531      7.268      1.00x      1.00x         -
   1000000     parallel        1      1.6516      1.6502      1.6658      7.266      1.00x      1.00x         -
   1000000     parallel        2      0.8017      0.7999      0.8100     14.968      2.06x      2.06x         -
   1000000     parallel        4      0.4092      0.4088      0.4183     29.325      4.03x      4.03x         -
   1000000     parallel        8      0.2266      0.2260      0.2305     52.954      7.28x      7.29x         -
   1000000     parallel       12      0.1638      0.1631      0.1837     73.260     10.07x     10.08x         -
   1000000     parallel       16      0.1604      0.1600      0.1609     74.816     10.29x     10.29x         -
   1000000     parallel       24      0.1846      0.1756      0.1988     64.988      8.93x      8.94x         -

   2000000       scalar        -      3.3101      3.3008      3.3349      7.251      1.00x      1.00x         -
   2000000   sequential        -      3.3044      3.2994      3.3276      7.263      1.00x      1.00x         -
   2000000     parallel        1      3.3033      3.3008      3.3160      7.265      1.00x      1.00x         -
   2000000     parallel        2      1.5899      1.5836      1.8005     15.095      2.08x      2.08x         -
   2000000     parallel        4      0.8042      0.8039      0.8138     29.845      4.12x      4.11x         -
   2000000     parallel        8      0.4439      0.4437      0.4592     54.060      7.46x      7.44x         -
   2000000     parallel       12      0.3116      0.3112      0.3226     77.028     10.62x     10.61x         -
   2000000     parallel       16      0.2970      0.2963      0.2977     80.815     11.15x     11.13x         -
   2000000     parallel       24      0.2321      0.2288      0.2436    103.420     14.26x     14.24x         -

   5000000       scalar        -      8.3377      8.3260      8.3889      7.196      1.00x      1.00x         -
   5000000   sequential        -      8.3379      8.3259      8.3827      7.196      1.00x      1.00x         -
   5000000     parallel        1      8.3592      8.3388      8.3990      7.178      1.00x      1.00x         -
   5000000     parallel        2      4.0256      3.9917      4.1417     14.905      2.07x      2.07x         -
   5000000     parallel        4      2.0236      2.0188      2.2640     29.650      4.12x      4.12x         -
   5000000     parallel        8      1.3411      1.1498      1.5312     44.738      6.22x      6.22x         -
   5000000     parallel       12      1.2950      1.2779      1.3902     46.334      6.44x      6.44x         -
   5000000     parallel       16      1.2049      1.1476      1.2897     49.795      6.92x      6.92x         -
   5000000     parallel       24      1.2672      1.1875      1.3565     47.348      6.58x      6.58x         -

  10000000       scalar        -     16.7136     16.6695     16.7740      7.180      1.00x      1.00x      0.93
  10000000   sequential        -     16.7220     16.6692     16.8369      7.176      1.00x      1.00x      0.93
  10000000     parallel        1     16.7118     16.6696     16.8230      7.181      1.00x      1.00x      1.04
  10000000     parallel        2      8.1129      8.0351      8.2887     14.791      2.06x      2.06x         -
  10000000     parallel        4      4.1269      4.0903      4.4467     29.078      4.05x      4.05x         -
  10000000     parallel        8      3.2085      2.6045      3.8271     37.401      5.21x      5.21x         -
  10000000     parallel       12      3.1528      3.0163      3.3992     38.061      5.30x      5.30x         -
  10000000     parallel       16      3.4753      3.3522      3.6222     34.529      4.81x      4.81x         -
  10000000     parallel       24      3.4928      3.3997      3.5499     34.356      4.79x      4.79x         -

  33554432       scalar        -     56.4400     56.2990     56.5496      7.134      1.00x      1.01x      0.98
  33554432   sequential        -     57.2832     57.2350     58.2920      7.029      0.99x      1.00x      1.00
  33554432     parallel        1     56.5606     56.4254     56.6420      7.119      1.00x      1.01x      0.95
  33554432     parallel        2     27.2222     27.1738     28.0232     14.791      2.07x      2.10x      2.10
  33554432     parallel        4     15.5996     15.0784     17.1156     25.812      3.62x      3.67x      3.67
  33554432     parallel        8     13.5741     13.2795     14.4370     29.663      4.16x      4.22x      7.60
  33554432     parallel       12     13.8035     13.5144     14.1850     29.170      4.09x      4.15x     10.54
  33554432     parallel       16     14.1489     14.0112     14.7482     28.458      3.99x      4.05x     13.04
  33554432     parallel       24     14.1161     13.9006     15.9422     28.524      4.00x      4.06x     25.64


== bias_add ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0004      0.0004      0.0004     21.708      1.00x      1.00x         -
      1000   sequential        -      0.0004      0.0004      0.0004     21.653      1.00x      1.00x         -
      1000     parallel        1      0.0006      0.0006      0.0006     13.652      0.63x      0.63x         -
      1000     parallel        2      0.0006      0.0006      0.0006     13.652      0.63x      0.63x         -
      1000     parallel        4      0.0006      0.0006      0.0006     13.657      0.63x      0.63x         -
      1000     parallel        8      0.0006      0.0006      0.0006     13.657      0.63x      0.63x         -
      1000     parallel       12      0.0006      0.0006      0.0006     13.655      0.63x      0.63x         -
      1000     parallel       16      0.0006      0.0006      0.0006     13.655      0.63x      0.63x         -
      1000     parallel       24      0.0006      0.0006      0.0007     13.642      0.63x      0.63x         -

     10000       scalar        -      0.0016      0.0016      0.0016     49.322      1.00x      1.00x         -
     10000   sequential        -      0.0016      0.0016      0.0016     49.331      1.00x      1.00x         -
     10000     parallel        1      0.0027      0.0027      0.0027     29.634      0.60x      0.60x         -
     10000     parallel        2      0.0027      0.0027      0.0027     29.650      0.60x      0.60x         -
     10000     parallel        4      0.0027      0.0027      0.0028     29.643      0.60x      0.60x         -
     10000     parallel        8      0.0027      0.0027      0.0027     29.635      0.60x      0.60x         -
     10000     parallel       12      0.0027      0.0027      0.0027     29.639      0.60x      0.60x         -
     10000     parallel       16      0.0027      0.0027      0.0027     29.648      0.60x      0.60x         -
     10000     parallel       24      0.0027      0.0027      0.0027     29.647      0.60x      0.60x         -

     32768       scalar        -      0.0031      0.0031      0.0031     84.896      1.00x      0.99x         -
     32768   sequential        -      0.0031      0.0031      0.0031     85.995      1.01x      1.00x         -
     32768     parallel        1      0.0034      0.0034      0.0034     78.046      0.92x      0.91x         -
     32768     parallel        2      0.0034      0.0034      0.0034     78.114      0.92x      0.91x         -
     32768     parallel        4      0.0034      0.0034      0.0034     78.241      0.92x      0.91x         -
     32768     parallel        8      0.0034      0.0034      0.0034     78.213      0.92x      0.91x         -
     32768     parallel       12      0.0034      0.0034      0.0034     78.288      0.92x      0.91x         -
     32768     parallel       16      0.0034      0.0034      0.0034     78.211      0.92x      0.91x         -
     32768     parallel       24      0.0034      0.0034      0.0034     78.161      0.92x      0.91x         -

     32769       scalar        -      0.0024      0.0024      0.0024    110.191      1.00x      1.00x         -
     32769   sequential        -      0.0024      0.0024      0.0024    110.232      1.00x      1.00x         -
     32769     parallel        1      0.0024      0.0024      0.0024    110.056      1.00x      1.00x         -
     32769     parallel        2      0.0160      0.0157      0.0163     16.404      0.15x      0.15x         -
     32769     parallel        4      0.0153      0.0150      0.0157     17.172      0.16x      0.16x         -
     32769     parallel        8      0.0223      0.0220      0.0230     11.760      0.11x      0.11x         -
     32769     parallel       12      0.0278      0.0277      0.0283      9.437      0.09x      0.09x         -
     32769     parallel       16      0.0303      0.0300      0.0309      8.650      0.08x      0.08x         -
     32769     parallel       24      0.0370      0.0362      0.0373      7.080      0.06x      0.06x         -

     40000       scalar        -      0.0039      0.0039      0.0039     82.732      1.00x      1.00x         -
     40000   sequential        -      0.0039      0.0039      0.0039     82.360      1.00x      1.00x         -
     40000     parallel        1      0.0049      0.0049      0.0049     65.392      0.79x      0.79x         -
     40000     parallel        2      0.0258      0.0253      0.0268     12.398      0.15x      0.15x         -
     40000     parallel        4      0.0201      0.0196      0.0206     15.965      0.19x      0.19x         -
     40000     parallel        8      0.0261      0.0250      0.0265     12.269      0.15x      0.15x         -
     40000     parallel       12      0.0295      0.0287      0.0303     10.840      0.13x      0.13x         -
     40000     parallel       16      0.0304      0.0301      0.0307     10.520      0.13x      0.13x         -
     40000     parallel       24      0.0391      0.0387      0.0400      8.182      0.10x      0.10x         -

     50000       scalar        -      0.0081      0.0081      0.0081     49.448      1.00x      1.00x         -
     50000   sequential        -      0.0081      0.0081      0.0082     49.481      1.00x      1.00x         -
     50000     parallel        1      0.0135      0.0135      0.0142     29.682      0.60x      0.60x         -
     50000     parallel        2      0.0293      0.0289      0.0305     13.663      0.28x      0.28x         -
     50000     parallel        4      0.0218      0.0215      0.0225     18.379      0.37x      0.37x         -
     50000     parallel        8      0.0249      0.0246      0.0259     16.069      0.32x      0.32x         -
     50000     parallel       12      0.0307      0.0300      0.0314     13.041      0.26x      0.26x         -
     50000     parallel       16      0.0312      0.0310      0.0329     12.837      0.26x      0.26x         -
     50000     parallel       24      0.0414      0.0406      0.0439      9.665      0.20x      0.20x         -

     75000       scalar        -      0.0272      0.0271      0.0274     22.057      1.00x      1.00x         -
     75000   sequential        -      0.0272      0.0271      0.0275     22.084      1.00x      1.00x         -
     75000     parallel        1      0.0433      0.0432      0.0435     13.850      0.63x      0.63x         -
     75000     parallel        2      0.0383      0.0381      0.0394     15.667      0.71x      0.71x         -
     75000     parallel        4      0.0263      0.0261      0.0267     22.809      1.03x      1.03x         -
     75000     parallel        8      0.0274      0.0272      0.0277     21.889      0.99x      0.99x         -
     75000     parallel       12      0.0326      0.0325      0.0335     18.391      0.83x      0.83x         -
     75000     parallel       16      0.0334      0.0332      0.0340     17.984      0.82x      0.81x         -
     75000     parallel       24      0.0433      0.0426      0.0443     13.856      0.63x      0.63x         -

    100000       scalar        -      0.0120      0.0120      0.0121     66.547      1.00x      0.99x         -
    100000   sequential        -      0.0120      0.0119      0.0120     66.921      1.01x      1.00x         -
    100000     parallel        1      0.0173      0.0173      0.0173     46.250      0.69x      0.69x         -
    100000     parallel        2      0.0482      0.0481      0.0484     16.601      0.25x      0.25x         -
    100000     parallel        4      0.0311      0.0309      0.0315     25.722      0.39x      0.38x         -
    100000     parallel        8      0.0318      0.0306      0.0327     25.190      0.38x      0.38x         -
    100000     parallel       12      0.0329      0.0313      0.0336     24.284      0.36x      0.36x         -
    100000     parallel       16      0.0364      0.0354      0.0384     21.958      0.33x      0.33x         -
    100000     parallel       24      0.0461      0.0457      0.0479     17.350      0.26x      0.26x         -

    200000       scalar        -      0.0208      0.0208      0.0209     76.788      1.00x      1.00x         -
    200000   sequential        -      0.0209      0.0208      0.0210     76.742      1.00x      1.00x         -
    200000     parallel        1      0.0252      0.0251      0.0255     63.522      0.83x      0.83x         -
    200000     parallel        2      0.0866      0.0865      0.0874     18.474      0.24x      0.24x         -
    200000     parallel        4      0.0508      0.0505      0.0529     31.520      0.41x      0.41x         -
    200000     parallel        8      0.0407      0.0393      0.0422     39.282      0.51x      0.51x         -
    200000     parallel       12      0.0438      0.0433      0.0443     36.556      0.48x      0.48x         -
    200000     parallel       16      0.0440      0.0431      0.0467     36.397      0.47x      0.47x         -
    200000     parallel       24      0.0540      0.0525      0.0559     29.610      0.39x      0.39x         -

    500000       scalar        -      0.0594      0.0593      0.0597     67.339      1.00x      1.00x         -
    500000   sequential        -      0.0596      0.0595      0.0598     67.134      1.00x      1.00x         -
    500000     parallel        1      0.0863      0.0862      0.0873     46.335      0.69x      0.69x         -
    500000     parallel        2      0.1976      0.1958      0.2025     20.247      0.30x      0.30x         -
    500000     parallel        4      0.1083      0.1079      0.1097     36.930      0.55x      0.55x         -
    500000     parallel        8      0.0681      0.0679      0.0699     58.761      0.87x      0.88x         -
    500000     parallel       12      0.0675      0.0630      0.0728     59.220      0.88x      0.88x         -
    500000     parallel       16      0.0687      0.0685      0.0691     58.221      0.86x      0.87x         -
    500000     parallel       24      0.0739      0.0730      0.0758     54.120      0.80x      0.81x         -

   1000000       scalar        -      0.1042      0.1041      0.1082     76.780      1.00x      1.02x         -
   1000000   sequential        -      0.1064      0.1057      0.1113     75.226      0.98x      1.00x         -
   1000000     parallel        1      0.1269      0.1249      0.1299     63.031      0.82x      0.84x         -
   1000000     parallel        2      0.3840      0.3829      0.3858     20.833      0.27x      0.28x         -
   1000000     parallel        4      0.2011      0.2007      0.2111     39.775      0.52x      0.53x         -
   1000000     parallel        8      0.1231      0.1213      0.1268     64.980      0.85x      0.86x         -
   1000000     parallel       12      0.1070      0.0994      0.1173     74.777      0.97x      0.99x         -
   1000000     parallel       16      0.1173      0.1107      0.1221     68.207      0.89x      0.91x         -
   1000000     parallel       24      0.1154      0.1137      0.1217     69.334      0.90x      0.92x         -

   2000000       scalar        -      0.2042      0.2038      0.2188     78.343      1.00x      1.02x         -
   2000000   sequential        -      0.2083      0.2076      0.2205     76.831      0.98x      1.00x         -
   2000000     parallel        1      0.2394      0.2347      0.2465     66.846      0.85x      0.87x         -
   2000000     parallel        2      0.7583      0.7555      0.7716     21.100      0.27x      0.27x         -
   2000000     parallel        4      0.3900      0.3883      0.3962     41.023      0.52x      0.53x         -
   2000000     parallel        8      0.2143      0.2123      0.2275     74.664      0.95x      0.97x         -
   2000000     parallel       12      0.2209      0.1956      0.2515     72.425      0.92x      0.94x         -
   2000000     parallel       16      0.1977      0.1955      0.2018     80.936      1.03x      1.05x         -
   2000000     parallel       24      0.2095      0.2035      0.2359     76.361      0.97x      0.99x         -

   5000000       scalar        -      1.0483      1.0160      1.0674     38.157      1.00x      1.12x         -
   5000000   sequential        -      1.1738      1.0871      1.2255     34.078      0.89x      1.00x         -
   5000000     parallel        1      1.1264      1.1052      1.1885     35.510      0.93x      1.04x         -
   5000000     parallel        2      1.9553      1.9479      1.9665     20.457      0.54x      0.60x         -
   5000000     parallel        4      1.0284      1.0038      1.2440     38.897      1.02x      1.14x         -
   5000000     parallel        8      0.5399      0.5305      0.5613     74.092      1.94x      2.17x         -
   5000000     parallel       12      0.4587      0.4271      0.5287     87.199      2.29x      2.56x         -
   5000000     parallel       16      0.5445      0.5165      0.6199     73.466      1.93x      2.16x         -
   5000000     parallel       24      0.5829      0.4995      0.6349     68.620      1.80x      2.01x         -

  10000000       scalar        -      2.6272      2.5319      2.7141     30.451      1.00x      1.06x         -
  10000000   sequential        -      2.7857      2.6413      3.0938     28.718      0.94x      1.00x         -
  10000000     parallel        1      2.6634      2.5638      2.8617     30.037      0.99x      1.05x         -
  10000000     parallel        2      3.9913      3.9750      4.0488     20.044      0.66x      0.70x         -
  10000000     parallel        4      2.2268      2.1454      2.4950     35.926      1.18x      1.25x         -
  10000000     parallel        8      1.5021      1.4280      1.6673     53.259      1.75x      1.85x         -
  10000000     parallel       12      2.2167      1.8875      2.3793     36.090      1.19x      1.26x         -
  10000000     parallel       16      2.2399      2.0920      2.3245     35.716      1.17x      1.24x         -
  10000000     parallel       24      2.5407      2.2046      2.7423     31.488      1.03x      1.10x         -

  33554432       scalar        -     11.4342     11.3442     11.8661     23.477      1.00x      0.97x      1.21
  33554432   sequential        -     11.1064     10.8649     11.7641     24.170      1.03x      1.00x      0.93
  33554432     parallel        1     12.1566     11.4188     12.7981     22.082      0.94x      0.91x      1.00
  33554432     parallel        2     16.5055     16.2847     16.6230     16.263      0.69x      0.67x      2.11
  33554432     parallel        4     10.5719     10.3383     11.3644     25.392      1.08x      1.05x         -
  33554432     parallel        8      9.8874      9.4513     11.0283     27.149      1.16x      1.12x         -
  33554432     parallel       12      9.8385      9.5413      9.9528     27.284      1.16x      1.13x         -
  33554432     parallel       16     10.6422     10.0921     11.3890     25.224      1.07x      1.04x         -
  33554432     parallel       24     10.1988      9.9272     10.5574     26.320      1.12x      1.09x         -


== dot ==
         n         impl  threads   median ms      min ms      max ms       GB/s  vs scalar     vs seq  cpu/wall
      1000       scalar        -      0.0009      0.0009      0.0009      9.357      1.00x      1.00x         -
      1000   sequential        -      0.0009      0.0009      0.0009      9.353      1.00x      1.00x         -
      1000     parallel        1      0.0009      0.0009      0.0009      9.335      1.00x      1.00x         -
      1000     parallel        2      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -
      1000     parallel        4      0.0009      0.0009      0.0009      9.335      1.00x      1.00x         -
      1000     parallel        8      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -
      1000     parallel       12      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -
      1000     parallel       16      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -
      1000     parallel       24      0.0009      0.0009      0.0009      9.336      1.00x      1.00x         -

     10000       scalar        -      0.0086      0.0086      0.0086      9.312      1.00x      1.00x         -
     10000   sequential        -      0.0086      0.0086      0.0087      9.310      1.00x      1.00x         -
     10000     parallel        1      0.0086      0.0086      0.0086      9.309      1.00x      1.00x         -
     10000     parallel        2      0.0086      0.0086      0.0086      9.309      1.00x      1.00x         -
     10000     parallel        4      0.0086      0.0086      0.0087      9.307      1.00x      1.00x         -
     10000     parallel        8      0.0086      0.0086      0.0087      9.309      1.00x      1.00x         -
     10000     parallel       12      0.0086      0.0086      0.0087      9.310      1.00x      1.00x         -
     10000     parallel       16      0.0086      0.0086      0.0088      9.310      1.00x      1.00x         -
     10000     parallel       24      0.0086      0.0086      0.0087      9.310      1.00x      1.00x         -

     32768       scalar        -      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32768   sequential        -      0.0281      0.0281      0.0284      9.313      1.00x      1.00x         -
     32768     parallel        1      0.0282      0.0281      0.0284      9.309      1.00x      1.00x         -
     32768     parallel        2      0.0282      0.0281      0.0282      9.312      1.00x      1.00x         -
     32768     parallel        4      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32768     parallel        8      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32768     parallel       12      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32768     parallel       16      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32768     parallel       24      0.0282      0.0281      0.0284      9.311      1.00x      1.00x         -

     32769       scalar        -      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32769   sequential        -      0.0282      0.0281      0.0284      9.312      1.00x      1.00x         -
     32769     parallel        1      0.0282      0.0282      0.0284      9.311      1.00x      1.00x         -
     32769     parallel        2      0.0249      0.0247      0.0252     10.516      1.13x      1.13x         -
     32769     parallel        4      0.0204      0.0196      0.0220     12.830      1.38x      1.38x         -
     32769     parallel        8      0.0247      0.0246      0.0249     10.630      1.14x      1.14x         -
     32769     parallel       12      0.0281      0.0278      0.0285      9.328      1.00x      1.00x         -
     32769     parallel       16      0.0302      0.0297      0.0310      8.683      0.93x      0.93x         -
     32769     parallel       24      0.0368      0.0359      0.0373      7.129      0.77x      0.77x         -

     40000       scalar        -      0.0344      0.0344      0.0344      9.314      1.00x      1.00x         -
     40000   sequential        -      0.0344      0.0344      0.0344      9.314      1.00x      1.00x         -
     40000     parallel        1      0.0344      0.0344      0.0344      9.314      1.00x      1.00x         -
     40000     parallel        2      0.0281      0.0281      0.0289     11.384      1.22x      1.22x         -
     40000     parallel        4      0.0211      0.0211      0.0230     15.138      1.63x      1.63x         -
     40000     parallel        8      0.0249      0.0248      0.0258     12.850      1.38x      1.38x         -
     40000     parallel       12      0.0286      0.0285      0.0296     11.187      1.20x      1.20x         -
     40000     parallel       16      0.0308      0.0304      0.0314     10.387      1.12x      1.12x         -
     40000     parallel       24      0.0388      0.0384      0.0408      8.245      0.89x      0.89x         -

     50000       scalar        -      0.0430      0.0429      0.0430      9.312      1.00x      1.00x         -
     50000   sequential        -      0.0430      0.0429      0.0430      9.308      1.00x      1.00x         -
     50000     parallel        1      0.0430      0.0429      0.0430      9.307      1.00x      1.00x         -
     50000     parallel        2      0.0323      0.0323      0.0332     12.375      1.33x      1.33x         -
     50000     parallel        4      0.0232      0.0232      0.0234     17.222      1.85x      1.85x         -
     50000     parallel        8      0.0255      0.0254      0.0259     15.681      1.68x      1.68x         -
     50000     parallel       12      0.0296      0.0293      0.0300     13.519      1.45x      1.45x         -
     50000     parallel       16      0.0315      0.0309      0.0322     12.696      1.36x      1.36x         -
     50000     parallel       24      0.0395      0.0392      0.0401     10.123      1.09x      1.09x         -

     75000       scalar        -      0.0645      0.0645      0.0647      9.296      1.00x      1.00x         -
     75000   sequential        -      0.0646      0.0645      0.0647      9.294      1.00x      1.00x         -
     75000     parallel        1      0.0645      0.0645      0.0646      9.297      1.00x      1.00x         -
     75000     parallel        2      0.0437      0.0436      0.0448     13.735      1.48x      1.48x         -
     75000     parallel        4      0.0287      0.0287      0.0290     20.874      2.25x      2.25x         -
     75000     parallel        8      0.0287      0.0285      0.0294     20.927      2.25x      2.25x         -
     75000     parallel       12      0.0311      0.0310      0.0316     19.320      2.08x      2.08x         -
     75000     parallel       16      0.0334      0.0327      0.0338     17.944      1.93x      1.93x         -
     75000     parallel       24      0.0439      0.0418      0.0456     13.679      1.47x      1.47x         -

    100000       scalar        -      0.0864      0.0860      0.0872      9.258      1.00x      1.00x         -
    100000   sequential        -      0.0861      0.0860      0.0865      9.291      1.00x      1.00x         -
    100000     parallel        1      0.0861      0.0860      0.0865      9.296      1.00x      1.00x         -
    100000     parallel        2      0.0535      0.0534      0.0542     14.959      1.62x      1.61x         -
    100000     parallel        4      0.0344      0.0343      0.0349     23.263      2.51x      2.50x         -
    100000     parallel        8      0.0310      0.0310      0.0315     25.770      2.78x      2.77x         -
    100000     parallel       12      0.0331      0.0328      0.0335     24.182      2.61x      2.60x         -
    100000     parallel       16      0.0342      0.0339      0.0345     23.415      2.53x      2.52x         -
    100000     parallel       24      0.0429      0.0425      0.0433     18.667      2.02x      2.01x         -

    200000       scalar        -      0.1721      0.1720      0.1794      9.296      1.00x      1.00x         -
    200000   sequential        -      0.1720      0.1719      0.1724      9.302      1.00x      1.00x         -
    200000     parallel        1      0.1721      0.1720      0.1743      9.295      1.00x      1.00x         -
    200000     parallel        2      0.0987      0.0986      0.0990     16.216      1.74x      1.74x         -
    200000     parallel        4      0.0568      0.0567      0.0584     28.144      3.03x      3.03x         -
    200000     parallel        8      0.0426      0.0423      0.0432     37.528      4.04x      4.03x         -
    200000     parallel       12      0.0404      0.0399      0.0422     39.584      4.26x      4.26x         -
    200000     parallel       16      0.0403      0.0400      0.0422     39.724      4.27x      4.27x         -
    200000     parallel       24      0.0464      0.0463      0.0487     34.469      3.71x      3.71x         -

    500000       scalar        -      0.4301      0.4298      0.4307      9.301      1.00x      1.00x         -
    500000   sequential        -      0.4300      0.4298      0.4308      9.302      1.00x      1.00x         -
    500000     parallel        1      0.4301      0.4298      0.4306      9.300      1.00x      1.00x         -
    500000     parallel        2      0.2288      0.2272      0.2292     17.483      1.88x      1.88x         -
    500000     parallel        4      0.1221      0.1220      0.1233     32.754      3.52x      3.52x         -
    500000     parallel        8      0.0755      0.0749      0.0765     52.985      5.70x      5.70x         -
    500000     parallel       12      0.0625      0.0622      0.0630     63.960      6.88x      6.88x         -
    500000     parallel       16      0.0561      0.0558      0.0568     71.347      7.67x      7.67x         -
    500000     parallel       24      0.0577      0.0576      0.0591     69.298      7.45x      7.45x         -

   1000000       scalar        -      0.8601      0.8594      0.8616      9.301      1.00x      1.00x         -
   1000000   sequential        -      0.8602      0.8600      0.8687      9.300      1.00x      1.00x         -
   1000000     parallel        1      0.8605      0.8600      0.8610      9.297      1.00x      1.00x         -
   1000000     parallel        2      0.4421      0.4410      0.4642     18.094      1.95x      1.95x         -
   1000000     parallel        4      0.2295      0.2292      0.2314     34.856      3.75x      3.75x         -
   1000000     parallel        8      0.1296      0.1293      0.1334     61.731      6.64x      6.64x         -
   1000000     parallel       12      0.0985      0.0981      0.0988     81.257      8.74x      8.74x         -
   1000000     parallel       16      0.0843      0.0839      0.0857     94.878     10.20x     10.20x         -
   1000000     parallel       24      0.0775      0.0772      0.0806    103.193     11.10x     11.10x         -

   2000000       scalar        -      1.7201      1.7188      1.7222      9.302      1.00x      1.00x         -
   2000000   sequential        -      1.7208      1.7189      1.7618      9.298      1.00x      1.00x         -
   2000000     parallel        1      1.7212      1.7204      1.7368      9.296      1.00x      1.00x         -
   2000000     parallel        2      0.8721      0.8719      0.8771     18.346      1.97x      1.97x         -
   2000000     parallel        4      0.4442      0.4439      0.4491     36.016      3.87x      3.87x         -
   2000000     parallel        8      0.2370      0.2365      0.2386     67.507      7.26x      7.26x         -
   2000000     parallel       12      0.1711      0.1697      0.1736     93.536     10.06x     10.06x         -
   2000000     parallel       16      0.1446      0.1427      0.1533    110.679     11.90x     11.90x         -
   2000000     parallel       24      0.1164      0.1160      0.1245    137.450     14.78x     14.78x         -

   5000000       scalar        -      4.4243      4.3939      4.4705      9.041      1.00x      1.00x         -
   5000000   sequential        -      4.4262      4.4176      4.4338      9.037      1.00x      1.00x         -
   5000000     parallel        1      4.4244      4.4222      4.4667      9.041      1.00x      1.00x         -
   5000000     parallel        2      2.2276      2.2248      2.2615     17.957      1.99x      1.99x         -
   5000000     parallel        4      1.1309      1.1268      1.1689     35.368      3.91x      3.91x         -
   5000000     parallel        8      0.5734      0.5686      0.5935     69.762      7.72x      7.72x         -
   5000000     parallel       12      0.4004      0.3985      0.4124     99.897     11.05x     11.05x         -
   5000000     parallel       16      0.3491      0.3400      0.3558    114.584     12.67x     12.68x         -
   5000000     parallel       24      0.2881      0.2474      0.3110    138.817     15.35x     15.36x         -

  10000000       scalar        -      8.9291      8.9117      8.9683      8.959      1.00x      1.00x         -
  10000000   sequential        -      8.9081      8.8905      8.9417      8.981      1.00x      1.00x         -
  10000000     parallel        1      8.9257      8.9100      9.1046      8.963      1.00x      1.00x         -
  10000000     parallel        2      4.4878      4.4778      4.5271     17.826      1.99x      1.98x         -
  10000000     parallel        4      2.2816      2.2721      2.2943     35.063      3.91x      3.90x         -
  10000000     parallel        8      1.1601      1.1581      1.1675     68.960      7.70x      7.68x         -
  10000000     parallel       12      0.9753      0.8248      1.3383     82.024      9.15x      9.13x         -
  10000000     parallel       16      1.4480      1.3495      1.5546     55.249      6.17x      6.15x         -
  10000000     parallel       24      1.2342      1.2227      1.3051     64.822      7.24x      7.22x         -

  33554432       scalar        -     29.8443     29.8057     29.8912      8.995      1.00x      1.00x      0.99
  33554432   sequential        -     29.8646     29.8184     31.4161      8.988      1.00x      1.00x      0.98
  33554432     parallel        1     29.8942     29.8225     29.9609      8.980      1.00x      1.00x      0.99
  33554432     parallel        2     15.0586     15.0132     15.1801     17.826      1.98x      1.98x      2.07
  33554432     parallel        4      7.7763      7.7330      8.1266     34.520      3.84x      3.84x         -
  33554432     parallel        8      5.8919      5.7256      6.8201     45.560      5.07x      5.07x         -
  33554432     parallel       12      5.6894      5.5362      5.8762     47.182      5.25x      5.25x         -
  33554432     parallel       16      6.3455      6.2661      6.4827     42.303      4.70x      4.71x         -
  33554432     parallel       24      6.0309      5.9684      6.1187     44.510      4.95x      4.95x         -
```

</details>

### dot through the public API (one run)

`dot_benchmark`, same build and conditions, run once after the two `cpu_parallel_benchmark` runs.

- One call of `mensurflow::dot` on 1 element costs 1.1 µs: the fixed overhead of the public API
  (dispatch, allocating the result, reading it back).
- From 100,000 elements up, the API costs within noise of calling the parallel kernel directly
  (`vs api` 0.83-1.06x in section 3; the 0.83x at 10 million is noise: min and max of those rows
  are up to about 30 % apart).
- Through the API, `dot` reaches 77-100 GB/s at 1 million elements (the data fits in cache; the
  two sections of the output differ) and 38-59 GB/s from 10 million up, against about 9 GB/s for
  the sequential loop.
- The `scalar` rows match `sequential` here too (see the note on MSVC above).

<details>
<summary>Full output: dot_benchmark (through the public API)</summary>

```
=== dot (float32) through mensurflow::dot ===
cpu: AMD Ryzen 9 5900X 12-Core Processor | hardware threads: 24 | OpenMP threads: 24
compiler: msvc 1951 | OpenMP: yes (2.0) | AVX2: yes | build: Release
timing: 1 warm-up call(s), median of 9 samples (3 if a call takes over 0.2 s), each sample at least 2 ms
for comparable numbers: plugged in, High performance power plan, other programs closed, run twice
devices: cpu:0
each call includes reading the result back; GB/s counts 8 bytes per element (two float reads)

== 1. dot on every device ==
         n         impl  threads   median ms      min ms      max ms       GB/s   vs cpu:0  cpu/wall
         1        cpu:0        -      0.0012      0.0011      0.0013      0.006      1.00x         -

      1000        cpu:0        -      0.0021      0.0020      0.0021      3.824      1.00x         -

    100000        cpu:0        -      0.0439      0.0434      0.0448     18.204      1.00x         -

   1000000        cpu:0        -      0.1039      0.0822      0.1069     77.002      1.00x         -

  10000000        cpu:0        -      1.5084      1.4611      1.6754     53.038      1.00x         -

  33554432        cpu:0        -      6.9826      6.2693      7.4661     38.443      1.00x         -


== 2. one call on 1 element (pure overhead) ==
         n         impl  threads   median ms      min ms      max ms       GB/s  cpu/wall
         1        cpu:0        -      0.0011      0.0011      0.0011          -         -

== 3. CPU: API vs the kernels called directly ==
         n         impl  threads   median ms      min ms      max ms       GB/s     vs api  cpu/wall
         1          api        -      0.0011      0.0011      0.0011      0.007      1.00x         -
         1     parallel       24      0.0000      0.0000      0.0000      1.552    214.72x         -
         1   sequential        -      0.0000      0.0000      0.0000      3.102    429.08x         -
         1       scalar        -      0.0000      0.0000      0.0000      3.103    429.24x         -

      1000          api        -      0.0020      0.0020      0.0020      4.082      1.00x         -
      1000     parallel       24      0.0009      0.0009      0.0009      9.338      2.29x         -
      1000   sequential        -      0.0009      0.0009      0.0009      9.357      2.29x         -
      1000       scalar        -      0.0009      0.0009      0.0009      9.357      2.29x         -

    100000          api        -      0.0445      0.0438      0.0477     17.970      1.00x         -
    100000     parallel       24      0.0421      0.0418      0.0435     18.982      1.06x         -
    100000   sequential        -      0.0860      0.0860      0.0862      9.297      0.52x         -
    100000       scalar        -      0.0860      0.0860      0.0863      9.299      0.52x         -

   1000000          api        -      0.0801      0.0794      0.0880     99.930      1.00x         -
   1000000     parallel       24      0.0777      0.0773      0.0787    102.981      1.03x         -
   1000000   sequential        -      0.8597      0.8590      0.8690      9.306      0.09x         -
   1000000       scalar        -      0.8596      0.8591      0.8613      9.306      0.09x         -

  10000000          api        -      1.3456      1.1856      1.5322     59.453      1.00x         -
  10000000     parallel       24      1.6165      1.3204      1.7062     49.488      0.83x         -
  10000000   sequential        -      9.2494      8.9022     11.2577      8.649      0.15x         -
  10000000       scalar        -      9.0734      8.8777      9.5078      8.817      0.15x         -

  33554432          api        -      6.2643      5.9306      7.1187     42.852      1.00x         -
  33554432     parallel       24      6.3183      6.0040      7.1419     42.485      0.99x         -
  33554432   sequential        -     29.8751     29.8248     29.9960      8.985      0.21x      0.99
  33554432       scalar        -     29.9379     29.8469     30.5051      8.966      0.21x      0.98
```

</details>

### Comparison

Best speedup over sequential at 10,000,000 elements, and sequential throughput. Both machines use
a separate sequential baseline with staggered arrays; the desktop numbers come from the benchmark
harness (`vs seq` column), the laptop's from the earlier format, which measures the same
comparison. Values are the range over each machine's two runs, with the best thread count.

| Op | Laptop speedup | Desktop speedup | Laptop sequential GB/s | Desktop sequential GB/s |
|---|---|---|---|---|
| add | 1.10-1.37x | 1.25-1.39x | 15-19 | 33.2-33.5 |
| mul | 1.09-1.21x | 1.30-1.48x | 16-17 | 31.8-32.0 |
| dot | 3.5-4.6x | 7.7-9.1x | about 5 | 9.0 |

Caveats for this comparison:

- Compiler: the desktop used MSVC (OpenMP 2.0) and the laptop GCC 16.1.0 (OpenMP 5.2). Part of
  the difference can come from the compiler rather than the hardware. Building the desktop with
  MinGW as well would separate the two.
- Power: neither machine ran on a High performance plan (laptop on battery with Balanced, desktop on
  "Dedera" with a 95 % processor cap). The desktop's sequential numbers would probably be higher
  with boost enabled.

## Matrix multiplication

`C = A * B` for square float32 matrices, `benchmarks/blas/matmul_benchmark.cpp`. The first two
result tables are from the laptop, with the same compiler and conditions as Machine 1 above (on
battery, Balanced plan, so expect noise). The desktop's results are at the end of this section,
under "Desktop: Ryzen 9 5900X (Machine 2)".

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

### Desktop: Ryzen 9 5900X (Machine 2)

Same build and conditions as Machine 2 above: benchmark harness (`feat/benchmark-harness`, commit
`dcdc3c7`), MSVC 19.51 (OpenMP 2.0), default build without `MENSURFLOW_NATIVE`, "Dedera" power plan
(95 % processor cap, probably no boost), 2026-11-01. Two runs; cells show run 1 / run 2 in
GFLOP/s, with the speedup over sequential in brackets. The `scalar` row is not scalar on MSVC (see
the note under Machine 2) and matches `sequential` at every size, so it is left out here.

| n | Sequential | Blocked, 1 thread | 12 threads | 24 threads |
|---|---|---|---|---|
| 64 | 21.6 / 20.9 GFLOP/s | 19.7 / 19.7 (0.91x / 0.94x) | 15.7 / 15.8 (0.73x / 0.76x) | 11.5 / 11.5 (0.53x / 0.55x) |
| 128 | 22.4 / 22.5 | 22.9 / 22.6 (1.02x / 1.01x) | 33.7 / 50.1 (1.51x / 2.23x) | 43.8 / 44.3 (1.96x / 1.97x) |
| 256 | 23.3 / 23.3 | 22.7 / 22.7 (0.98x / 0.98x) | 40.8 / 60.7 (1.75x / 2.60x) | 69.9 / 61.8 (3.00x / 2.65x) |
| 512 | 22.3 / 22.4 | 22.4 / 22.3 (1.00x / 1.00x) | 70.3 / 39.3 (3.15x / 1.75x) | 71.4 / 71.5 (3.21x / 3.19x) |
| 1024 | 22.9 / 22.9 | 19.4 / 19.4 (0.85x / 0.85x) | 63.5 / 66.0 (2.77x / 2.89x) | 69.6 / 70.7 (3.04x / 3.09x) |

What it shows:

- The two runs agree within 4 % in every row except the 12-thread rows (n = 128 to 512) and
  24 threads at n = 256. Those jump between runs: 12 threads give either about 1.5-1.75x or
  2.2-3.15x, a pattern that fits the threads sometimes sharing physical cores (not confirmed).
- The sequential loop runs at about 21-23 GFLOP/s, about twice the laptop's 11, and cache blocking
  alone does not help (0.85-1.02x).
- 24 threads give at most 3.2x (71 GFLOP/s) on this 12-core CPU, less than the laptop's 4.3x on
  4 cores. n = 64 is slower with every thread count.
- Surprising: 2 threads are about half as fast as 1 thread at every size in both runs
  (0.48-0.56x). From n = 128 up, 4 threads only match 1 thread (1.01-1.08x), 8 threads give
  1.79-2.15x and 16 threads 1.76-2.23x. The `cpu/wall` column shows that the threads are busy, not
  waiting: 1.99-2.04 with 2 threads and 3.70-4.05 with 4 (run 1). So each thread runs at about a
  quarter of the 1-thread speed. An earlier run with the previous benchmark on the same desktop
  showed the same pattern. The laptop's GCC build does not (2 threads: 1.8-2.3x), so it may come
  from MSVC's OpenMP 2.0 or from the threads getting in each other's way on this CPU; the cause was
  not identified.

<details>
<summary>Full output: matmul_benchmark, desktop (default build), run 1</summary>

```
=== Matrix multiplication C = A * B, square float32 matrices ===
cpu: AMD Ryzen 9 5900X 12-Core Processor | hardware threads: 24 | OpenMP threads: 24
compiler: msvc 1951 | OpenMP: yes (2.0) | AVX2: yes | build: Release
timing: 1 warm-up call(s), median of 9 samples (3 if a call takes over 0.2 s), each sample at least 2 ms
for comparable numbers: plugged in, High performance power plan, other programs closed, run twice
impl: scalar = plain i-k-j loop, no SIMD; sequential = the same loop with the compiler's SIMD; parallel = blocked kernel with N threads
vs scalar / vs seq = that baseline's median time / this row's median time

== matmul ==
         n         impl  threads   median ms      min ms      max ms    GFLOP/s  vs scalar     vs seq  cpu/wall
        64       scalar        -      0.0242      0.0242      0.0243     21.626      1.00x      1.00x         -
        64   sequential        -      0.0242      0.0242      0.0242     21.641      1.00x      1.00x         -
        64     parallel        1      0.0266      0.0265      0.0268     19.740      0.91x      0.91x         -
        64     parallel        2      0.0505      0.0505      0.0506     10.385      0.48x      0.48x         -
        64     parallel        4      0.0324      0.0324      0.0334     16.171      0.75x      0.75x         -
        64     parallel        8      0.0308      0.0308      0.0317     16.996      0.79x      0.79x         -
        64     parallel       12      0.0333      0.0330      0.0356     15.746      0.73x      0.73x         -
        64     parallel       16      0.0358      0.0355      0.0373     14.663      0.68x      0.68x         -
        64     parallel       24      0.0458      0.0447      0.0464     11.459      0.53x      0.53x         -

       128       scalar        -      0.1871      0.1871      0.1883     22.415      1.00x      1.00x         -
       128   sequential        -      0.1875      0.1871      0.1892     22.370      1.00x      1.00x         -
       128     parallel        1      0.1835      0.1834      0.1846     22.861      1.02x      1.02x         -
       128     parallel        2      0.3323      0.3320      0.3360     12.621      0.56x      0.56x         -
       128     parallel        4      0.1734      0.1734      0.1745     24.186      1.08x      1.08x         -
       128     parallel        8      0.1043      0.1014      0.1144     40.230      1.79x      1.80x         -
       128     parallel       12      0.1245      0.1098      0.1319     33.698      1.50x      1.51x         -
       128     parallel       16      0.1052      0.1049      0.1062     39.870      1.78x      1.78x         -
       128     parallel       24      0.0957      0.0949      0.0979     43.825      1.96x      1.96x         -

       256       scalar        -      1.4397      1.4386      1.4442     23.307      1.00x      1.00x         -
       256   sequential        -      1.4397      1.4388      1.4555     23.307      1.00x      1.00x         -
       256     parallel        1      1.4765      1.4752      1.4783     22.726      0.98x      0.98x         -
       256     parallel        2      2.6425      2.6405      2.6548     12.698      0.54x      0.54x         -
       256     parallel        4      1.3303      1.3281      1.3324     25.222      1.08x      1.08x         -
       256     parallel        8      0.6842      0.6830      0.6914     49.038      2.10x      2.10x         -
       256     parallel       12      0.8223      0.7809      0.8236     40.804      1.75x      1.75x         -
       256     parallel       16      0.6676      0.6660      0.6690     50.259      2.16x      2.16x         -
       256     parallel       24      0.4802      0.4776      0.4838     69.881      3.00x      3.00x         -

       512       scalar        -     11.8090     11.7056     11.9250     22.731      1.00x      1.02x      1.03
       512   sequential        -     12.0526     11.8734     12.1265     22.272      0.98x      1.00x      1.01
       512     parallel        1     12.0096     11.9928     12.1260     22.352      0.98x      1.00x      1.01
       512     parallel        2     22.1589     22.1254     22.2506     12.114      0.53x      0.54x      2.04
       512     parallel        4     11.1447     11.1145     12.1628     24.086      1.06x      1.08x      3.70
       512     parallel        8      5.6000      5.5940      5.6325     47.935      2.11x      2.15x         -
       512     parallel       12      3.8209      3.8057      3.8385     70.255      3.09x      3.15x         -
       512     parallel       16      5.3930      5.3288      5.4947     49.775      2.19x      2.23x         -
       512     parallel       24      3.7578      3.7158      3.7898     71.434      3.14x      3.21x         -

      1024       scalar        -     94.3030     93.9472     95.5023     22.772      1.00x      0.99x      0.99
      1024   sequential        -     93.7404     93.4546     94.9100     22.909      1.01x      1.00x      1.00
      1024     parallel        1    110.9128    110.3705    111.9518     19.362      0.85x      0.85x      0.99
      1024     parallel        2    183.5897    182.7800    184.8622     11.697      0.51x      0.51x      1.99
      1024     parallel        4     92.1513     91.9984     92.2705     23.304      1.02x      1.02x      4.05
      1024     parallel        8     46.1034     46.0040     58.9795     46.580      2.05x      2.03x      7.89
      1024     parallel       12     33.7945     31.1160     49.1255     63.545      2.79x      2.77x     10.34
      1024     parallel       16     43.0128     38.4291     44.8630     49.927      2.19x      2.18x     11.26
      1024     parallel       24     30.8452     30.0536     39.4570     69.621      3.06x      3.04x     22.37
```

</details>

<details>
<summary>Full output: matmul_benchmark, desktop (default build), run 2</summary>

```n=== Matrix multiplication C = A * B, square float32 matrices ===
cpu: AMD Ryzen 9 5900X 12-Core Processor | hardware threads: 24 | OpenMP threads: 24
compiler: msvc 1951 | OpenMP: yes (2.0) | AVX2: yes | build: Release
timing: 1 warm-up call(s), median of 9 samples (3 if a call takes over 0.2 s), each sample at least 2 ms
for comparable numbers: plugged in, High performance power plan, other programs closed, run twice
impl: scalar = plain i-k-j loop, no SIMD; sequential = the same loop with the compiler's SIMD; parallel = blocked kernel with N threads
vs scalar / vs seq = that baseline's median time / this row's median time

== matmul ==
         n         impl  threads   median ms      min ms      max ms    GFLOP/s  vs scalar     vs seq  cpu/wall
        64       scalar        -      0.0244      0.0242      0.0272     21.454      1.00x      1.03x         -
        64   sequential        -      0.0251      0.0242      0.0261     20.916      0.97x      1.00x         -
        64     parallel        1      0.0266      0.0265      0.0273     19.721      0.92x      0.94x         -
        64     parallel        2      0.0503      0.0502      0.0533     10.414      0.49x      0.50x         -
        64     parallel        4      0.0330      0.0329      0.0338     15.880      0.74x      0.76x         -
        64     parallel        8      0.0315      0.0303      0.0323     16.639      0.78x      0.80x         -
        64     parallel       12      0.0332      0.0330      0.0353     15.792      0.74x      0.76x         -
        64     parallel       16      0.0362      0.0360      0.0382     14.481      0.67x      0.69x         -
        64     parallel       24      0.0457      0.0450      0.0479     11.470      0.53x      0.55x         -

       128       scalar        -      0.1865      0.1865      0.1956     22.488      1.00x      1.00x         -
       128   sequential        -      0.1866      0.1865      0.1874     22.472      1.00x      1.00x         -
       128     parallel        1      0.1852      0.1849      0.1947     22.647      1.01x      1.01x         -
       128     parallel        2      0.3337      0.3321      0.3463     12.568      0.56x      0.56x         -
       128     parallel        4      0.1765      0.1740      0.1859     23.765      1.06x      1.06x         -
       128     parallel        8      0.1044      0.1012      0.1092     40.179      1.79x      1.79x         -
       128     parallel       12      0.0838      0.0814      0.0926     50.074      2.23x      2.23x         -
       128     parallel       16      0.1061      0.1052      0.1110     39.521      1.76x      1.76x         -
       128     parallel       24      0.0947      0.0945      0.0960     44.302      1.97x      1.97x         -

       256       scalar        -      1.4473      1.4394      1.5446     23.184      1.00x      0.99x         -
       256   sequential        -      1.4395      1.4389      1.4835     23.309      1.01x      1.00x         -
       256     parallel        1      1.4761      1.4756      1.4787     22.733      0.98x      0.98x         -
       256     parallel        2      2.6899      2.6450      2.7519     12.474      0.54x      0.54x         -
       256     parallel        4      1.3300      1.3284      1.3749     25.229      1.09x      1.08x         -
       256     parallel        8      0.6857      0.6816      0.6962     48.936      2.11x      2.10x         -
       256     parallel       12      0.5527      0.4849      0.6022     60.713      2.62x      2.60x         -
       256     parallel       16      0.6716      0.6665      0.6875     49.960      2.15x      2.14x         -
       256     parallel       24      0.5433      0.4818      0.5517     61.763      2.66x      2.65x         -

       512       scalar        -     11.7799     11.6975     12.4085     22.788      1.00x      1.02x      1.02
       512   sequential        -     11.9924     11.9100     12.0479     22.384      0.98x      1.00x      1.01
       512     parallel        1     12.0148     11.9642     12.1375     22.342      0.98x      1.00x      1.01
       512     parallel        2     22.5552     22.1439     23.1388     11.901      0.52x      0.53x      1.92
       512     parallel        4     11.2188     11.1258     12.4695     23.927      1.05x      1.07x      3.05
       512     parallel        8      5.6238      5.5922      5.7403     47.732      2.09x      2.13x         -
       512     parallel       12      6.8383      4.9037      7.2140     39.255      1.72x      1.75x         -
       512     parallel       16      5.4479      5.3519      5.4828     49.273      2.16x      2.20x         -
       512     parallel       24      3.7557      3.7287      3.7847     71.474      3.14x      3.19x         -

      1024       scalar        -     96.0113     94.3851     96.4091     22.367      1.00x      0.98x      0.93
      1024   sequential        -     93.9461     93.8008     94.2583     22.859      1.02x      1.00x      1.00
      1024     parallel        1    110.9679    110.4935    111.6147     19.352      0.87x      0.85x      1.02
      1024     parallel        2    184.2647    183.8186    184.8778     11.654      0.52x      0.51x      2.01
      1024     parallel        4     92.5670     91.9973     93.8322     23.199      1.04x      1.01x      3.90
      1024     parallel        8     46.7416     46.3294     58.7058     45.944      2.05x      2.01x      7.19
      1024     parallel       12     32.5372     31.0395     41.5591     66.001      2.95x      2.89x     11.32
      1024     parallel       16     43.6631     38.5966     44.3679     49.183      2.20x      2.15x     12.53
      1024     parallel       24     30.3662     29.9405     32.3634     70.720      3.16x      3.09x     22.38
```

</details>

## Machine 3: laptop

| | |
|---|---|
| Model | Acer Nitro AN515-46 (reported by system DMI) |
| CPU | AMD Ryzen 5 6600H with Radeon Graphics |
| Cores / threads | 6 / 12 |
| Clock | Linux reports a maximum of 4.564 GHz; clocks during the measurements were not recorded |
| Cache | L2 3 MiB (512 KiB per core), L3 16 MiB |
| RAM | Approximately 14.8 GiB usable memory reported by Linux; memory speed was not recorded |
| OS | Ubuntu 24.04.4 LTS, Linux 6.8.0-110-generic |
| Compiler | GCC 13.3.0 (Ubuntu), Release, `-O3 -DNDEBUG -fopenmp`, `MENSURFLOW_NATIVE=OFF` |
| OpenMP | 4.5 reported by CMake (libgomp) |
| Code | branch `feat/cpu-elementwise-cutoff`, commit `84b3cbf` plus the local `sqrt`/`exp`/`sigmoid` implementation and benchmark changes (uncommitted when measured) |
| Power state | AC power connected and Linux CPU governor `performance`, observed before and after the two runs; background activity was not controlled |
| Thread limit | `OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE`; measured with 1, 2, 4, 6, 8, and 12 threads |
| Measurement date | 2026-10-31; CUDA, ROCm, and Metal backends disabled |

### Results (two runs, sequential implementation as the baseline)

The baseline is the plain sequential kernel (`src/backends/cpu/sequential.cpp`, no OpenMP).
Cells show run 1 / run 2. "Best" is the fastest measured count among 1, 2, 4, 6, 8, and 12,
shown in brackets. Affinity was not fixed; six threads do not establish one worker per core.
The table uses Machine 1's four summary sizes; the benchmark also checks smaller inputs.

| Op | n | Sequential (ms) | Speedup, 12 threads | Speedup, best (threads) |
|---|---|---|---|---|
| add | 100,000 | 0.010 / 0.010 | 1.881x / 2.606x | 3.484x (6) / 2.908x (6) |
| add | 1,000,000 | 0.184 / 0.232 | 3.777x / 1.512x | 4.236x (6) / 3.140x (8) |
| add | 10,000,000 | 6.632 / 6.615 | 0.945x / 1.104x | 1.129x (4) / 1.104x (12) |
| add | 33,554,432 | 23.453 / 22.727 | 0.940x / 0.904x | 1.115x (2) / 1.062x (6) |
| sub | 100,000 | 0.009 / 0.010 | 2.438x / 2.512x | 2.438x (12) / 2.828x (6) |
| sub | 1,000,000 | 0.192 / 0.201 | 3.877x / 4.223x | 3.877x (12) / 4.223x (12) |
| sub | 10,000,000 | 6.892 / 6.218 | 1.066x / 0.577x | 1.123x (8) / 0.986x (1) |
| sub | 33,554,432 | 23.392 / 22.708 | 1.042x / 1.088x | 1.122x (2) / 1.088x (12) |
| mul | 100,000 | 0.009 / 0.010 | 2.367x / 2.704x | 2.827x (6) / 3.027x (6) |
| mul | 1,000,000 | 0.210 / 0.173 | 4.308x / 4.461x | 4.497x (6) / 4.461x (12) |
| mul | 10,000,000 | 6.467 / 6.198 | 0.999x / 0.927x | 1.094x (4) / 1.060x (2) |
| mul | 33,554,432 | 23.673 / 23.421 | 0.925x / 1.040x | 1.066x (4) / 1.150x (4) |
| div | 100,000 | 0.022 / 0.019 | 4.104x / 3.924x | 4.845x (6) / 4.398x (6) |
| div | 1,000,000 | 0.278 / 0.233 | 0.087x / 0.039x | 4.340x (8) / 2.808x (4) |
| div | 10,000,000 | 7.081 / 7.187 | 0.776x / 1.099x | 1.148x (2) / 1.208x (6) |
| div | 33,554,432 | 23.523 / 23.433 | 1.078x / 1.118x | 1.126x (2) / 1.122x (2) |
| neg | 100,000 | 0.007 / 0.006 | 2.474x / 2.267x | 2.474x (12) / 2.267x (12) |
| neg | 1,000,000 | 0.088 / 0.100 | 1.411x / 3.533x | 2.308x (8) / 3.533x (12) |
| neg | 10,000,000 | 4.955 / 4.929 | 1.059x / 0.907x | 1.129x (2) / 1.114x (2) |
| neg | 33,554,432 | 17.835 / 17.749 | 0.906x / 1.106x | 1.077x (6) / 1.120x (2) |
| abs | 100,000 | 0.011 / 0.011 | 4.551x / 4.593x | 5.530x (6) / 5.854x (6) |
| abs | 1,000,000 | 0.124 / 0.128 | 4.252x / 0.030x | 5.568x (6) / 4.316x (8) |
| abs | 10,000,000 | 5.076 / 5.023 | 0.776x / 1.078x | 1.143x (2) / 1.128x (6) |
| abs | 33,554,432 | 18.028 / 17.397 | 1.106x / 1.001x | 1.106x (12) / 1.080x (8) |
| square | 100,000 | 0.007 / 0.007 | 2.726x / 2.529x | 3.293x (6) / 2.986x (4) |
| square | 1,000,000 | 0.095 / 0.096 | 3.646x / 3.776x | 4.013x (6) / 4.155x (6) |
| square | 10,000,000 | 5.172 / 5.602 | 0.991x / 0.916x | 1.124x (8) / 1.219x (2) |
| square | 33,554,432 | 18.154 / 17.932 | 1.126x / 1.088x | 1.218x (2) / 1.157x (4) |
| relu | 100,000 | 0.008 / 0.008 | 2.106x / 2.165x | 2.716x (6) / 2.476x (6) |
| relu | 1,000,000 | 0.105 / 0.109 | 2.684x / 3.232x | 3.394x (6) / 3.556x (6) |
| relu | 10,000,000 | 4.875 / 5.182 | 0.801x / 0.513x | 1.096x (4) / 1.084x (2) |
| relu | 33,554,432 | 17.168 / 18.009 | 1.071x / 1.092x | 1.146x (2) / 1.175x (2) |
| sqrt | 100,000 | 0.111 / 0.111 | 4.892x / 5.088x | 5.663x (6) / 5.662x (6) |
| sqrt | 1,000,000 | 1.109 / 1.109 | 5.430x / 5.299x | 5.513x (6) / 5.638x (6) |
| sqrt | 10,000,000 | 11.480 / 11.587 | 2.128x / 1.722x | 2.563x (6) / 2.551x (4) |
| sqrt | 33,554,432 | 38.478 / 38.631 | 1.870x / 2.160x | 2.285x (4) / 2.408x (6) |
| exp | 100,000 | 0.249 / 0.245 | 5.449x / 5.340x | 5.449x (12) / 5.340x (12) |
| exp | 1,000,000 | 2.498 / 2.453 | 4.934x / 5.489x | 5.012x (6) / 5.489x (12) |
| exp | 10,000,000 | 25.377 / 24.927 | 4.082x / 4.198x | 4.293x (6) / 4.517x (6) |
| exp | 33,554,432 | 85.207 / 84.234 | 4.358x / 4.730x | 4.358x (12) / 4.730x (12) |
| sigmoid | 100,000 | 0.272 / 0.272 | 4.799x / 4.778x | 4.799x (12) / 4.778x (12) |
| sigmoid | 1,000,000 | 2.803 / 2.752 | 3.831x / 4.153x | 4.318x (6) / 4.705x (6) |
| sigmoid | 10,000,000 | 28.194 / 27.674 | 3.580x / 2.061x | 4.577x (6) / 3.299x (8) |
| sigmoid | 33,554,432 | 91.236 / 93.958 | 4.330x / 3.460x | 4.330x (12) / 4.423x (6) |
| dot | 100,000 | 0.068 / 0.069 | 5.656x / 5.989x | 5.656x (12) / 5.989x (12) |
| dot | 1,000,000 | 0.682 / 0.667 | 5.185x / 5.713x | 5.185x (12) / 5.713x (12) |
| dot | 10,000,000 | 7.066 / 7.029 | 1.079x / 2.128x | 2.533x (4) / 2.568x (4) |
| dot | 33,554,432 | 25.146 / 25.429 | 2.375x / 2.104x | 2.779x (4) / 2.784x (8) |

What it shows:

- The fastest measured count depends on operation and size. Simple elementwise operations
  are mainly limited by data movement on large arrays: twelve-thread gains at 33,554,432
  elements range from about 0.90x to 1.13x.
- At that size, `exp` reaches 4.36-4.73x and `sigmoid` 3.46-4.33x with twelve threads;
  `sqrt` reaches 1.87-2.16x and `dot` 2.10-2.38x.
- Keep 32,768 elements as an initial cutoff. Smaller inputs use the serial fallback, so
  these measurements do not establish a lower optimum or a universal best thread count.
- Both runs passed all checks: elementwise results matched bit for bit and `dot` passed
  its tolerance check.

Reliability: these results are provisional. With twelve threads, `div` at 1,000,000
inputs takes 3.216 / 5.997 ms and `abs` 0.029 / 4.186 ms across runs. The cause was not
identified; repeat these cases before choosing a thread policy. Background activity,
compiler differences, and Machine 1's different thread count limit comparisons.

Method: preallocated arrays staggered by 1 KB, warmup, and median of nine timed samples;
fast calls are repeated. Tensor allocation, dispatch, packing, and input preparation
are excluded. Binary operations count 12 logical bytes per element; unary and `dot`
count 8. `sqrt` uses nonnegative inputs, other unary inputs lie in [-0.75, 0.75].
Printed times have three decimals; speedups use unrounded times.

Run with the Release CPU build and `MENSURFLOW_NATIVE=OFF`:

```bash
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark
```

### AXPY and bias addition (2026-10-31)

Same Machine 3, GCC 13.3.0 Release build, and timing method as above, with
`MENSURFLOW_NATIVE=OFF` and 1, 2, 4, 6, 8, and 12 threads. AC power was connected
and the CPU governor was `performance` before and after the runs; background activity
was not controlled. Code: `2c8180e` plus the local `axpy`/`bias_add` additions.
Cells show run 1 / run 2; best means the fastest measured thread count in each run.
Both runs completed all 210 timing rows and passed every bit-exact result check.

AXPY uses `alpha=1.25` with fused float32 multiply-add. Bias addition uses a compact
bias whose feature count is the largest power-of-two divisor of n, capped at 256.
For the sizes below the feature counts are 32, 64, 32, 64, 128, 64, and 256 respectively;
the matrix shape is `[n / features, features]`. Logical traffic is 12*n bytes for AXPY
and 8*n + 4*features bytes for bias addition. Tensor preparation is outside timing.

| Op | n | Sequential (ms) | Speedup, 12 threads | Speedup, best (threads) |
|---|---|---|---|---|
| axpy | 100,000 | 0.202 / 0.205 | 7.006x / 0.034x | 7.006x (12) / 5.057x (8) |
| axpy | 200,000 | 0.406 / 0.406 | 5.251x / 9.953x | 5.406x (8) / 9.953x (12) |
| axpy | 500,000 | 1.015 / 1.014 | 7.314x / 0.169x | 7.314x (12) / 5.489x (8) |
| axpy | 1,000,000 | 2.030 / 2.041 | 6.078x / 0.340x | 6.078x (12) / 5.336x (8) |
| axpy | 2,000,000 | 4.102 / 4.247 | 3.358x / 0.782x | 4.677x (8) / 4.350x (8) |
| axpy | 5,000,000 | 10.324 / 11.373 | 2.662x / 1.860x | 3.376x (6) / 3.223x (6) |
| axpy | 33,554,432 | 70.014 / 71.716 | 2.872x / 2.497x | 3.419x (6) / 3.032x (6) |
| bias_add | 100,000 | 0.009 / 0.010 | 2.336x / 0.002x | 2.795x (6) / 2.180x (8) |
| bias_add | 200,000 | 0.016 / 0.021 | 2.883x / 0.003x | 2.883x (12) / 3.009x (8) |
| bias_add | 500,000 | 0.045 / 0.083 | 2.709x / 4.451x | 2.872x (6) / 4.451x (12) |
| bias_add | 1,000,000 | 0.096 / 0.165 | 3.579x / 2.859x | 3.657x (6) / 5.831x (6) |
| bias_add | 2,000,000 | 0.558 / 0.699 | 2.660x / 2.492x | 2.953x (8) / 3.183x (8) |
| bias_add | 5,000,000 | 2.428 / 2.470 | 1.167x / 0.814x | 1.233x (2) / 1.258x (6) |
| bias_add | 33,554,432 | 18.125 / 18.969 | 1.086x / 1.074x | 1.205x (6) / 1.153x (2) |

What it shows:

- Bias addition has little arithmetic per element; data movement becomes the main cost
  on large matrices. At the largest size, the best measured gains are about 1.15-1.20x.
- AXPY has low arithmetic intensity, but this portable build also pays for `std::fma`.
  Its observed scaling includes math-library execution costs as well as memory traffic.
  At the largest size, six threads are fastest in both runs, at about 3.03-3.42x.
- Retain 32,768 elements as the initial cutoff. Four threads improve both operations at
  32,769 elements in both runs. The bias probe there has only one feature; it does not
  calibrate the best cutoff for every matrix shape. Sizes at or below the cutoff use
  the serial fallback, so no optimal threshold below it has been established.

Reliability: the second run contains severe twelve-thread slowdowns, including AXPY
at 100,000 elements (7.006x / 0.034x) and bias addition at the same size
(2.336x / 0.002x). Both values are retained. Their cause was not identified, so these
results are provisional and do not establish a fixed thread policy.

Run just these operations with the existing Release CPU build:

```bash
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark axpy bias_add
```
