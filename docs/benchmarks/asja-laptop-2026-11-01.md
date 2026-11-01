# Benchmark result: Asja, laptop, 2026-11-01

A filled-in example of [TEMPLATE.md](TEMPLATE.md). The same numbers appear, with more discussion, in
[BENCHMARKS.md](../BENCHMARKS.md) under Machine 1.

## Who and when

| | |
|---|---|
| Name | Asja |
| Date | 2026-11-01 |
| Machine nickname | laptop |

## Machine

| | |
|---|---|
| CPU model | Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz |
| Cores / threads | 4 / 8 |
| Cache | L2 1 MB (256 KB per core), L3 8 MB |
| RAM | 2 x 16 GB DDR4: one stick at 3200 MHz, one at 2667 MHz (mismatched, so probably runs at the slower speed) |
| OS | Windows 11 Pro, build 26200 |
| GPU | Intel UHD Graphics (not used by these benchmarks) |

## Build

| | |
|---|---|
| Commit | `dcdc3c7` plus an output-only change to `benchmarks/bench.cpp` (more digits for very fast calls), committed later in `56f291e` |
| Branch | `feat/benchmark-harness` |
| Compiler and version | gcc 16.1.0 (MSYS2 UCRT64) |
| OpenMP | yes (5.2) |
| CMake command | `cmake -S . -B bu -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe -DCMAKE_MAKE_PROGRAM=C:/msys64/mingw64/bin/mingw32-make.exe` (Release is the default, no `MENSURFLOW_NATIVE`) |
| Toolchain notes | UCRT64, not the older `mingw64`: that one has a non-fused `fmaf` and fails the `axpy` tests. The CPU has AVX2 but the build only uses baseline SSE2 |
| Header said `build: Release`? | yes |

## Conditions

- [x] Plugged in (battery charging, 63 % rising to 95 % during the runs)
- [x] Power plan / mode: Windows reports "Balanced" (the Best performance mode overlay could not be checked)
- [x] Other programs: Chrome (12 processes), Edge and Avast were still running but idle (CPU load about 5 %)
- [x] Waited 3 minutes before each run
- [x] Ran `cpu_parallel_benchmark` twice
- [ ] Threads pinned: no (default placement)

## What I ran

```
(3 minute pause)  cpu_parallel_benchmark > cpu_parallel_run1.txt
(3 minute pause)  cpu_parallel_benchmark > cpu_parallel_run2.txt
(3 minute pause)  matmul_benchmark > matmul_run1.txt
```

Started 11:37, 11:43 and 11:48. `cpu_parallel_benchmark` covers all 14 ops at 15 sizes.

## Results

Cells read `run 1 / run 2`. Speedup = sequential time / parallel time. Times are median milliseconds.

### SIMD: scalar vs sequential (n = 1,000,000)

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

### Threads (n = 100,000 and n = 10,000,000)

| op | n | sequential ms | 1 thread | all threads (8) | best of 2, 4, 8 |
|---|---|---|---|---|---|
| add | 100,000 | 0.02079 / 0.02069 | 1.11x / 0.96x | 0.36x / 0.32x | 0.59x / 0.50x |
| sub | 100,000 | 0.01919 / 0.02267 | 0.94x / 1.05x | 0.28x / 0.34x | 0.37x / 0.64x |
| mul | 100,000 | 0.02016 / 0.02313 | 0.98x / 1.09x | 0.30x / 0.34x | 0.38x / 0.66x |
| div | 100,000 | 0.02063 / 0.02261 | 0.98x / 0.99x | 0.31x / 0.35x | 0.39x / 0.56x |
| neg | 100,000 | 0.01566 / 0.01660 | 1.00x / 1.00x | 0.24x / 0.26x | 0.31x / 0.46x |
| abs | 100,000 | 0.01590 / 0.01755 | 0.99x / 1.00x | 0.24x / 0.26x | 0.33x / 0.54x |
| square | 100,000 | 0.01657 / 0.01695 | 1.01x / 1.01x | 0.23x / 0.27x | 0.32x / 0.47x |
| relu | 100,000 | 0.01751 / 0.01852 | 1.03x / 1.03x | 0.26x / 0.27x | 0.35x / 0.57x |
| sqrt | 100,000 | 0.08388 / 0.08929 | 1.00x / 1.01x | 0.96x / 1.06x | 0.99x / 1.40x |
| exp | 100,000 | 0.4516 / 0.4871 | 1.00x / 1.00x | 2.31x / 2.37x | 2.31x / 2.86x |
| sigmoid | 100,000 | 0.5180 / 0.5475 | 0.99x / 0.99x | 2.30x / 2.33x | 2.30x / 2.76x |
| axpy | 100,000 | 0.1908 / 0.1960 | 1.00x / 1.17x | 1.77x / 1.87x | 1.80x / 2.23x |
| bias_add | 100,000 | 0.02020 / 0.02148 | 0.93x / 0.90x | 0.27x / 0.30x | 0.37x / 0.56x |
| dot | 100,000 | 0.1149 / 0.1119 | 1.21x / 1.00x | 1.35x / 1.32x | 1.42x / 1.60x |
| add | 10,000,000 | 6.6756 / 6.5806 | 1.11x / 1.03x | 1.16x / 1.07x | 1.16x / 1.10x |
| sub | 10,000,000 | 6.4391 / 6.5929 | 1.02x / 1.01x | 1.12x / 1.09x | 1.12x / 1.18x |
| mul | 10,000,000 | 5.9621 / 6.4490 | 1.00x / 1.09x | 1.00x / 1.08x | 1.00x / 1.08x |
| div | 10,000,000 | 6.6056 / 6.6819 | 1.00x / 1.02x | 1.09x / 1.11x | 1.12x / 1.17x |
| neg | 10,000,000 | 4.2196 / 4.3606 | 1.04x / 1.02x | 0.88x / 0.92x | 0.99x / 0.94x |
| abs | 10,000,000 | 4.2196 / 4.2737 | 0.84x / 1.01x | 0.89x / 0.92x | 1.07x / 0.94x |
| square | 10,000,000 | 5.1371 / 4.2686 | 1.25x / 0.80x | 1.10x / 0.94x | 1.28x / 0.94x |
| relu | 10,000,000 | 4.3281 / 4.3852 | 1.05x / 1.03x | 0.94x / 0.97x | 0.94x / 0.97x |
| sqrt | 10,000,000 | 8.9500 / 9.1125 | 1.05x / 1.00x | 1.84x / 1.93x | 1.84x / 1.93x |
| exp | 10,000,000 | 43.0849 / 45.7848 | 1.01x / 1.00x | 3.47x / 3.49x | 3.47x / 3.49x |
| sigmoid | 10,000,000 | 46.0249 / 49.9925 | 0.93x / 1.00x | 3.06x / 3.12x | 3.06x / 3.53x |
| axpy | 10,000,000 | 19.2260 / 19.1744 | 0.99x / 1.03x | 2.99x / 3.11x | 2.99x / 3.11x |
| bias_add | 10,000,000 | 5.3904 / 4.3526 | 1.25x / 1.00x | 1.17x / 0.93x | 1.36x / 1.00x |
| dot | 10,000,000 | 12.5836 / 11.9880 | 1.03x / 1.00x | 4.63x / 4.46x | 4.63x / 4.46x |

### Matrix multiplication (GFLOP/s, one run)

| n | scalar | sequential | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|---|---|
| 64 | 3.887 | 12.073 | 11.181 | 9.718 | 8.402 | 6.487 |
| 128 | 3.705 | 11.749 | 11.863 | 19.566 | 19.817 | 22.973 |
| 256 | 4.069 | 14.003 | 12.615 | 26.189 | 27.289 | 36.336 |
| 512 | 4.269 | 12.096 | 13.777 | 27.706 | 52.134 | 49.869 |
| 1024 | 4.301 | 12.616 | 11.423 | 13.112 | 23.803 | 38.989 |

## Stability

Output of `awk -f docs/benchmarks/compare_runs.awk cpu_parallel_run1.txt cpu_parallel_run2.txt`:

```
group            rows      >10%      >25%  run2 slower  run2 faster
parallel t=1      198        28         1           22            6
parallel t=2      199       134       109           10          124
parallel t=4      199       126        72            7          119
parallel t=8      200        39         4           19           20
scalar            199        37         1           32            5
sequential        199        26         1           18            8
(66 rows under 1 microsecond skipped)
```

Reading: the scalar, sequential, 1-thread and 8-thread rows agree reasonably (about 15-20 % of rows
differ by more than 10 %, almost none by more than 25 %). The **2- and 4-thread rows are not reliable**:
more than half differ by over 10 %, and run 2 was faster in about 120 of them. Probably thread
placement on hyperthreads; not confirmed.

## Observations

- SIMD gives 1.3-2.7x on simple arithmetic and nothing on `sqrt`, `exp`, `sigmoid`, `axpy` and `dot`
  (the compiler did not vectorize them).
- Memory-bound ops (`add`, `mul`, `relu`, ...) are 3-4x **slower** with 8 threads at n = 100,000 and
  gain at most about 1.3x at n = 10,000,000. Compute-bound ops (`exp`, `sigmoid`, `axpy`, `dot`)
  gain 3-4.5x at n = 10,000,000.
- `matmul` reaches 39 GFLOP/s at n = 1024 with 8 threads and 52 GFLOP/s at n = 512 with 4 threads.

## Problems

- Two runs started back to back (an earlier pair, not shown) were slower in run 2 in almost every row,
  including single-thread ones: the CPU slows down when hot. Hence the 3-minute pauses.
- The power mode could not be confirmed as Best performance.
