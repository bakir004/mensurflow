# Benchmark result: <your name>, <machine nickname>, <YYYY-MM-DD>

Copy this file to `<yourname>-<machine>-<YYYY-MM-DD>.md` and fill in every field. Delete the
instructions in italics. See [README.md](README.md).

## Who and when

| | |
|---|---|
| Name | |
| Date | YYYY-MM-DD |
| Machine nickname | (laptop, desktop, ...) |

## Machine

| | |
|---|---|
| CPU model | *(the `cpu:` line of the benchmark header)* |
| Cores / threads | *(physical cores / hardware threads)* |
| Cache | *(L2 and L3 sizes, if you know them)* |
| RAM | *(size, type, speed in MHz, and number of sticks; list every stick)* |
| OS | *(name and version or build)* |
| GPU | *(only if relevant; the CPU benchmarks do not use it)* |

## Build

| | |
|---|---|
| Commit | *(`git log -1 --format=%h`; say so if you had uncommitted changes)* |
| Branch | |
| Compiler and version | *(the `compiler:` line of the header)* |
| OpenMP | *(the `OpenMP:` field of the header)* |
| CMake command | *(the exact command; `-DMENSURFLOW_NATIVE=OFF`, Release)* |
| Toolchain notes | *(for example MSYS2 UCRT64, MSVC, Linux GCC)* |
| Header said `build: Release`? | yes / no |

## Conditions

*Tick what is true. Write the real value where it asks.*

- [ ] Plugged in (battery charging or full)
- [ ] Power plan / mode: *(what Windows, Linux or macOS really reports, not what you hoped for)*
- [ ] Other programs closed; still running: *(browsers, antivirus, chat, ...)*
- [ ] Waited about 3 minutes before each run (the CPU cools down)
- [ ] Ran `cpu_parallel_benchmark` twice
- [ ] Threads pinned (`OMP_PLACES`, `OMP_PROC_BIND`)? *(yes with values / no)*

## What I ran

*Paste the commands and which benchmarks, for example:*

```
cpu_parallel_benchmark > run1.txt
cpu_parallel_benchmark > run2.txt
matmul_benchmark > matmul.txt
```

## Results

*Paste summary tables, not the raw output. Use the same cell style as BENCHMARKS.md: `run 1 / run 2`,
speedup = sequential time / parallel time. A good minimum: for every op, n = 1,000,000 (SIMD gain) and
n = 100,000 and 10,000,000 (1, 8, and best thread count).*

### SIMD: scalar vs sequential (n = 1,000,000)

| op | scalar ms | sequential ms | SIMD gain |
|---|---|---|---|

### Threads (n = 100,000 and n = 10,000,000)

| op | n | sequential ms | 1 thread | all threads | best of 2, 4, all |
|---|---|---|---|---|---|

### Matrix multiplication (GFLOP/s)

| n | scalar | sequential | 1 thread | 2 threads | 4 threads | all threads |
|---|---|---|---|---|---|---|

## Stability

*Output of `awk -f docs/benchmarks/compare_runs.awk run1.txt run2.txt`:*

```
```

*Your reading of it: were the runs close enough to trust? Which thread counts were noisy?*

## Observations

*What surprised you? Which ops gained from SIMD, which from threads, which from neither?*

## Problems

*Compile errors, failed checks, flags that did not work, anything a teammate should know.*
