# CPU benchmarks: sequential vs parallel

Results from the float32 CPU kernels (`add`, `sub`, `mul`, `div`, `neg`, `abs`,
`square`, `relu`, `sqrt`, `exp`, `sigmoid`, `dot`), measured on each machine we have
so the numbers can be compared. Keep the build and run conditions below the same on every machine,
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

The baseline is the plain sequential kernel (`src/backends/cpu/sequential.cpp`: one loop, no OpenMP).
Cells show run 1 / run 2 from two complete runs of all 12 operations with the full thread range.
"12 threads" requests all 12 hardware threads; "best" is the fastest measured thread count
among 1, 2, 4, 6, 8, and 12, in brackets. Thread affinity was not fixed, so six requested threads
do not establish one worker per physical core. The table uses the same four input sizes as
Machine 1; the full outputs below also include smaller sizes and cutoff checks.

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

- The best measured thread count depends on the operation and input size. More threads do not
  consistently improve performance; the paired best-thread counts above retain both runs' winners.
- At 100,000 elements, six threads are fastest in both runs for `add`, `mul`, and `sqrt`, while
  twelve are fastest in both runs for `neg`, `exp`, and `dot`.
- At 33,554,432 elements, twelve-thread speedups for the simple elementwise operations
  (`add`, `sub`, `mul`, `div`, `neg`, `abs`, `square`, `relu`) range from about
  0.90x to 1.13x. This is consistent with data movement becoming the main cost.
- At the same size, `exp` reaches 4.36-4.73x and `sigmoid` reaches 3.46-4.33x
  with twelve threads. Compare their four-thread rows in these same runs to assess the benefit
  of the additional workers; the earlier four-thread-only runs had different conditions.
- `sqrt` reaches 1.87-2.16x and `dot` reaches 2.10-2.38x with twelve threads at the
  largest size. Their best measured thread counts are shown separately in the results table.
- Rows at or below 32,768 elements use the serial fallback regardless of the requested thread
  count. They do not measure actual multithreaded execution. The existing cutoff is retained;
  these runs do not establish an optimal cutoff or a single best thread count for every workload.
- Both runs completed all 1,260 timing rows and passed every built-in result check. Elementwise
  outputs matched the sequential baseline bit for bit; `dot` passed its tolerance check because
  parallel summation can change the rounding order.

Reliability of these numbers: both complete runs are retained, including slower results.
Some rows vary between runs, and background activity was not controlled. In particular, `div`
at 1,000,000 elements with twelve threads takes 3.216 / 5.997 ms, versus 0.086 / 0.083 ms with
four threads; `abs` at the same size with twelve threads varies from 0.029 to 4.186 ms.
The cause of these severe slowdowns was not identified. Repeat those cases under quieter
conditions before treating them as a property of the operation or selecting a thread policy.
"Best" means fastest in that measurement; small differences and winners that change between runs
do not establish a reliable preference. Comparisons with Machine 1 also involve a different
compiler and thread count.
The summary selects the best row from printed throughput, which retains more precision than
the displayed milliseconds.

Benchmark method note: kernels write into preallocated buffers whose starting offsets are
staggered by 1 KB, using the same allocation arrangement as Machine 1's corrected benchmark.
Each measurement starts with a warmup, followed by nine timed samples whose median is reported,
with repeated calls for fast operations. Tensor allocation, dispatch, packing, and input
preparation are outside the timed calls. Unary operations and `dot` count 8 logical bytes per
element; binary elementwise operations count 12. `sqrt` uses nonnegative inputs; the other unary
operations use values in [-0.75, 0.75]. Times are printed to three decimal places; 0.000 ms means
below print precision, and speedups are calculated from unrounded timings.

Run with the Release CPU build and `MENSURFLOW_NATIVE=OFF`:

```bash
OMP_NUM_THREADS=12 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark
```

<details>
<summary>Full output: cpu_parallel_benchmark, Machine 3, full thread range, run 1</summary>

```text
Sequential vs parallel on the CPU (float32)
compiler: gcc 13.3.0, OpenMP: yes, max threads: 12, hardware threads: 12, parallel cutoff: 32768 elements
sequential = plain loop, no OpenMP. parallel = parallel kernel with N threads.
Speedup = sequential time / this time. Median of 9 samples.

== add ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   190.603     1.000x
        1000    parallel        1       0.000   192.588     1.010x
        1000    parallel        2       0.000   189.258     0.993x
        1000    parallel        4       0.000   188.572     0.989x
        1000    parallel        6       0.000   183.657     0.964x
        1000    parallel        8       0.000   171.414     0.899x
        1000    parallel       12       0.000   188.123     0.987x

       10000  sequential        -       0.001   136.174     1.000x
       10000    parallel        1       0.001   142.573     1.047x
       10000    parallel        2       0.001   142.098     1.044x
       10000    parallel        4       0.001   141.206     1.037x
       10000    parallel        6       0.001   141.948     1.042x
       10000    parallel        8       0.001   142.380     1.046x
       10000    parallel       12       0.001   142.192     1.044x

       32768  sequential        -       0.003   134.382     1.000x
       32768    parallel        1       0.003   131.993     0.982x
       32768    parallel        2       0.003   131.780     0.981x
       32768    parallel        4       0.003   129.536     0.964x
       32768    parallel        6       0.003   126.148     0.939x
       32768    parallel        8       0.003   125.558     0.934x
       32768    parallel       12       0.003   131.697     0.980x

       32769  sequential        -       0.003   130.593     1.000x
       32769    parallel        1       0.003   133.565     1.023x
       32769    parallel        2       0.002   158.496     1.214x
       32769    parallel        4       0.002   227.340     1.741x
       32769    parallel        6       0.001   279.600     2.141x
       32769    parallel        8       0.002   211.194     1.617x
       32769    parallel       12       0.003   127.962     0.980x

       40000  sequential        -       0.004   126.224     1.000x
       40000    parallel        1       0.004   124.799     0.989x
       40000    parallel        2       0.003   161.200     1.277x
       40000    parallel        4       0.002   254.599     2.017x
       40000    parallel        6       0.002   196.030     1.553x
       40000    parallel        8       0.002   212.961     1.687x
       40000    parallel       12       0.002   199.493     1.580x

       50000  sequential        -       0.005   120.323     1.000x
       50000    parallel        1       0.005   117.499     0.977x
       50000    parallel        2       0.004   167.615     1.393x
       50000    parallel        4       0.002   262.534     2.182x
       50000    parallel        6       0.002   295.059     2.452x
       50000    parallel        8       0.002   244.426     2.031x
       50000    parallel       12       0.004   142.069     1.181x

       75000  sequential        -       0.008   111.793     1.000x
       75000    parallel        1       0.008   111.895     1.001x
       75000    parallel        2       0.006   159.048     1.423x
       75000    parallel        4       0.003   275.192     2.462x
       75000    parallel        6       0.002   390.535     3.493x
       75000    parallel        8       0.004   255.183     2.283x
       75000    parallel       12       0.004   230.183     2.059x

      100000  sequential        -       0.010   119.223     1.000x
      100000    parallel        1       0.010   120.568     1.011x
      100000    parallel        2       0.007   168.374     1.412x
      100000    parallel        4       0.004   297.403     2.495x
      100000    parallel        6       0.003   415.314     3.484x
      100000    parallel        8       0.004   279.294     2.343x
      100000    parallel       12       0.005   224.317     1.881x

      200000  sequential        -       0.022   110.799     1.000x
      200000    parallel        1       0.020   122.579     1.106x
      200000    parallel        2       0.014   165.549     1.494x
      200000    parallel        4       0.008   282.860     2.553x
      200000    parallel        6       0.006   417.259     3.766x
      200000    parallel        8       0.008   292.153     2.637x
      200000    parallel       12       0.007   356.394     3.217x

      500000  sequential        -       0.051   117.187     1.000x
      500000    parallel        1       0.052   116.302     0.992x
      500000    parallel        2       0.032   187.312     1.598x
      500000    parallel        4       0.017   359.281     3.066x
      500000    parallel        6       0.014   433.854     3.702x
      500000    parallel        8       0.019   316.860     2.704x
      500000    parallel       12       0.014   434.220     3.705x

     1000000  sequential        -       0.184    65.070     1.000x
     1000000    parallel        1       0.187    64.002     0.984x
     1000000    parallel        2       0.099   121.733     1.871x
     1000000    parallel        4       0.059   205.060     3.151x
     1000000    parallel        6       0.044   275.631     4.236x
     1000000    parallel        8       0.047   255.927     3.933x
     1000000    parallel       12       0.049   245.743     3.777x

     2000000  sequential        -       0.878    27.334     1.000x
     2000000    parallel        1       0.840    28.567     1.045x
     2000000    parallel        2       0.685    35.032     1.282x
     2000000    parallel        4       0.708    33.883     1.240x
     2000000    parallel        6       0.772    31.071     1.137x
     2000000    parallel        8       0.733    32.736     1.198x
     2000000    parallel       12       0.803    29.879     1.093x

     5000000  sequential        -       3.112    19.282     1.000x
     5000000    parallel        1       3.016    19.894     1.032x
     5000000    parallel        2       2.977    20.155     1.045x
     5000000    parallel        4       2.990    20.065     1.041x
     5000000    parallel        6       2.921    20.544     1.065x
     5000000    parallel        8       2.905    20.655     1.071x
     5000000    parallel       12       3.344    17.944     0.931x

    10000000  sequential        -       6.632    18.094     1.000x
    10000000    parallel        1       6.199    19.359     1.070x
    10000000    parallel        2       5.964    20.122     1.112x
    10000000    parallel        4       5.876    20.422     1.129x
    10000000    parallel        6       6.202    19.349     1.069x
    10000000    parallel        8       6.186    19.399     1.072x
    10000000    parallel       12       7.019    17.097     0.945x

    33554432  sequential        -      23.453    17.168     1.000x
    33554432    parallel        1      23.339    17.252     1.005x
    33554432    parallel        2      21.029    19.147     1.115x
    33554432    parallel        4      22.202    18.136     1.056x
    33554432    parallel        6      23.700    16.990     0.990x
    33554432    parallel        8      23.596    17.064     0.994x
    33554432    parallel       12      24.960    16.132     0.940x


== sub ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   184.923     1.000x
        1000    parallel        1       0.000   184.361     0.997x
        1000    parallel        2       0.000   184.773     0.999x
        1000    parallel        4       0.000   180.900     0.978x
        1000    parallel        6       0.000   186.411     1.008x
        1000    parallel        8       0.000   171.765     0.929x
        1000    parallel       12       0.000   185.577     1.004x

       10000  sequential        -       0.001   132.460     1.000x
       10000    parallel        1       0.001   136.432     1.030x
       10000    parallel        2       0.001   136.520     1.031x
       10000    parallel        4       0.001   137.629     1.039x
       10000    parallel        6       0.001   135.409     1.022x
       10000    parallel        8       0.001   137.179     1.036x
       10000    parallel       12       0.001   137.542     1.038x

       32768  sequential        -       0.003   130.412     1.000x
       32768    parallel        1       0.003   130.909     1.004x
       32768    parallel        2       0.003   130.905     1.004x
       32768    parallel        4       0.003   130.889     1.004x
       32768    parallel        6       0.003   130.925     1.004x
       32768    parallel        8       0.003   131.036     1.005x
       32768    parallel       12       0.003   131.850     1.011x

       32769  sequential        -       0.003   130.194     1.000x
       32769    parallel        1       0.003   129.676     0.996x
       32769    parallel        2       0.003   154.944     1.190x
       32769    parallel        4       0.002   224.325     1.723x
       32769    parallel        6       0.002   253.123     1.944x
       32769    parallel        8       0.002   195.234     1.500x
       32769    parallel       12       0.002   195.071     1.498x

       40000  sequential        -       0.004   130.571     1.000x
       40000    parallel        1       0.004   130.257     0.998x
       40000    parallel        2       0.003   160.486     1.229x
       40000    parallel        4       0.002   242.773     1.859x
       40000    parallel        6       0.002   277.342     2.124x
       40000    parallel        8       0.002   215.254     1.649x
       40000    parallel       12       0.002   229.240     1.756x

       50000  sequential        -       0.005   127.299     1.000x
       50000    parallel        1       0.005   123.876     0.973x
       50000    parallel        2       0.004   166.309     1.306x
       50000    parallel        4       0.002   273.396     2.148x
       50000    parallel        6       0.003   188.303     1.479x
       50000    parallel        8       0.003   225.078     1.768x
       50000    parallel       12       0.002   247.458     1.944x

       75000  sequential        -       0.007   125.174     1.000x
       75000    parallel        1       0.007   121.723     0.972x
       75000    parallel        2       0.006   160.482     1.282x
       75000    parallel        4       0.003   289.205     2.310x
       75000    parallel        6       0.003   345.778     2.762x
       75000    parallel        8       0.004   256.918     2.052x
       75000    parallel       12       0.003   284.380     2.272x

      100000  sequential        -       0.009   126.417     1.000x
      100000    parallel        1       0.011   109.727     0.868x
      100000    parallel        2       0.008   159.104     1.259x
      100000    parallel        4       0.004   300.749     2.379x
      100000    parallel        6       0.005   226.972     1.795x
      100000    parallel        8       0.004   274.485     2.171x
      100000    parallel       12       0.004   308.160     2.438x

      200000  sequential        -       0.020   122.043     1.000x
      200000    parallel        1       0.020   122.239     1.002x
      200000    parallel        2       0.015   162.173     1.329x
      200000    parallel        4       0.008   298.642     2.447x
      200000    parallel        6       0.006   393.033     3.220x
      200000    parallel        8       0.008   283.535     2.323x
      200000    parallel       12       0.005   453.772     3.718x

      500000  sequential        -       0.050   119.004     1.000x
      500000    parallel        1       0.050   120.223     1.010x
      500000    parallel        2       0.035   169.098     1.421x
      500000    parallel        4       0.022   269.093     2.261x
      500000    parallel        6       0.018   338.718     2.846x
      500000    parallel        8       0.021   282.574     2.374x
      500000    parallel       12       0.018   339.083     2.849x

     1000000  sequential        -       0.192    62.584     1.000x
     1000000    parallel        1       0.172    69.965     1.118x
     1000000    parallel        2       0.118   101.771     1.626x
     1000000    parallel        4       0.082   146.945     2.348x
     1000000    parallel        6       0.051   236.624     3.781x
     1000000    parallel        8       0.050   239.932     3.834x
     1000000    parallel       12       0.049   242.660     3.877x

     2000000  sequential        -       0.843    28.473     1.000x
     2000000    parallel        1       0.844    28.430     0.999x
     2000000    parallel        2       0.683    35.116     1.233x
     2000000    parallel        4       0.679    35.321     1.241x
     2000000    parallel        6       0.676    35.490     1.246x
     2000000    parallel        8       0.693    34.613     1.216x
     2000000    parallel       12       0.968    24.789     0.871x

     5000000  sequential        -       3.530    16.995     1.000x
     5000000    parallel        1       3.523    17.033     1.002x
     5000000    parallel        2       3.200    18.748     1.103x
     5000000    parallel        4       3.133    19.148     1.127x
     5000000    parallel        6       3.137    19.126     1.125x
     5000000    parallel        8       3.042    19.725     1.161x
     5000000    parallel       12       3.518    17.055     1.003x

    10000000  sequential        -       6.892    17.412     1.000x
    10000000    parallel        1       6.857    17.500     1.005x
    10000000    parallel        2       6.229    19.266     1.106x
    10000000    parallel        4       6.530    18.376     1.055x
    10000000    parallel        6       6.541    18.346     1.054x
    10000000    parallel        8       6.135    19.561     1.123x
    10000000    parallel       12       6.464    18.563     1.066x

    33554432  sequential        -      23.392    17.213     1.000x
    33554432    parallel        1      23.828    16.898     0.982x
    33554432    parallel        2      20.852    19.310     1.122x
    33554432    parallel        4      21.029    19.147     1.112x
    33554432    parallel        6      20.925    19.243     1.118x
    33554432    parallel        8      21.176    19.015     1.105x
    33554432    parallel       12      22.450    17.936     1.042x


== mul ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   193.369     1.000x
        1000    parallel        1       0.000   186.635     0.965x
        1000    parallel        2       0.000   186.074     0.962x
        1000    parallel        4       0.000   186.832     0.966x
        1000    parallel        6       0.000   186.931     0.967x
        1000    parallel        8       0.000   187.256     0.968x
        1000    parallel       12       0.000   187.499     0.970x

       10000  sequential        -       0.001   132.097     1.000x
       10000    parallel        1       0.001   139.930     1.059x
       10000    parallel        2       0.001   142.625     1.080x
       10000    parallel        4       0.001   142.472     1.079x
       10000    parallel        6       0.001   142.692     1.080x
       10000    parallel        8       0.001   142.697     1.080x
       10000    parallel       12       0.001   143.126     1.083x

       32768  sequential        -       0.003   132.324     1.000x
       32768    parallel        1       0.003   136.641     1.033x
       32768    parallel        2       0.003   136.851     1.034x
       32768    parallel        4       0.003   135.938     1.027x
       32768    parallel        6       0.003   137.137     1.036x
       32768    parallel        8       0.003   136.322     1.030x
       32768    parallel       12       0.003   136.657     1.033x

       32769  sequential        -       0.003   132.803     1.000x
       32769    parallel        1       0.003   135.825     1.023x
       32769    parallel        2       0.002   161.178     1.214x
       32769    parallel        4       0.002   232.937     1.754x
       32769    parallel        6       0.002   252.032     1.898x
       32769    parallel        8       0.002   198.509     1.495x
       32769    parallel       12       0.002   198.557     1.495x

       40000  sequential        -       0.004   132.158     1.000x
       40000    parallel        1       0.004   124.620     0.943x
       40000    parallel        2       0.003   164.154     1.242x
       40000    parallel        4       0.002   249.186     1.886x
       40000    parallel        6       0.002   264.471     2.001x
       40000    parallel        8       0.002   211.098     1.597x
       40000    parallel       12       0.002   256.273     1.939x

       50000  sequential        -       0.005   128.146     1.000x
       50000    parallel        1       0.005   126.189     0.985x
       50000    parallel        2       0.004   163.012     1.272x
       50000    parallel        4       0.002   262.669     2.050x
       50000    parallel        6       0.002   302.933     2.364x
       50000    parallel        8       0.003   229.748     1.793x
       50000    parallel       12       0.003   237.792     1.856x

       75000  sequential        -       0.007   124.341     1.000x
       75000    parallel        1       0.007   121.171     0.975x
       75000    parallel        2       0.005   174.319     1.402x
       75000    parallel        4       0.003   285.059     2.293x
       75000    parallel        6       0.003   341.227     2.744x
       75000    parallel        8       0.004   254.529     2.047x
       75000    parallel       12       0.004   253.734     2.041x

      100000  sequential        -       0.009   126.999     1.000x
      100000    parallel        1       0.010   119.621     0.942x
      100000    parallel        2       0.008   159.043     1.252x
      100000    parallel        4       0.004   293.993     2.315x
      100000    parallel        6       0.003   359.024     2.827x
      100000    parallel        8       0.004   271.530     2.138x
      100000    parallel       12       0.004   300.596     2.367x

      200000  sequential        -       0.019   123.829     1.000x
      200000    parallel        1       0.020   119.988     0.969x
      200000    parallel        2       0.013   178.412     1.441x
      200000    parallel        4       0.009   267.664     2.162x
      200000    parallel        6       0.011   222.345     1.796x
      200000    parallel        8       0.009   280.472     2.265x
      200000    parallel       12       0.008   301.891     2.438x

      500000  sequential        -       0.052   114.849     1.000x
      500000    parallel        1       0.050   120.232     1.047x
      500000    parallel        2       0.032   187.613     1.634x
      500000    parallel        4       0.021   282.847     2.463x
      500000    parallel        6       0.018   338.667     2.949x
      500000    parallel        8       0.021   287.729     2.505x
      500000    parallel       12       1.423     4.217     0.037x

     1000000  sequential        -       0.210    57.133     1.000x
     1000000    parallel        1       0.189    63.410     1.110x
     1000000    parallel        2       0.116   103.492     1.811x
     1000000    parallel        4       0.062   194.583     3.406x
     1000000    parallel        6       0.047   256.925     4.497x
     1000000    parallel        8       0.050   240.014     4.201x
     1000000    parallel       12       0.049   246.147     4.308x

     2000000  sequential        -       0.910    26.359     1.000x
     2000000    parallel        1       0.854    28.109     1.066x
     2000000    parallel        2       0.722    33.228     1.261x
     2000000    parallel        4       0.784    30.617     1.162x
     2000000    parallel        6       0.826    29.051     1.102x
     2000000    parallel        8       0.861    27.870     1.057x
     2000000    parallel       12       0.822    29.190     1.107x

     5000000  sequential        -       3.190    18.806     1.000x
     5000000    parallel        1       3.108    19.302     1.026x
     5000000    parallel        2       2.903    20.670     1.099x
     5000000    parallel        4       2.781    21.577     1.147x
     5000000    parallel        6       2.868    20.923     1.113x
     5000000    parallel        8       3.049    19.681     1.046x
     5000000    parallel       12       3.006    19.958     1.061x

    10000000  sequential        -       6.467    18.556     1.000x
    10000000    parallel        1       6.263    19.159     1.033x
    10000000    parallel        2       5.926    20.248     1.091x
    10000000    parallel        4       5.912    20.299     1.094x
    10000000    parallel        6       6.046    19.849     1.070x
    10000000    parallel        8       6.074    19.756     1.065x
    10000000    parallel       12       6.473    18.538     0.999x

    33554432  sequential        -      23.673    17.009     1.000x
    33554432    parallel        1      23.603    17.059     1.003x
    33554432    parallel        2      24.181    16.652     0.979x
    33554432    parallel        4      22.210    18.129     1.066x
    33554432    parallel        6      22.512    17.886     1.052x
    33554432    parallel        8      22.589    17.826     1.048x
    33554432    parallel       12      25.590    15.735     0.925x


== div ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    61.191     1.000x
        1000    parallel        1       0.000    62.162     1.016x
        1000    parallel        2       0.000    62.223     1.017x
        1000    parallel        4       0.000    62.187     1.016x
        1000    parallel        6       0.000    62.388     1.020x
        1000    parallel        8       0.000    62.434     1.020x
        1000    parallel       12       0.000    62.300     1.018x

       10000  sequential        -       0.002    63.816     1.000x
       10000    parallel        1       0.002    62.346     0.977x
       10000    parallel        2       0.002    62.544     0.980x
       10000    parallel        4       0.002    63.475     0.995x
       10000    parallel        6       0.002    62.587     0.981x
       10000    parallel        8       0.002    62.369     0.977x
       10000    parallel       12       0.002    62.259     0.976x

       32768  sequential        -       0.006    62.772     1.000x
       32768    parallel        1       0.006    64.513     1.028x
       32768    parallel        2       0.006    64.772     1.032x
       32768    parallel        4       0.006    64.270     1.024x
       32768    parallel        6       0.006    64.460     1.027x
       32768    parallel        8       0.006    63.813     1.017x
       32768    parallel       12       0.006    64.000     1.020x

       32769  sequential        -       0.006    61.953     1.000x
       32769    parallel        1       0.006    63.907     1.032x
       32769    parallel        2       0.004   103.279     1.667x
       32769    parallel        4       0.002   176.251     2.845x
       32769    parallel        6       0.002   206.366     3.331x
       32769    parallel        8       0.002   159.900     2.581x
       32769    parallel       12       0.002   178.705     2.885x

       40000  sequential        -       0.008    62.226     1.000x
       40000    parallel        1       0.008    63.838     1.026x
       40000    parallel        2       0.004   111.848     1.797x
       40000    parallel        4       0.003   182.439     2.932x
       40000    parallel        6       0.002   219.832     3.533x
       40000    parallel        8       0.003   170.716     2.744x
       40000    parallel       12       0.002   193.464     3.109x

       50000  sequential        -       0.010    62.381     1.000x
       50000    parallel        1       0.010    62.073     0.995x
       50000    parallel        2       0.006   108.392     1.738x
       50000    parallel        4       0.003   185.995     2.982x
       50000    parallel        6       0.003   197.660     3.169x
       50000    parallel        8       0.003   176.832     2.835x
       50000    parallel       12       0.003   190.608     3.056x

       75000  sequential        -       0.015    61.823     1.000x
       75000    parallel        1       0.015    61.981     1.003x
       75000    parallel        2       0.008   114.322     1.849x
       75000    parallel        4       0.004   203.466     3.291x
       75000    parallel        6       0.005   197.446     3.194x
       75000    parallel        8       0.005   186.765     3.021x
       75000    parallel       12       4.999     0.180     0.003x

      100000  sequential        -       0.022    53.809     1.000x
      100000    parallel        1       0.023    51.131     0.950x
      100000    parallel        2       0.019    63.464     1.179x
      100000    parallel        4       0.006   198.299     3.685x
      100000    parallel        6       0.005   260.715     4.845x
      100000    parallel        8       0.006   197.844     3.677x
      100000    parallel       12       0.005   220.836     4.104x

      200000  sequential        -       0.039    61.847     1.000x
      200000    parallel        1       0.039    61.823     1.000x
      200000    parallel        2       0.020   122.349     1.978x
      200000    parallel        4       0.012   197.605     3.195x
      200000    parallel        6       0.009   277.983     4.495x
      200000    parallel        8       0.012   203.639     3.293x
      200000    parallel       12       0.011   214.266     3.464x

      500000  sequential        -       0.098    61.465     1.000x
      500000    parallel        1       0.101    59.420     0.967x
      500000    parallel        2       0.050   121.080     1.970x
      500000    parallel        4       0.029   206.646     3.362x
      500000    parallel        6       0.023   262.184     4.266x
      500000    parallel        8       0.029   207.916     3.383x
      500000    parallel       12       0.037   163.422     2.659x

     1000000  sequential        -       0.278    43.124     1.000x
     1000000    parallel        1       0.255    47.051     1.091x
     1000000    parallel        2       0.153    78.518     1.821x
     1000000    parallel        4       0.086   138.960     3.222x
     1000000    parallel        6       0.085   140.645     3.261x
     1000000    parallel        8       0.064   187.138     4.340x
     1000000    parallel       12       3.216     3.731     0.087x

     2000000  sequential        -       1.076    22.298     1.000x
     2000000    parallel        1       0.954    25.169     1.129x
     2000000    parallel        2       0.747    32.109     1.440x
     2000000    parallel        4       0.748    32.099     1.440x
     2000000    parallel        6       0.827    29.027     1.302x
     2000000    parallel        8       0.770    31.183     1.398x
     2000000    parallel       12       5.975     4.016     0.180x

     5000000  sequential        -       3.290    18.235     1.000x
     5000000    parallel        1       3.109    19.299     1.058x
     5000000    parallel        2       2.981    20.126     1.104x
     5000000    parallel        4       2.956    20.297     1.113x
     5000000    parallel        6       3.047    19.694     1.080x
     5000000    parallel        8       2.982    20.122     1.104x
     5000000    parallel       12       5.997    10.006     0.549x

    10000000  sequential        -       7.081    16.948     1.000x
    10000000    parallel        1       6.466    18.560     1.095x
    10000000    parallel        2       6.167    19.459     1.148x
    10000000    parallel        4       7.002    17.137     1.011x
    10000000    parallel        6       6.612    18.150     1.071x
    10000000    parallel        8       6.574    18.254     1.077x
    10000000    parallel       12       9.130    13.143     0.776x

    33554432  sequential        -      23.523    17.117     1.000x
    33554432    parallel        1      23.286    17.292     1.010x
    33554432    parallel        2      20.885    19.279     1.126x
    33554432    parallel        4      20.961    19.209     1.122x
    33554432    parallel        6      21.236    18.961     1.108x
    33554432    parallel        8      20.978    19.194     1.121x
    33554432    parallel       12      21.831    18.444     1.078x


== neg ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   128.597     1.000x
        1000    parallel        1       0.000   127.238     0.989x
        1000    parallel        2       0.000   128.585     1.000x
        1000    parallel        4       0.000   127.517     0.992x
        1000    parallel        6       0.000   127.596     0.992x
        1000    parallel        8       0.000   128.427     0.999x
        1000    parallel       12       0.000   127.078     0.988x

       10000  sequential        -       0.001   130.630     1.000x
       10000    parallel        1       0.001   136.532     1.045x
       10000    parallel        2       0.001   134.805     1.032x
       10000    parallel        4       0.001   134.506     1.030x
       10000    parallel        6       0.001   134.054     1.026x
       10000    parallel        8       0.001   132.384     1.013x
       10000    parallel       12       0.001   133.700     1.023x

       32768  sequential        -       0.002   134.584     1.000x
       32768    parallel        1       0.002   124.098     0.922x
       32768    parallel        2       0.002   126.582     0.941x
       32768    parallel        4       0.002   127.303     0.946x
       32768    parallel        6       0.002   127.575     0.948x
       32768    parallel        8       0.002   127.002     0.944x
       32768    parallel       12       0.002   127.577     0.948x

       32769  sequential        -       0.002   135.167     1.000x
       32769    parallel        1       0.002   126.236     0.934x
       32769    parallel        2       0.002   106.128     0.785x
       32769    parallel        4       0.002   161.489     1.195x
       32769    parallel        6       0.001   178.320     1.319x
       32769    parallel        8       0.002   143.051     1.058x
       32769    parallel       12       0.002   147.770     1.093x

       40000  sequential        -       0.002   131.927     1.000x
       40000    parallel        1       0.002   133.131     1.009x
       40000    parallel        2       0.003   112.680     0.854x
       40000    parallel        4       0.002   164.736     1.249x
       40000    parallel        6       0.003   127.727     0.968x
       40000    parallel        8       0.002   141.119     1.070x
       40000    parallel       12       6.003     0.053     0.000x

       50000  sequential        -       0.003   123.890     1.000x
       50000    parallel        1       0.003   121.184     0.978x
       50000    parallel        2       0.004   110.798     0.894x
       50000    parallel        4       0.002   167.149     1.349x
       50000    parallel        6       0.003   132.497     1.069x
       50000    parallel        8       0.003   149.832     1.209x
       50000    parallel       12       5.998     0.067     0.001x

       75000  sequential        -       0.005   116.094     1.000x
       75000    parallel        1       0.005   111.069     0.957x
       75000    parallel        2       0.006    97.626     0.841x
       75000    parallel        4       0.003   179.112     1.543x
       75000    parallel        6       0.004   142.615     1.228x
       75000    parallel        8       0.004   165.235     1.423x
       75000    parallel       12       0.002   253.218     2.181x

      100000  sequential        -       0.007   107.208     1.000x
      100000    parallel        1       0.008    96.499     0.900x
      100000    parallel        2       0.008   104.388     0.974x
      100000    parallel        4       0.005   165.540     1.544x
      100000    parallel        6       0.006   136.962     1.278x
      100000    parallel        8       0.004   181.595     1.694x
      100000    parallel       12       0.003   265.252     2.474x

      200000  sequential        -       0.014   111.476     1.000x
      200000    parallel        1       0.015   109.650     0.984x
      200000    parallel        2       0.014   116.512     1.045x
      200000    parallel        4       0.008   199.706     1.791x
      200000    parallel        6       0.006   256.992     2.305x
      200000    parallel        8       0.008   208.708     1.872x
      200000    parallel       12       0.006   246.935     2.215x

      500000  sequential        -       0.033   120.551     1.000x
      500000    parallel        1       0.035   114.819     0.952x
      500000    parallel        2       0.037   108.355     0.899x
      500000    parallel        4       0.021   192.959     1.601x
      500000    parallel        6       0.015   263.267     2.184x
      500000    parallel        8       0.018   217.263     1.802x
      500000    parallel       12       0.019   215.665     1.789x

     1000000  sequential        -       0.088    91.038     1.000x
     1000000    parallel        1       0.085    93.941     1.032x
     1000000    parallel        2       0.064   125.630     1.380x
     1000000    parallel        4       0.038   208.402     2.289x
     1000000    parallel        6       0.050   161.341     1.772x
     1000000    parallel        8       0.038   210.128     2.308x
     1000000    parallel       12       0.062   128.489     1.411x

     2000000  sequential        -       0.575    27.808     1.000x
     2000000    parallel        1       0.550    29.088     1.046x
     2000000    parallel        2       0.310    51.640     1.857x
     2000000    parallel        4       0.229    69.897     2.514x
     2000000    parallel        6       0.190    84.246     3.030x
     2000000    parallel        8       0.206    77.798     2.798x
     2000000    parallel       12       0.257    62.338     2.242x

     5000000  sequential        -       2.419    16.538     1.000x
     5000000    parallel        1       2.497    16.022     0.969x
     5000000    parallel        2       2.125    18.819     1.138x
     5000000    parallel        4       2.191    18.255     1.104x
     5000000    parallel        6       2.002    19.977     1.208x
     5000000    parallel        8       2.213    18.075     1.093x
     5000000    parallel       12       1.965    20.351     1.231x

    10000000  sequential        -       4.955    16.146     1.000x
    10000000    parallel        1       4.996    16.012     0.992x
    10000000    parallel        2       4.389    18.228     1.129x
    10000000    parallel        4       4.507    17.751     1.099x
    10000000    parallel        6       4.520    17.700     1.096x
    10000000    parallel        8       4.645    17.222     1.067x
    10000000    parallel       12       4.677    17.106     1.059x

    33554432  sequential        -      17.835    15.051     1.000x
    33554432    parallel        1      18.289    14.678     0.975x
    33554432    parallel        2      18.575    14.452     0.960x
    33554432    parallel        4      17.581    15.268     1.014x
    33554432    parallel        6      16.553    16.217     1.077x
    33554432    parallel        8      16.600    16.171     1.074x
    33554432    parallel       12      19.688    13.634     0.906x


== abs ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    64.308     1.000x
        1000    parallel        1       0.000   124.647     1.938x
        1000    parallel        2       0.000   125.529     1.952x
        1000    parallel        4       0.000   125.033     1.944x
        1000    parallel        6       0.000   124.135     1.930x
        1000    parallel        8       0.000   125.746     1.955x
        1000    parallel       12       0.000   126.226     1.963x

       10000  sequential        -       0.001    69.887     1.000x
       10000    parallel        1       0.001   133.565     1.911x
       10000    parallel        2       0.001   134.161     1.920x
       10000    parallel        4       0.001   134.098     1.919x
       10000    parallel        6       0.001   134.046     1.918x
       10000    parallel        8       0.001   133.386     1.909x
       10000    parallel       12       0.001   133.739     1.914x

       32768  sequential        -       0.004    69.857     1.000x
       32768    parallel        1       0.002   129.167     1.849x
       32768    parallel        2       0.002   129.115     1.848x
       32768    parallel        4       0.002   126.726     1.814x
       32768    parallel        6       0.002   127.086     1.819x
       32768    parallel        8       0.002   127.551     1.826x
       32768    parallel       12       0.002   128.757     1.843x

       32769  sequential        -       0.004    69.543     1.000x
       32769    parallel        1       0.002   127.994     1.841x
       32769    parallel        2       0.002   163.618     2.353x
       32769    parallel        4       0.001   221.191     3.181x
       32769    parallel        6       0.001   214.853     3.090x
       32769    parallel        8       0.001   183.746     2.642x
       32769    parallel       12       0.002   153.193     2.203x

       40000  sequential        -       0.005    69.135     1.000x
       40000    parallel        1       0.002   129.343     1.871x
       40000    parallel        2       0.002   178.186     2.577x
       40000    parallel        4       0.001   249.444     3.608x
       40000    parallel        6       0.001   247.286     3.577x
       40000    parallel        8       0.002   210.928     3.051x
       40000    parallel       12       0.002   194.178     2.809x

       50000  sequential        -       0.006    67.753     1.000x
       50000    parallel        1       0.003   124.257     1.834x
       50000    parallel        2       0.002   192.128     2.836x
       50000    parallel        4       0.001   281.466     4.154x
       50000    parallel        6       0.001   300.215     4.431x
       50000    parallel        8       0.002   238.202     3.516x
       50000    parallel       12       0.002   214.130     3.160x

       75000  sequential        -       0.009    70.397     1.000x
       75000    parallel        1       0.005   118.354     1.681x
       75000    parallel        2       0.003   192.975     2.741x
       75000    parallel        4       0.002   303.364     4.309x
       75000    parallel        6       0.002   353.761     5.025x
       75000    parallel        8       0.002   285.507     4.056x
       75000    parallel       12       0.002   297.431     4.225x

      100000  sequential        -       0.011    70.240     1.000x
      100000    parallel        1       0.007   118.540     1.688x
      100000    parallel        2       0.004   205.077     2.920x
      100000    parallel        4       0.002   333.622     4.750x
      100000    parallel        6       0.002   388.399     5.530x
      100000    parallel        8       0.003   306.368     4.362x
      100000    parallel       12       0.003   319.693     4.551x

      200000  sequential        -       0.023    69.377     1.000x
      200000    parallel        1       0.014   118.466     1.708x
      200000    parallel        2       0.009   182.004     2.623x
      200000    parallel        4       0.005   326.086     4.700x
      200000    parallel        6       0.004   454.969     6.558x
      200000    parallel        8       0.005   326.616     4.708x
      200000    parallel       12       0.004   381.405     5.498x

      500000  sequential        -       0.062    64.193     1.000x
      500000    parallel        1       0.040    98.991     1.542x
      500000    parallel        2       0.022   182.744     2.847x
      500000    parallel        4       0.014   290.889     4.531x
      500000    parallel        6       0.012   324.529     5.056x
      500000    parallel        8       0.012   329.547     5.134x
      500000    parallel       12       0.012   333.555     5.196x

     1000000  sequential        -       0.124    64.578     1.000x
     1000000    parallel        1       0.083    96.587     1.496x
     1000000    parallel        2       0.048   165.196     2.558x
     1000000    parallel        4       0.029   275.504     4.266x
     1000000    parallel        6       0.022   359.588     5.568x
     1000000    parallel        8       0.027   301.758     4.673x
     1000000    parallel       12       0.029   274.581     4.252x

     2000000  sequential        -       0.528    30.280     1.000x
     2000000    parallel        1       0.484    33.060     1.092x
     2000000    parallel        2       0.311    51.442     1.699x
     2000000    parallel        4       0.223    71.767     2.370x
     2000000    parallel        6       0.205    78.005     2.576x
     2000000    parallel        8       0.222    72.101     2.381x
     2000000    parallel       12       0.276    57.869     1.911x

     5000000  sequential        -       2.529    15.820     1.000x
     5000000    parallel        1       2.252    17.760     1.123x
     5000000    parallel        2       1.955    20.460     1.293x
     5000000    parallel        4       2.035    19.658     1.243x
     5000000    parallel        6       2.008    19.922     1.259x
     5000000    parallel        8       2.004    19.963     1.262x
     5000000    parallel       12       2.012    19.878     1.257x

    10000000  sequential        -       5.076    15.762     1.000x
    10000000    parallel        1       4.989    16.036     1.017x
    10000000    parallel        2       4.440    18.017     1.143x
    10000000    parallel        4       4.537    17.634     1.119x
    10000000    parallel        6       4.882    16.387     1.040x
    10000000    parallel        8       4.623    17.304     1.098x
    10000000    parallel       12       6.541    12.231     0.776x

    33554432  sequential        -      18.028    14.890     1.000x
    33554432    parallel        1      19.329    13.888     0.933x
    33554432    parallel        2      16.754    16.022     1.076x
    33554432    parallel        4      17.007    15.784     1.060x
    33554432    parallel        6      16.727    16.048     1.078x
    33554432    parallel        8      16.594    16.177     1.086x
    33554432    parallel       12      16.294    16.474     1.106x


== square ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   126.552     1.000x
        1000    parallel        1       0.000   124.096     0.981x
        1000    parallel        2       0.000   124.282     0.982x
        1000    parallel        4       0.000   125.181     0.989x
        1000    parallel        6       0.000   125.451     0.991x
        1000    parallel        8       0.000   125.760     0.994x
        1000    parallel       12       0.000   125.516     0.992x

       10000  sequential        -       0.001   134.680     1.000x
       10000    parallel        1       0.001   134.061     0.995x
       10000    parallel        2       0.001   133.795     0.993x
       10000    parallel        4       0.001   133.998     0.995x
       10000    parallel        6       0.001   132.534     0.984x
       10000    parallel        8       0.001   131.728     0.978x
       10000    parallel       12       0.001   133.494     0.991x

       32768  sequential        -       0.002   132.365     1.000x
       32768    parallel        1       0.002   133.983     1.012x
       32768    parallel        2       0.002   133.983     1.012x
       32768    parallel        4       0.002   133.473     1.008x
       32768    parallel        6       0.002   134.135     1.013x
       32768    parallel        8       0.002   133.279     1.007x
       32768    parallel       12       0.002   134.133     1.013x

       32769  sequential        -       0.002   134.799     1.000x
       32769    parallel        1       0.002   135.033     1.002x
       32769    parallel        2       0.002   170.800     1.267x
       32769    parallel        4       0.001   227.299     1.686x
       32769    parallel        6       0.001   238.139     1.767x
       32769    parallel        8       0.001   193.024     1.432x
       32769    parallel       12       0.002   170.517     1.265x

       40000  sequential        -       0.002   133.375     1.000x
       40000    parallel        1       0.002   137.877     1.034x
       40000    parallel        2       0.002   186.628     1.399x
       40000    parallel        4       0.001   248.546     1.864x
       40000    parallel        6       0.001   270.502     2.028x
       40000    parallel        8       0.002   212.652     1.594x
       40000    parallel       12       0.002   190.520     1.428x

       50000  sequential        -       0.003   128.732     1.000x
       50000    parallel        1       0.003   127.386     0.990x
       50000    parallel        2       0.002   193.600     1.504x
       50000    parallel        4       0.001   272.015     2.113x
       50000    parallel        6       0.001   298.570     2.319x
       50000    parallel        8       0.002   236.499     1.837x
       50000    parallel       12       0.002   230.617     1.791x

       75000  sequential        -       0.005   121.594     1.000x
       75000    parallel        1       0.005   116.011     0.954x
       75000    parallel        2       0.003   198.088     1.629x
       75000    parallel        4       0.002   320.837     2.639x
       75000    parallel        6       0.002   357.381     2.939x
       75000    parallel        8       0.002   273.721     2.251x
       75000    parallel       12       0.002   305.499     2.512x

      100000  sequential        -       0.007   122.178     1.000x
      100000    parallel        1       0.007   116.495     0.953x
      100000    parallel        2       0.004   199.537     1.633x
      100000    parallel        4       0.002   346.206     2.834x
      100000    parallel        6       0.002   402.388     3.293x
      100000    parallel        8       0.003   307.175     2.514x
      100000    parallel       12       0.002   333.093     2.726x

      200000  sequential        -       0.013   121.288     1.000x
      200000    parallel        1       0.014   117.719     0.971x
      200000    parallel        2       0.008   202.032     1.666x
      200000    parallel        4       0.008   204.048     1.682x
      200000    parallel        6       0.006   278.572     2.297x
      200000    parallel        8       0.005   339.952     2.803x
      200000    parallel       12       0.004   389.365     3.210x

      500000  sequential        -       0.033   121.472     1.000x
      500000    parallel        1       0.034   116.284     0.957x
      500000    parallel        2       0.020   200.812     1.653x
      500000    parallel        4       0.014   290.454     2.391x
      500000    parallel        6       0.010   383.388     3.156x
      500000    parallel        8       0.012   328.838     2.707x
      500000    parallel       12       0.013   317.246     2.612x

     1000000  sequential        -       0.095    84.353     1.000x
     1000000    parallel        1       0.098    81.491     0.966x
     1000000    parallel        2       0.049   164.232     1.947x
     1000000    parallel        4       0.028   287.260     3.405x
     1000000    parallel        6       0.024   338.483     4.013x
     1000000    parallel        8       0.026   304.203     3.606x
     1000000    parallel       12       0.026   307.524     3.646x

     2000000  sequential        -       0.524    30.536     1.000x
     2000000    parallel        1       0.704    22.721     0.744x
     2000000    parallel        2       0.327    48.923     1.602x
     2000000    parallel        4       0.235    68.131     2.231x
     2000000    parallel        6       0.213    75.181     2.462x
     2000000    parallel        8       0.219    73.207     2.397x
     2000000    parallel       12       0.248    64.396     2.109x

     5000000  sequential        -       2.283    17.519     1.000x
     5000000    parallel        1       2.259    17.703     1.011x
     5000000    parallel        2       2.088    19.153     1.093x
     5000000    parallel        4       1.977    20.233     1.155x
     5000000    parallel        6       2.292    17.453     0.996x
     5000000    parallel        8       2.201    18.176     1.038x
     5000000    parallel       12       6.970     5.739     0.328x

    10000000  sequential        -       5.172    15.468     1.000x
    10000000    parallel        1       5.331    15.005     0.970x
    10000000    parallel        2       4.687    17.067     1.103x
    10000000    parallel        4       4.904    16.314     1.055x
    10000000    parallel        6       4.778    16.745     1.083x
    10000000    parallel        8       4.603    17.380     1.124x
    10000000    parallel       12       5.221    15.321     0.991x

    33554432  sequential        -      18.154    14.786     1.000x
    33554432    parallel        1      18.239    14.718     0.995x
    33554432    parallel        2      14.899    18.017     1.218x
    33554432    parallel        4      16.027    16.749     1.133x
    33554432    parallel        6      16.013    16.764     1.134x
    33554432    parallel        8      15.501    17.317     1.171x
    33554432    parallel       12      16.123    16.649     1.126x


== relu ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   110.057     1.000x
        1000    parallel        1       0.000    66.539     0.605x
        1000    parallel        2       0.000    66.555     0.605x
        1000    parallel        4       0.000    66.679     0.606x
        1000    parallel        6       0.000    65.637     0.596x
        1000    parallel        8       0.000    65.352     0.594x
        1000    parallel       12       0.000    64.816     0.589x

       10000  sequential        -       0.001   100.923     1.000x
       10000    parallel        1       0.001    67.266     0.667x
       10000    parallel        2       0.001    66.986     0.664x
       10000    parallel        4       0.001    67.222     0.666x
       10000    parallel        6       0.001    67.432     0.668x
       10000    parallel        8       0.001    68.174     0.676x
       10000    parallel       12       0.001    68.131     0.675x

       32768  sequential        -       0.003   101.304     1.000x
       32768    parallel        1       0.004    68.424     0.675x
       32768    parallel        2       0.004    68.659     0.678x
       32768    parallel        4       0.004    68.678     0.678x
       32768    parallel        6       0.004    68.788     0.679x
       32768    parallel        8       0.004    69.239     0.683x
       32768    parallel       12       0.004    68.748     0.679x

       32769  sequential        -       0.003   102.688     1.000x
       32769    parallel        1       0.004    68.889     0.671x
       32769    parallel        2       0.002   107.398     1.046x
       32769    parallel        4       0.002   155.413     1.513x
       32769    parallel        6       0.001   178.839     1.742x
       32769    parallel        8       0.002   138.632     1.350x
       32769    parallel       12       0.002   140.790     1.371x

       40000  sequential        -       0.003   104.341     1.000x
       40000    parallel        1       0.005    68.743     0.659x
       40000    parallel        2       0.003   109.954     1.054x
       40000    parallel        4       0.002   169.545     1.625x
       40000    parallel        6       0.002   188.055     1.802x
       40000    parallel        8       0.002   148.416     1.422x
       40000    parallel       12       0.002   138.644     1.329x

       50000  sequential        -       0.004   100.322     1.000x
       50000    parallel        1       0.006    67.991     0.678x
       50000    parallel        2       0.004   111.963     1.116x
       50000    parallel        4       0.002   171.237     1.707x
       50000    parallel        6       0.002   205.261     2.046x
       50000    parallel        8       0.003   159.969     1.595x
       50000    parallel       12       0.002   178.271     1.777x

       75000  sequential        -       0.006   101.168     1.000x
       75000    parallel        1       0.009    65.500     0.647x
       75000    parallel        2       0.005   118.474     1.171x
       75000    parallel        4       0.003   195.930     1.937x
       75000    parallel        6       0.003   237.090     2.344x
       75000    parallel        8       0.003   179.226     1.772x
       75000    parallel       12       0.003   190.353     1.882x

      100000  sequential        -       0.008    94.339     1.000x
      100000    parallel        1       0.012    67.220     0.713x
      100000    parallel        2       0.006   123.910     1.313x
      100000    parallel        4       0.004   206.412     2.188x
      100000    parallel        6       0.003   256.246     2.716x
      100000    parallel        8       0.004   184.638     1.957x
      100000    parallel       12       0.004   198.708     2.106x

      200000  sequential        -       0.018    89.262     1.000x
      200000    parallel        1       0.024    67.191     0.753x
      200000    parallel        2       0.013   126.672     1.419x
      200000    parallel        4       0.008   209.286     2.345x
      200000    parallel        6       0.006   264.764     2.966x
      200000    parallel        8       0.008   204.747     2.294x
      200000    parallel       12       0.007   234.981     2.632x

      500000  sequential        -       0.042    95.961     1.000x
      500000    parallel        1       0.060    66.486     0.693x
      500000    parallel        2       0.031   128.551     1.340x
      500000    parallel        4       0.021   191.694     1.998x
      500000    parallel        6       0.016   248.982     2.595x
      500000    parallel        8       0.021   193.462     2.016x
      500000    parallel       12       5.998     0.667     0.007x

     1000000  sequential        -       0.105    76.108     1.000x
     1000000    parallel        1       0.148    53.887     0.708x
     1000000    parallel        2       0.063   126.929     1.668x
     1000000    parallel        4       0.043   185.177     2.433x
     1000000    parallel        6       0.031   258.328     3.394x
     1000000    parallel        8       0.039   204.257     2.684x
     1000000    parallel       12       0.039   204.264     2.684x

     2000000  sequential        -       0.690    23.183     1.000x
     2000000    parallel        1       0.766    20.878     0.901x
     2000000    parallel        2       0.338    47.369     2.043x
     2000000    parallel        4       0.266    60.176     2.596x
     2000000    parallel        6       0.219    72.980     3.148x
     2000000    parallel        8       0.178    89.970     3.881x
     2000000    parallel       12       6.000     2.667     0.115x

     5000000  sequential        -       2.252    17.765     1.000x
     5000000    parallel        1       2.241    17.848     1.005x
     5000000    parallel        2       1.988    20.116     1.132x
     5000000    parallel        4       2.044    19.573     1.102x
     5000000    parallel        6       2.054    19.479     1.096x
     5000000    parallel        8       1.958    20.432     1.150x
     5000000    parallel       12       1.987    20.134     1.133x

    10000000  sequential        -       4.875    16.409     1.000x
    10000000    parallel        1       5.077    15.756     0.960x
    10000000    parallel        2       4.474    17.880     1.090x
    10000000    parallel        4       4.448    17.984     1.096x
    10000000    parallel        6       4.660    17.168     1.046x
    10000000    parallel        8       4.818    16.603     1.012x
    10000000    parallel       12       6.089    13.139     0.801x

    33554432  sequential        -      17.168    15.636     1.000x
    33554432    parallel        1      18.513    14.500     0.927x
    33554432    parallel        2      14.983    17.916     1.146x
    33554432    parallel        4      15.293    17.552     1.123x
    33554432    parallel        6      16.226    16.543     1.058x
    33554432    parallel        8      15.778    17.013     1.088x
    33554432    parallel       12      16.029    16.747     1.071x


== sqrt ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001     7.231     1.000x
        1000    parallel        1       0.001     7.234     1.000x
        1000    parallel        2       0.001     7.234     1.000x
        1000    parallel        4       0.001     7.233     1.000x
        1000    parallel        6       0.001     7.226     0.999x
        1000    parallel        8       0.001     7.231     1.000x
        1000    parallel       12       0.001     7.232     1.000x

       10000  sequential        -       0.011     7.231     1.000x
       10000    parallel        1       0.011     7.236     1.001x
       10000    parallel        2       0.011     7.230     1.000x
       10000    parallel        4       0.011     7.224     0.999x
       10000    parallel        6       0.011     7.225     0.999x
       10000    parallel        8       0.011     7.235     1.000x
       10000    parallel       12       0.011     7.235     1.001x

       32768  sequential        -       0.036     7.231     1.000x
       32768    parallel        1       0.036     7.226     0.999x
       32768    parallel        2       0.036     7.229     1.000x
       32768    parallel        4       0.036     7.235     1.001x
       32768    parallel        6       0.036     7.228     1.000x
       32768    parallel        8       0.036     7.224     0.999x
       32768    parallel       12       0.036     7.222     0.999x

       32769  sequential        -       0.036     7.233     1.000x
       32769    parallel        1       0.036     7.226     0.999x
       32769    parallel        2       0.019    13.993     1.935x
       32769    parallel        4       0.010    26.905     3.720x
       32769    parallel        6       0.013    20.912     2.891x
       32769    parallel        8       0.010    26.915     3.721x
       32769    parallel       12       0.009    29.748     4.113x

       40000  sequential        -       0.044     7.212     1.000x
       40000    parallel        1       0.044     7.218     1.001x
       40000    parallel        2       0.023    14.041     1.947x
       40000    parallel        4       0.012    27.215     3.774x
       40000    parallel        6       0.008    37.983     5.267x
       40000    parallel        8       0.012    26.342     3.653x
       40000    parallel       12       0.010    33.263     4.612x

       50000  sequential        -       0.055     7.226     1.000x
       50000    parallel        1       0.055     7.222     0.999x
       50000    parallel        2       0.028    14.137     1.957x
       50000    parallel        4       0.015    27.534     3.811x
       50000    parallel        6       0.010    38.282     5.298x
       50000    parallel        8       0.015    26.639     3.687x
       50000    parallel       12       0.012    33.628     4.654x

       75000  sequential        -       0.083     7.216     1.000x
       75000    parallel        1       0.083     7.226     1.001x
       75000    parallel        2       0.042    14.224     1.971x
       75000    parallel        4       0.021    27.961     3.875x
       75000    parallel        6       0.015    39.214     5.434x
       75000    parallel        8       0.022    26.954     3.735x
       75000    parallel       12       0.016    36.461     5.053x

      100000  sequential        -       0.111     7.220     1.000x
      100000    parallel        1       0.111     7.219     1.000x
      100000    parallel        2       0.056    14.285     1.979x
      100000    parallel        4       0.028    28.151     3.899x
      100000    parallel        6       0.020    40.886     5.663x
      100000    parallel        8       0.029    27.457     3.803x
      100000    parallel       12       0.023    35.319     4.892x

      200000  sequential        -       0.223     7.174     1.000x
      200000    parallel        1       0.222     7.220     1.006x
      200000    parallel        2       0.112    14.330     1.998x
      200000    parallel        4       0.056    28.467     3.968x
      200000    parallel        6       0.039    40.625     5.663x
      200000    parallel        8       0.057    27.844     3.881x
      200000    parallel       12       0.042    38.183     5.323x

      500000  sequential        -       0.555     7.211     1.000x
      500000    parallel        1       0.554     7.217     1.001x
      500000    parallel        2       0.278    14.397     1.997x
      500000    parallel        4       0.147    27.231     3.776x
      500000    parallel        6       0.100    39.951     5.540x
      500000    parallel        8       0.148    27.045     3.751x
      500000    parallel       12       0.123    32.582     4.518x

     1000000  sequential        -       1.109     7.212     1.000x
     1000000    parallel        1       1.108     7.218     1.001x
     1000000    parallel        2       0.557    14.373     1.993x
     1000000    parallel        4       0.284    28.125     3.900x
     1000000    parallel        6       0.201    39.759     5.513x
     1000000    parallel        8       0.294    27.223     3.775x
     1000000    parallel       12       0.204    39.162     5.430x

     2000000  sequential        -       2.235     7.158     1.000x
     2000000    parallel        1       2.229     7.177     1.003x
     2000000    parallel        2       1.125    14.218     1.986x
     2000000    parallel        4       0.599    26.713     3.732x
     2000000    parallel        6       0.756    21.152     2.955x
     2000000    parallel        8       0.601    26.625     3.719x
     2000000    parallel       12       0.460    34.746     4.854x

     5000000  sequential        -       5.730     6.980     1.000x
     5000000    parallel        1       5.751     6.955     0.996x
     5000000    parallel        2       3.064    13.054     1.870x
     5000000    parallel        4       2.447    16.349     2.342x
     5000000    parallel        6       2.145    18.645     2.671x
     5000000    parallel        8       2.028    19.728     2.826x
     5000000    parallel       12       2.289    17.471     2.503x

    10000000  sequential        -      11.480     6.969     1.000x
    10000000    parallel        1      11.585     6.905     0.991x
    10000000    parallel        2       6.131    13.048     1.872x
    10000000    parallel        4       4.882    16.386     2.351x
    10000000    parallel        6       4.478    17.864     2.563x
    10000000    parallel        8       4.599    17.394     2.496x
    10000000    parallel       12       5.394    14.832     2.128x

    33554432  sequential        -      38.478     6.976     1.000x
    33554432    parallel        1      39.234     6.842     0.981x
    33554432    parallel        2      21.025    12.767     1.830x
    33554432    parallel        4      16.837    15.943     2.285x
    33554432    parallel        6      19.049    14.092     2.020x
    33554432    parallel        8      17.717    15.152     2.172x
    33554432    parallel       12      20.575    13.047     1.870x


== exp ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.003     3.191     1.000x
        1000    parallel        1       0.003     2.896     0.907x
        1000    parallel        2       0.003     2.902     0.909x
        1000    parallel        4       0.003     2.889     0.905x
        1000    parallel        6       0.003     2.845     0.891x
        1000    parallel        8       0.003     2.878     0.902x
        1000    parallel       12       0.003     2.931     0.918x

       10000  sequential        -       0.025     3.207     1.000x
       10000    parallel        1       0.027     2.939     0.916x
       10000    parallel        2       0.027     2.944     0.918x
       10000    parallel        4       0.027     2.943     0.918x
       10000    parallel        6       0.027     2.947     0.919x
       10000    parallel        8       0.027     2.938     0.916x
       10000    parallel       12       0.027     2.934     0.915x

       32768  sequential        -       0.082     3.207     1.000x
       32768    parallel        1       0.089     2.946     0.918x
       32768    parallel        2       0.089     2.938     0.916x
       32768    parallel        4       0.089     2.943     0.918x
       32768    parallel        6       0.089     2.941     0.917x
       32768    parallel        8       0.089     2.946     0.918x
       32768    parallel       12       0.089     2.938     0.916x

       32769  sequential        -       0.082     3.207     1.000x
       32769    parallel        1       0.089     2.946     0.918x
       32769    parallel        2       0.042     6.250     1.949x
       32769    parallel        4       0.023    11.241     3.505x
       32769    parallel        6       0.017    15.410     4.804x
       32769    parallel        8       0.019    14.082     4.390x
       32769    parallel       12       0.011    22.873     7.131x

       40000  sequential        -       0.100     3.213     1.000x
       40000    parallel        1       0.109     2.941     0.915x
       40000    parallel        2       0.051     6.299     1.960x
       40000    parallel        4       0.027    11.704     3.643x
       40000    parallel        6       0.021    15.373     4.784x
       40000    parallel        8       0.024    13.382     4.165x
       40000    parallel       12       0.018    17.728     5.517x

       50000  sequential        -       0.125     3.213     1.000x
       50000    parallel        1       0.136     2.943     0.916x
       50000    parallel        2       0.064     6.297     1.960x
       50000    parallel        4       0.034    11.808     3.676x
       50000    parallel        6       0.027    14.987     4.665x
       50000    parallel        8       0.028    14.140     4.401x
       50000    parallel       12       0.024    16.482     5.130x

       75000  sequential        -       0.187     3.213     1.000x
       75000    parallel        1       0.204     2.940     0.915x
       75000    parallel        2       0.095     6.325     1.969x
       75000    parallel        4       0.051    11.709     3.645x
       75000    parallel        6       0.039    15.479     4.818x
       75000    parallel        8       0.042    14.200     4.420x
       75000    parallel       12       0.037    16.087     5.007x

      100000  sequential        -       0.249     3.209     1.000x
      100000    parallel        1       0.272     2.940     0.916x
      100000    parallel        2       0.127     6.301     1.963x
      100000    parallel        4       0.068    11.740     3.658x
      100000    parallel        6       0.050    15.872     4.946x
      100000    parallel        8       0.056    14.178     4.418x
      100000    parallel       12       0.046    17.488     5.449x

      200000  sequential        -       0.488     3.277     1.000x
      200000    parallel        1       0.533     3.002     0.916x
      200000    parallel        2       0.253     6.319     1.928x
      200000    parallel        4       0.132    12.138     3.704x
      200000    parallel        6       0.104    15.350     4.683x
      200000    parallel        8       0.110    14.609     4.457x
      200000    parallel       12       0.096    16.640     5.077x

      500000  sequential        -       1.249     3.203     1.000x
      500000    parallel        1       1.364     2.933     0.916x
      500000    parallel        2       0.630     6.351     1.983x
      500000    parallel        4       0.335    11.926     3.724x
      500000    parallel        6       0.336    11.893     3.713x
      500000    parallel        8       0.278    14.409     4.499x
      500000    parallel       12       0.157    25.468     7.952x

     1000000  sequential        -       2.498     3.203     1.000x
     1000000    parallel        1       2.736     2.924     0.913x
     1000000    parallel        2       1.268     6.309     1.970x
     1000000    parallel        4       0.687    11.642     3.635x
     1000000    parallel        6       0.498    16.050     5.012x
     1000000    parallel        8       0.576    13.899     4.340x
     1000000    parallel       12       0.506    15.801     4.934x

     2000000  sequential        -       5.067     3.158     1.000x
     2000000    parallel        1       5.558     2.879     0.912x
     2000000    parallel        2       2.574     6.215     1.968x
     2000000    parallel        4       1.432    11.177     3.539x
     2000000    parallel        6       1.181    13.543     4.288x
     2000000    parallel        8       1.185    13.500     4.275x
     2000000    parallel       12       1.118    14.316     4.533x

     5000000  sequential        -      12.896     3.102     1.000x
     5000000    parallel        1      13.956     2.866     0.924x
     5000000    parallel        2       6.739     5.936     1.914x
     5000000    parallel        4       3.742    10.690     3.446x
     5000000    parallel        6       3.796    10.536     3.397x
     5000000    parallel        8       2.977    13.437     4.332x
     5000000    parallel       12       6.037     6.626     2.136x

    10000000  sequential        -      25.377     3.152     1.000x
    10000000    parallel        1      27.660     2.892     0.917x
    10000000    parallel        2      12.914     6.195     1.965x
    10000000    parallel        4       7.449    10.739     3.407x
    10000000    parallel        6       5.911    13.533     4.293x
    10000000    parallel        8       6.075    13.170     4.178x
    10000000    parallel       12       6.216    12.870     4.082x

    33554432  sequential        -      85.207     3.150     1.000x
    33554432    parallel        1      92.947     2.888     0.917x
    33554432    parallel        2      44.301     6.059     1.923x
    33554432    parallel        4      25.260    10.627     3.373x
    33554432    parallel        6      20.452    13.125     4.166x
    33554432    parallel        8      20.009    13.416     4.258x
    33554432    parallel       12      19.553    13.729     4.358x


== sigmoid ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.003     2.864     1.000x
        1000    parallel        1       0.003     2.603     0.909x
        1000    parallel        2       0.005     1.653     0.577x
        1000    parallel        4       0.005     1.649     0.576x
        1000    parallel        6       0.003     2.720     0.950x
        1000    parallel        8       0.003     2.817     0.984x
        1000    parallel       12       0.003     2.821     0.985x

       10000  sequential        -       0.027     2.918     1.000x
       10000    parallel        1       0.028     2.820     0.966x
       10000    parallel        2       0.028     2.836     0.972x
       10000    parallel        4       0.028     2.837     0.972x
       10000    parallel        6       0.028     2.836     0.972x
       10000    parallel        8       0.028     2.836     0.972x
       10000    parallel       12       0.028     2.836     0.972x

       32768  sequential        -       0.090     2.928     1.000x
       32768    parallel        1       0.092     2.837     0.969x
       32768    parallel        2       0.092     2.837     0.969x
       32768    parallel        4       0.092     2.838     0.969x
       32768    parallel        6       0.092     2.838     0.969x
       32768    parallel        8       0.093     2.833     0.968x
       32768    parallel       12       0.093     2.823     0.964x

       32769  sequential        -       0.090     2.926     1.000x
       32769    parallel        1       0.093     2.823     0.965x
       32769    parallel        2       0.047     5.561     1.900x
       32769    parallel        4       0.025    10.332     3.531x
       32769    parallel        6       0.020    13.422     4.587x
       32769    parallel        8       0.023    11.185     3.823x
       32769    parallel       12       0.021    12.446     4.254x

       40000  sequential        -       0.109     2.941     1.000x
       40000    parallel        1       0.113     2.828     0.962x
       40000    parallel        2       0.057     5.568     1.894x
       40000    parallel        4       0.033     9.621     3.272x
       40000    parallel        6       0.037     8.702     2.959x
       40000    parallel        8       0.029    11.053     3.759x
       40000    parallel       12       0.023    14.019     4.767x

       50000  sequential        -       0.136     2.943     1.000x
       50000    parallel        1       0.141     2.832     0.962x
       50000    parallel        2       0.072     5.538     1.882x
       50000    parallel        4       0.039    10.267     3.489x
       50000    parallel        6       0.030    13.223     4.493x
       50000    parallel        8       0.036    11.150     3.789x
       50000    parallel       12       0.028    14.329     4.869x

       75000  sequential        -       0.205     2.932     1.000x
       75000    parallel        1       0.212     2.831     0.966x
       75000    parallel        2       0.108     5.562     1.897x
       75000    parallel        4       0.058    10.415     3.553x
       75000    parallel        6       0.045    13.464     4.593x
       75000    parallel        8       0.052    11.450     3.906x
       75000    parallel       12       0.043    14.074     4.801x

      100000  sequential        -       0.272     2.939     1.000x
      100000    parallel        1       0.283     2.825     0.961x
      100000    parallel        2       0.144     5.547     1.887x
      100000    parallel        4       0.075    10.631     3.617x
      100000    parallel        6       0.063    12.651     4.304x
      100000    parallel        8       0.070    11.385     3.873x
      100000    parallel       12       0.057    14.108     4.799x

      200000  sequential        -       0.544     2.942     1.000x
      200000    parallel        1       0.567     2.821     0.959x
      200000    parallel        2       0.288     5.562     1.890x
      200000    parallel        4       0.168     9.543     3.243x
      200000    parallel        6       0.183     8.734     2.968x
      200000    parallel        8       0.143    11.184     3.801x
      200000    parallel       12       4.545     0.352     0.120x

      500000  sequential        -       1.375     2.909     1.000x
      500000    parallel        1       1.438     2.782     0.956x
      500000    parallel        2       0.731     5.472     1.881x
      500000    parallel        4       0.411     9.727     3.343x
      500000    parallel        6       0.477     8.392     2.885x
      500000    parallel        8       0.370    10.816     3.718x
      500000    parallel       12       6.003     0.666     0.229x

     1000000  sequential        -       2.803     2.854     1.000x
     1000000    parallel        1       2.798     2.859     1.002x
     1000000    parallel        2       1.450     5.518     1.933x
     1000000    parallel        4       0.795    10.068     3.527x
     1000000    parallel        6       0.649    12.324     4.318x
     1000000    parallel        8       0.742    10.783     3.778x
     1000000    parallel       12       0.732    10.936     3.831x

     2000000  sequential        -       5.518     2.900     1.000x
     2000000    parallel        1       5.883     2.720     0.938x
     2000000    parallel        2       2.949     5.426     1.871x
     2000000    parallel        4       1.878     8.518     2.937x
     2000000    parallel        6       1.716     9.326     3.216x
     2000000    parallel        8       1.474    10.855     3.743x
     2000000    parallel       12       1.705     9.386     3.237x

     5000000  sequential        -      14.043     2.848     1.000x
     5000000    parallel        1      14.586     2.742     0.963x
     5000000    parallel        2       7.577     5.279     1.853x
     5000000    parallel        4       5.668     7.057     2.478x
     5000000    parallel        6       3.391    11.795     4.141x
     5000000    parallel        8       3.854    10.378     3.644x
     5000000    parallel       12       5.237     7.638     2.682x

    10000000  sequential        -      28.194     2.837     1.000x
    10000000    parallel        1      28.985     2.760     0.973x
    10000000    parallel        2      14.687     5.447     1.920x
    10000000    parallel        4       8.434     9.486     3.343x
    10000000    parallel        6       6.160    12.987     4.577x
    10000000    parallel        8       7.282    10.985     3.872x
    10000000    parallel       12       7.876    10.157     3.580x

    33554432  sequential        -      91.236     2.942     1.000x
    33554432    parallel        1      95.301     2.817     0.957x
    33554432    parallel        2      50.093     5.359     1.821x
    33554432    parallel        4      29.709     9.035     3.071x
    33554432    parallel        6      23.593    11.378     3.867x
    33554432    parallel        8      24.243    11.073     3.763x
    33554432    parallel       12      21.069    12.741     4.330x


== dot ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001    12.139     1.000x
        1000    parallel        1       0.001    12.069     0.994x
        1000    parallel        2       0.001    11.954     0.985x
        1000    parallel        4       0.001    11.950     0.984x
        1000    parallel        6       0.001    11.936     0.983x
        1000    parallel        8       0.001    11.950     0.984x
        1000    parallel       12       0.001    11.930     0.983x

       10000  sequential        -       0.007    11.877     1.000x
       10000    parallel        1       0.007    11.822     0.995x
       10000    parallel        2       0.007    11.847     0.998x
       10000    parallel        4       0.007    11.838     0.997x
       10000    parallel        6       0.007    11.851     0.998x
       10000    parallel        8       0.007    11.830     0.996x
       10000    parallel       12       0.007    11.854     0.998x

       32768  sequential        -       0.022    11.926     1.000x
       32768    parallel        1       0.022    11.828     0.992x
       32768    parallel        2       0.022    11.829     0.992x
       32768    parallel        4       0.022    11.822     0.991x
       32768    parallel        6       0.022    11.813     0.990x
       32768    parallel        8       0.022    11.821     0.991x
       32768    parallel       12       0.022    11.819     0.991x

       32769  sequential        -       0.022    11.941     1.000x
       32769    parallel        1       0.022    11.836     0.991x
       32769    parallel        2       0.012    22.296     1.867x
       32769    parallel        4       0.006    40.654     3.405x
       32769    parallel        6       0.005    52.931     4.433x
       32769    parallel        8       0.005    49.221     4.122x
       32769    parallel       12       0.005    56.373     4.721x

       40000  sequential        -       0.027    11.892     1.000x
       40000    parallel        1       0.027    11.784     0.991x
       40000    parallel        2       0.014    22.296     1.875x
       40000    parallel        4       0.008    41.791     3.514x
       40000    parallel        6       0.008    42.255     3.553x
       40000    parallel        8       0.006    50.219     4.223x
       40000    parallel       12       0.005    60.227     5.064x

       50000  sequential        -       0.034    11.841     1.000x
       50000    parallel        1       0.034    11.750     0.992x
       50000    parallel        2       0.018    22.522     1.902x
       50000    parallel        4       0.009    42.664     3.603x
       50000    parallel        6       0.007    55.874     4.719x
       50000    parallel        8       0.008    51.623     4.360x
       50000    parallel       12       0.014    27.590     2.330x

       75000  sequential        -       0.051    11.705     1.000x
       75000    parallel        1       0.051    11.670     0.997x
       75000    parallel        2       0.026    22.792     1.947x
       75000    parallel        4       0.014    43.587     3.724x
       75000    parallel        6       0.012    49.133     4.197x
       75000    parallel        8       0.011    53.736     4.591x
       75000    parallel       12       0.009    63.751     5.446x

      100000  sequential        -       0.068    11.822     1.000x
      100000    parallel        1       0.069    11.665     0.987x
      100000    parallel        2       0.035    22.710     1.921x
      100000    parallel        4       0.018    44.532     3.767x
      100000    parallel        6       0.014    57.107     4.831x
      100000    parallel        8       0.015    54.519     4.612x
      100000    parallel       12       0.012    66.862     5.656x

      200000  sequential        -       0.135    11.844     1.000x
      200000    parallel        1       0.136    11.723     0.990x
      200000    parallel        2       0.070    22.863     1.930x
      200000    parallel        4       0.037    43.359     3.661x
      200000    parallel        6       0.027    58.582     4.946x
      200000    parallel        8       0.029    55.543     4.689x
      200000    parallel       12       0.024    65.313     5.514x

      500000  sequential        -       0.338    11.827     1.000x
      500000    parallel        1       0.341    11.722     0.991x
      500000    parallel        2       0.172    23.215     1.963x
      500000    parallel        4       0.091    43.889     3.711x
      500000    parallel        6       0.076    52.758     4.461x
      500000    parallel        8       0.076    52.976     4.479x
      500000    parallel       12       0.068    58.784     4.970x

     1000000  sequential        -       0.682    11.733     1.000x
     1000000    parallel        1       0.686    11.668     0.994x
     1000000    parallel        2       0.348    23.020     1.962x
     1000000    parallel        4       0.186    43.042     3.668x
     1000000    parallel        6       0.142    56.434     4.810x
     1000000    parallel        8       0.143    55.850     4.760x
     1000000    parallel       12       0.131    60.838     5.185x

     2000000  sequential        -       1.404    11.397     1.000x
     2000000    parallel        1       1.417    11.290     0.991x
     2000000    parallel        2       0.732    21.867     1.919x
     2000000    parallel        4       0.388    41.261     3.620x
     2000000    parallel        6       0.311    51.440     4.514x
     2000000    parallel        8       0.299    53.577     4.701x
     2000000    parallel       12       2.100     7.619     0.668x

     5000000  sequential        -       3.527    11.341     1.000x
     5000000    parallel        1       3.562    11.228     0.990x
     5000000    parallel        2       1.846    21.674     1.911x
     5000000    parallel        4       1.376    29.060     2.562x
     5000000    parallel        6       1.403    28.511     2.514x
     5000000    parallel        8       1.468    27.250     2.403x
     5000000    parallel       12       2.184    18.315     1.615x

    10000000  sequential        -       7.066    11.322     1.000x
    10000000    parallel        1       7.128    11.223     0.991x
    10000000    parallel        2       3.773    21.204     1.873x
    10000000    parallel        4       2.790    28.676     2.533x
    10000000    parallel        6       3.068    26.074     2.303x
    10000000    parallel        8       2.989    26.764     2.364x
    10000000    parallel       12       6.549    12.216     1.079x

    33554432  sequential        -      25.146    10.675     1.000x
    33554432    parallel        1      25.329    10.598     0.993x
    33554432    parallel        2      12.805    20.964     1.964x
    33554432    parallel        4       9.047    29.671     2.779x
    33554432    parallel        6       9.445    28.422     2.662x
    33554432    parallel        8       9.413    28.519     2.671x
    33554432    parallel       12      10.586    25.357     2.375x
```

</details>

<details>
<summary>Full output: cpu_parallel_benchmark, Machine 3, full thread range, run 2</summary>

```text
Sequential vs parallel on the CPU (float32)
compiler: gcc 13.3.0, OpenMP: yes, max threads: 12, hardware threads: 12, parallel cutoff: 32768 elements
sequential = plain loop, no OpenMP. parallel = parallel kernel with N threads.
Speedup = sequential time / this time. Median of 9 samples.

== add ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   192.781     1.000x
        1000    parallel        1       0.000   188.662     0.979x
        1000    parallel        2       0.000   183.846     0.954x
        1000    parallel        4       0.000   178.875     0.928x
        1000    parallel        6       0.000   182.507     0.947x
        1000    parallel        8       0.000   170.267     0.883x
        1000    parallel       12       0.000   164.525     0.853x

       10000  sequential        -       0.001   137.525     1.000x
       10000    parallel        1       0.001   136.232     0.991x
       10000    parallel        2       0.001   137.392     0.999x
       10000    parallel        4       0.001   139.103     1.011x
       10000    parallel        6       0.001   138.251     1.005x
       10000    parallel        8       0.001   139.230     1.012x
       10000    parallel       12       0.001   138.730     1.009x

       32768  sequential        -       0.003   138.250     1.000x
       32768    parallel        1       0.003   128.882     0.932x
       32768    parallel        2       0.003   128.486     0.929x
       32768    parallel        4       0.003   118.311     0.856x
       32768    parallel        6       0.003   118.630     0.858x
       32768    parallel        8       0.003   129.054     0.933x
       32768    parallel       12       0.003   128.688     0.931x

       32769  sequential        -       0.003   133.086     1.000x
       32769    parallel        1       0.003   133.909     1.006x
       32769    parallel        2       0.003   153.140     1.151x
       32769    parallel        4       0.002   230.765     1.734x
       32769    parallel        6       0.002   177.428     1.333x
       32769    parallel        8       0.002   197.659     1.485x
       32769    parallel       12       0.003   115.878     0.871x

       40000  sequential        -       0.004   133.239     1.000x
       40000    parallel        1       0.004   128.036     0.961x
       40000    parallel        2       0.003   163.518     1.227x
       40000    parallel        4       0.002   256.570     1.926x
       40000    parallel        6       0.002   261.072     1.959x
       40000    parallel        8       0.002   208.059     1.562x
       40000    parallel       12       0.009    54.986     0.413x

       50000  sequential        -       0.005   123.521     1.000x
       50000    parallel        1       0.005   123.185     0.997x
       50000    parallel        2       0.004   169.742     1.374x
       50000    parallel        4       0.003   186.586     1.511x
       50000    parallel        6       0.003   173.083     1.401x
       50000    parallel        8       0.003   239.985     1.943x
       50000    parallel       12       0.003   215.407     1.744x

       75000  sequential        -       0.007   123.103     1.000x
       75000    parallel        1       0.008   117.224     0.952x
       75000    parallel        2       0.005   166.421     1.352x
       75000    parallel        4       0.003   265.442     2.156x
       75000    parallel        6       0.003   338.415     2.749x
       75000    parallel        8       0.004   245.185     1.992x
       75000    parallel       12       0.005   196.078     1.593x

      100000  sequential        -       0.010   115.736     1.000x
      100000    parallel        1       0.010   122.667     1.060x
      100000    parallel        2       0.008   151.111     1.306x
      100000    parallel        4       0.004   291.861     2.522x
      100000    parallel        6       0.004   336.605     2.908x
      100000    parallel        8       0.004   270.191     2.335x
      100000    parallel       12       0.004   301.641     2.606x

      200000  sequential        -       0.019   123.300     1.000x
      200000    parallel        1       0.020   121.618     0.986x
      200000    parallel        2       0.015   161.365     1.309x
      200000    parallel        4       0.009   270.757     2.196x
      200000    parallel        6       0.010   232.253     1.884x
      200000    parallel        8       0.008   288.489     2.340x
      200000    parallel       12       0.007   330.785     2.683x

      500000  sequential        -       0.057   105.789     1.000x
      500000    parallel        1       0.054   110.828     1.048x
      500000    parallel        2       0.035   172.175     1.628x
      500000    parallel        4       0.021   280.696     2.653x
      500000    parallel        6       0.017   354.435     3.350x
      500000    parallel        8       0.021   287.462     2.717x
      500000    parallel       12       0.023   261.767     2.474x

     1000000  sequential        -       0.232    51.656     1.000x
     1000000    parallel        1       0.202    59.322     1.148x
     1000000    parallel        2       0.171    70.143     1.358x
     1000000    parallel        4       0.126    95.458     1.848x
     1000000    parallel        6       0.118   101.679     1.968x
     1000000    parallel        8       0.074   162.189     3.140x
     1000000    parallel       12       0.154    78.087     1.512x

     2000000  sequential        -       1.148    20.898     1.000x
     2000000    parallel        1       1.024    23.432     1.121x
     2000000    parallel        2       1.169    20.522     0.982x
     2000000    parallel        4       0.886    27.075     1.296x
     2000000    parallel        6       0.790    30.381     1.454x
     2000000    parallel        8       0.812    29.541     1.414x
     2000000    parallel       12       5.989     4.007     0.192x

     5000000  sequential        -       3.278    18.303     1.000x
     5000000    parallel        1       3.197    18.769     1.025x
     5000000    parallel        2       3.109    19.296     1.054x
     5000000    parallel        4       3.137    19.129     1.045x
     5000000    parallel        6       3.119    19.239     1.051x
     5000000    parallel        8       3.149    19.054     1.041x
     5000000    parallel       12       6.044     9.927     0.542x

    10000000  sequential        -       6.615    18.140     1.000x
    10000000    parallel        1       6.804    17.636     0.972x
    10000000    parallel        2       6.023    19.922     1.098x
    10000000    parallel        4       6.391    18.776     1.035x
    10000000    parallel        6       6.687    17.944     0.989x
    10000000    parallel        8       6.561    18.289     1.008x
    10000000    parallel       12       5.995    20.018     1.104x

    33554432  sequential        -      22.727    17.717     1.000x
    33554432    parallel        1      22.116    18.206     1.028x
    33554432    parallel        2      21.928    18.363     1.036x
    33554432    parallel        4      21.913    18.375     1.037x
    33554432    parallel        6      21.394    18.820     1.062x
    33554432    parallel        8      22.017    18.288     1.032x
    33554432    parallel       12      25.136    16.019     0.904x


== sub ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   187.326     1.000x
        1000    parallel        1       0.000   186.674     0.997x
        1000    parallel        2       0.000   187.899     1.003x
        1000    parallel        4       0.000   185.239     0.989x
        1000    parallel        6       0.000   187.209     0.999x
        1000    parallel        8       0.000   187.822     1.003x
        1000    parallel       12       0.000   187.541     1.001x

       10000  sequential        -       0.001   136.476     1.000x
       10000    parallel        1       0.001   138.491     1.015x
       10000    parallel        2       0.001   138.584     1.015x
       10000    parallel        4       0.001   138.401     1.014x
       10000    parallel        6       0.001   138.426     1.014x
       10000    parallel        8       0.001   138.568     1.015x
       10000    parallel       12       0.001   138.178     1.012x

       32768  sequential        -       0.003   132.435     1.000x
       32768    parallel        1       0.003   129.385     0.977x
       32768    parallel        2       0.003   130.353     0.984x
       32768    parallel        4       0.003   130.042     0.982x
       32768    parallel        6       0.003   126.226     0.953x
       32768    parallel        8       0.003   130.522     0.986x
       32768    parallel       12       0.003   130.440     0.985x

       32769  sequential        -       0.003   133.451     1.000x
       32769    parallel        1       0.003   124.181     0.931x
       32769    parallel        2       0.002   158.299     1.186x
       32769    parallel        4       0.002   239.727     1.796x
       32769    parallel        6       0.002   261.340     1.958x
       32769    parallel        8       0.002   204.699     1.534x
       32769    parallel       12       0.002   203.623     1.526x

       40000  sequential        -       0.004   131.134     1.000x
       40000    parallel        1       0.004   129.102     0.985x
       40000    parallel        2       0.003   161.873     1.234x
       40000    parallel        4       0.002   253.189     1.931x
       40000    parallel        6       0.002   268.999     2.051x
       40000    parallel        8       0.002   239.256     1.825x
       40000    parallel       12       0.002   201.387     1.536x

       50000  sequential        -       0.005   121.619     1.000x
       50000    parallel        1       0.005   123.494     1.015x
       50000    parallel        2       0.004   166.296     1.367x
       50000    parallel        4       0.002   273.230     2.247x
       50000    parallel        6       0.002   310.119     2.550x
       50000    parallel        8       0.003   230.032     1.891x
       50000    parallel       12       0.002   250.811     2.062x

       75000  sequential        -       0.007   123.787     1.000x
       75000    parallel        1       0.007   121.937     0.985x
       75000    parallel        2       0.005   173.198     1.399x
       75000    parallel        4       0.003   300.994     2.432x
       75000    parallel        6       0.003   355.516     2.872x
       75000    parallel        8       0.003   258.086     2.085x
       75000    parallel       12       0.003   284.932     2.302x

      100000  sequential        -       0.010   125.721     1.000x
      100000    parallel        1       0.010   121.614     0.967x
      100000    parallel        2       0.007   177.346     1.411x
      100000    parallel        4       0.004   297.058     2.363x
      100000    parallel        6       0.003   355.567     2.828x
      100000    parallel        8       0.004   268.851     2.138x
      100000    parallel       12       0.004   315.775     2.512x

      200000  sequential        -       0.019   125.286     1.000x
      200000    parallel        1       0.019   124.036     0.990x
      200000    parallel        2       0.015   160.394     1.280x
      200000    parallel        4       0.008   295.944     2.362x
      200000    parallel        6       0.006   397.685     3.174x
      200000    parallel        8       0.009   281.336     2.246x
      200000    parallel       12       0.007   339.552     2.710x

      500000  sequential        -       0.057   105.525     1.000x
      500000    parallel        1       0.053   114.211     1.082x
      500000    parallel        2       0.032   185.931     1.762x
      500000    parallel        4       0.022   274.277     2.599x
      500000    parallel        6       0.017   355.982     3.373x
      500000    parallel        8       0.021   290.640     2.754x
      500000    parallel       12       0.019   309.639     2.934x

     1000000  sequential        -       0.201    59.637     1.000x
     1000000    parallel        1       0.186    64.368     1.079x
     1000000    parallel        2       0.096   124.795     2.093x
     1000000    parallel        4       0.063   189.056     3.170x
     1000000    parallel        6       0.071   170.037     2.851x
     1000000    parallel        8       0.050   237.861     3.988x
     1000000    parallel       12       0.048   251.874     4.223x

     2000000  sequential        -       0.848    28.295     1.000x
     2000000    parallel        1       0.939    25.554     0.903x
     2000000    parallel        2       0.668    35.914     1.269x
     2000000    parallel        4       0.818    29.344     1.037x
     2000000    parallel        6       0.754    31.846     1.125x
     2000000    parallel        8       0.804    29.859     1.055x
     2000000    parallel       12       4.371     5.491     0.194x

     5000000  sequential        -       3.209    18.696     1.000x
     5000000    parallel        1       3.487    17.205     0.920x
     5000000    parallel        2       2.954    20.309     1.086x
     5000000    parallel        4       3.005    19.966     1.068x
     5000000    parallel        6       2.866    20.937     1.120x
     5000000    parallel        8       3.016    19.897     1.064x
     5000000    parallel       12       3.483    17.224     0.921x

    10000000  sequential        -       6.218    19.297     1.000x
    10000000    parallel        1       6.310    19.018     0.986x
    10000000    parallel        2       7.560    15.872     0.823x
    10000000    parallel        4       7.038    17.050     0.884x
    10000000    parallel        6       6.501    18.458     0.957x
    10000000    parallel        8       7.668    15.650     0.811x
    10000000    parallel       12      10.769    11.143     0.577x

    33554432  sequential        -      22.708    17.732     1.000x
    33554432    parallel        1      22.829    17.638     0.995x
    33554432    parallel        2      20.886    19.278     1.087x
    33554432    parallel        4      21.050    19.129     1.079x
    33554432    parallel        6      21.043    19.135     1.079x
    33554432    parallel        8      21.175    19.015     1.072x
    33554432    parallel       12      20.865    19.298     1.088x


== mul ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   191.655     1.000x
        1000    parallel        1       0.000   190.388     0.993x
        1000    parallel        2       0.000   190.777     0.995x
        1000    parallel        4       0.000   190.357     0.993x
        1000    parallel        6       0.000   190.020     0.991x
        1000    parallel        8       0.000   192.293     1.003x
        1000    parallel       12       0.000   190.691     0.995x

       10000  sequential        -       0.001   136.250     1.000x
       10000    parallel        1       0.001   142.557     1.046x
       10000    parallel        2       0.001   142.158     1.043x
       10000    parallel        4       0.001   139.318     1.023x
       10000    parallel        6       0.001   133.517     0.980x
       10000    parallel        8       0.001   139.558     1.024x
       10000    parallel       12       0.001   138.595     1.017x

       32768  sequential        -       0.003   133.013     1.000x
       32768    parallel        1       0.003   132.887     0.999x
       32768    parallel        2       0.003   131.134     0.986x
       32768    parallel        4       0.003   132.812     0.998x
       32768    parallel        6       0.003   132.352     0.995x
       32768    parallel        8       0.003   133.300     1.002x
       32768    parallel       12       0.003   133.281     1.002x

       32769  sequential        -       0.003   131.023     1.000x
       32769    parallel        1       0.003   129.677     0.990x
       32769    parallel        2       0.002   158.933     1.213x
       32769    parallel        4       0.002   240.728     1.837x
       32769    parallel        6       0.001   264.776     2.021x
       32769    parallel        8       0.002   201.177     1.535x
       32769    parallel       12       0.002   195.632     1.493x

       40000  sequential        -       0.004   124.820     1.000x
       40000    parallel        1       0.004   129.636     1.039x
       40000    parallel        2       0.003   159.030     1.274x
       40000    parallel        4       0.002   225.558     1.807x
       40000    parallel        6       0.002   269.372     2.158x
       40000    parallel        8       0.002   211.874     1.697x
       40000    parallel       12       0.002   215.109     1.723x

       50000  sequential        -       0.005   128.270     1.000x
       50000    parallel        1       0.005   128.068     0.998x
       50000    parallel        2       0.004   166.576     1.299x
       50000    parallel        4       0.002   260.590     2.032x
       50000    parallel        6       0.002   284.074     2.215x
       50000    parallel        8       0.003   228.201     1.779x
       50000    parallel       12       0.002   244.238     1.904x

       75000  sequential        -       0.007   124.602     1.000x
       75000    parallel        1       0.007   121.844     0.978x
       75000    parallel        2       0.005   174.709     1.402x
       75000    parallel        4       0.003   292.807     2.350x
       75000    parallel        6       0.003   269.578     2.164x
       75000    parallel        8       0.004   255.048     2.047x
       75000    parallel       12       0.004   245.095     1.967x

      100000  sequential        -       0.010   122.859     1.000x
      100000    parallel        1       0.010   121.282     0.987x
      100000    parallel        2       0.008   151.282     1.231x
      100000    parallel        4       0.004   303.694     2.472x
      100000    parallel        6       0.003   371.926     3.027x
      100000    parallel        8       0.004   273.197     2.224x
      100000    parallel       12       0.004   332.263     2.704x

      200000  sequential        -       0.019   124.616     1.000x
      200000    parallel        1       0.020   122.598     0.984x
      200000    parallel        2       0.015   161.656     1.297x
      200000    parallel        4       0.009   253.535     2.035x
      200000    parallel        6       0.007   361.127     2.898x
      200000    parallel        8       0.009   279.834     2.246x
      200000    parallel       12       0.007   334.788     2.687x

      500000  sequential        -       0.055   108.683     1.000x
      500000    parallel        1       0.056   107.727     0.991x
      500000    parallel        2       0.036   166.138     1.529x
      500000    parallel        4       0.022   269.538     2.480x
      500000    parallel        6       0.018   337.179     3.102x
      500000    parallel        8       0.021   289.978     2.668x
      500000    parallel       12       0.017   353.290     3.251x

     1000000  sequential        -       0.173    69.392     1.000x
     1000000    parallel        1       0.175    68.547     0.988x
     1000000    parallel        2       0.183    65.399     0.942x
     1000000    parallel        4       0.149    80.617     1.162x
     1000000    parallel        6       0.068   176.767     2.547x
     1000000    parallel        8       0.055   218.488     3.149x
     1000000    parallel       12       0.039   309.571     4.461x

     2000000  sequential        -       0.911    26.336     1.000x
     2000000    parallel        1       0.872    27.525     1.045x
     2000000    parallel        2       0.901    26.633     1.011x
     2000000    parallel        4       0.829    28.944     1.099x
     2000000    parallel        6       0.878    27.340     1.038x
     2000000    parallel        8       0.777    30.892     1.173x
     2000000    parallel       12       0.943    25.445     0.966x

     5000000  sequential        -       3.224    18.613     1.000x
     5000000    parallel        1       3.181    18.862     1.013x
     5000000    parallel        2       3.264    18.385     0.988x
     5000000    parallel        4       3.111    19.289     1.036x
     5000000    parallel        6       3.113    19.271     1.035x
     5000000    parallel        8       3.021    19.858     1.067x
     5000000    parallel       12       4.806    12.485     0.671x

    10000000  sequential        -       6.198    19.362     1.000x
    10000000    parallel        1       6.294    19.065     0.985x
    10000000    parallel        2       5.847    20.523     1.060x
    10000000    parallel        4       5.898    20.346     1.051x
    10000000    parallel        6       5.950    20.169     1.042x
    10000000    parallel        8       5.994    20.019     1.034x
    10000000    parallel       12       6.684    17.952     0.927x

    33554432  sequential        -      23.421    17.192     1.000x
    33554432    parallel        1      23.328    17.261     1.004x
    33554432    parallel        2      20.725    19.428     1.130x
    33554432    parallel        4      20.372    19.766     1.150x
    33554432    parallel        6      23.022    17.490     1.017x
    33554432    parallel        8      21.918    18.371     1.069x
    33554432    parallel       12      22.516    17.883     1.040x


== div ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    62.218     1.000x
        1000    parallel        1       0.000    62.162     0.999x
        1000    parallel        2       0.000    62.126     0.999x
        1000    parallel        4       0.000    62.332     1.002x
        1000    parallel        6       0.000    62.158     0.999x
        1000    parallel        8       0.000    62.256     1.001x
        1000    parallel       12       0.000    62.118     0.998x

       10000  sequential        -       0.002    62.153     1.000x
       10000    parallel        1       0.002    62.359     1.003x
       10000    parallel        2       0.002    62.377     1.004x
       10000    parallel        4       0.002    62.397     1.004x
       10000    parallel        6       0.002    62.393     1.004x
       10000    parallel        8       0.002    62.282     1.002x
       10000    parallel       12       0.002    62.349     1.003x

       32768  sequential        -       0.006    62.305     1.000x
       32768    parallel        1       0.006    62.434     1.002x
       32768    parallel        2       0.006    62.260     0.999x
       32768    parallel        4       0.006    62.378     1.001x
       32768    parallel        6       0.006    62.401     1.002x
       32768    parallel        8       0.006    62.381     1.001x
       32768    parallel       12       0.006    62.381     1.001x

       32769  sequential        -       0.006    62.207     1.000x
       32769    parallel        1       0.006    63.549     1.022x
       32769    parallel        2       0.004   110.747     1.780x
       32769    parallel        4       0.002   183.401     2.948x
       32769    parallel        6       0.002   220.849     3.550x
       32769    parallel        8       0.002   163.325     2.626x
       32769    parallel       12       0.002   160.757     2.584x

       40000  sequential        -       0.008    62.097     1.000x
       40000    parallel        1       0.008    63.247     1.019x
       40000    parallel        2       0.004   114.750     1.848x
       40000    parallel        4       0.003   187.965     3.027x
       40000    parallel        6       0.002   227.017     3.656x
       40000    parallel        8       0.003   170.988     2.754x
       40000    parallel       12       0.003   151.799     2.445x

       50000  sequential        -       0.010    61.610     1.000x
       50000    parallel        1       0.010    62.725     1.018x
       50000    parallel        2       0.005   110.890     1.800x
       50000    parallel        4       0.003   195.055     3.166x
       50000    parallel        6       0.002   247.258     4.013x
       50000    parallel        8       0.003   177.383     2.879x
       50000    parallel       12       0.002   246.995     4.009x

       75000  sequential        -       0.015    61.890     1.000x
       75000    parallel        1       0.014    62.577     1.011x
       75000    parallel        2       0.007   121.857     1.969x
       75000    parallel        4       0.004   209.279     3.381x
       75000    parallel        6       0.003   260.419     4.208x
       75000    parallel        8       0.005   195.734     3.163x
       75000    parallel       12       0.004   217.477     3.514x

      100000  sequential        -       0.019    61.945     1.000x
      100000    parallel        1       0.019    62.586     1.010x
      100000    parallel        2       0.010   119.383     1.927x
      100000    parallel        4       0.006   208.821     3.371x
      100000    parallel        6       0.004   272.403     4.398x
      100000    parallel        8       0.006   202.384     3.267x
      100000    parallel       12       0.005   243.046     3.924x

      200000  sequential        -       0.039    61.768     1.000x
      200000    parallel        1       0.039    61.869     1.002x
      200000    parallel        2       0.020   121.958     1.974x
      200000    parallel        4       0.012   201.967     3.270x
      200000    parallel        6       0.008   283.897     4.596x
      200000    parallel        8       0.012   204.939     3.318x
      200000    parallel       12       0.009   262.062     4.243x

      500000  sequential        -       0.097    62.004     1.000x
      500000    parallel        1       0.097    61.541     0.993x
      500000    parallel        2       0.048   125.504     2.024x
      500000    parallel        4       0.029   210.058     3.388x
      500000    parallel        6       0.036   167.916     2.708x
      500000    parallel        8       0.029   207.459     3.346x
      500000    parallel       12       0.025   243.172     3.922x

     1000000  sequential        -       0.233    51.548     1.000x
     1000000    parallel        1       0.234    51.292     0.995x
     1000000    parallel        2       0.189    63.582     1.233x
     1000000    parallel        4       0.083   144.727     2.808x
     1000000    parallel        6       0.089   135.125     2.621x
     1000000    parallel        8       0.091   131.299     2.547x
     1000000    parallel       12       5.997     2.001     0.039x

     2000000  sequential        -       1.184    20.268     1.000x
     2000000    parallel        1       1.007    23.823     1.175x
     2000000    parallel        2       0.819    29.287     1.445x
     2000000    parallel        4       0.859    27.943     1.379x
     2000000    parallel        6       0.812    29.547     1.458x
     2000000    parallel        8       0.823    29.147     1.438x
     2000000    parallel       12       5.858     4.097     0.202x

     5000000  sequential        -       3.482    17.231     1.000x
     5000000    parallel        1       3.294    18.217     1.057x
     5000000    parallel        2       2.991    20.061     1.164x
     5000000    parallel        4       3.035    19.769     1.147x
     5000000    parallel        6       3.027    19.819     1.150x
     5000000    parallel        8       3.137    19.125     1.110x
     5000000    parallel       12       5.988    10.021     0.582x

    10000000  sequential        -       7.187    16.697     1.000x
    10000000    parallel        1       7.289    16.463     0.986x
    10000000    parallel        2       6.705    17.897     1.072x
    10000000    parallel        4       5.987    20.043     1.200x
    10000000    parallel        6       5.948    20.176     1.208x
    10000000    parallel        8       6.080    19.737     1.182x
    10000000    parallel       12       6.539    18.350     1.099x

    33554432  sequential        -      23.433    17.183     1.000x
    33554432    parallel        1      23.232    17.332     1.009x
    33554432    parallel        2      20.889    19.276     1.122x
    33554432    parallel        4      20.916    19.251     1.120x
    33554432    parallel        6      21.318    18.888     1.099x
    33554432    parallel        8      21.074    19.106     1.112x
    33554432    parallel       12      20.965    19.206     1.118x


== neg ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   126.271     1.000x
        1000    parallel        1       0.000   123.649     0.979x
        1000    parallel        2       0.000   124.296     0.984x
        1000    parallel        4       0.000   123.167     0.975x
        1000    parallel        6       0.000   124.487     0.986x
        1000    parallel        8       0.000   125.137     0.991x
        1000    parallel       12       0.000   124.836     0.989x

       10000  sequential        -       0.001   132.890     1.000x
       10000    parallel        1       0.001   135.448     1.019x
       10000    parallel        2       0.001   135.473     1.019x
       10000    parallel        4       0.001   134.775     1.014x
       10000    parallel        6       0.001   134.644     1.013x
       10000    parallel        8       0.001   134.327     1.011x
       10000    parallel       12       0.001   134.330     1.011x

       32768  sequential        -       0.002   136.339     1.000x
       32768    parallel        1       0.002   136.485     1.001x
       32768    parallel        2       0.002   136.575     1.002x
       32768    parallel        4       0.002   136.385     1.000x
       32768    parallel        6       0.002   136.064     0.998x
       32768    parallel        8       0.002   136.392     1.000x
       32768    parallel       12       0.002   136.221     0.999x

       32769  sequential        -       0.002   136.452     1.000x
       32769    parallel        1       0.002   136.312     0.999x
       32769    parallel        2       0.002   107.288     0.786x
       32769    parallel        4       0.002   157.453     1.154x
       32769    parallel        6       0.001   177.979     1.304x
       32769    parallel        8       0.002   141.228     1.035x
       32769    parallel       12       0.002   141.466     1.037x

       40000  sequential        -       0.002   138.072     1.000x
       40000    parallel        1       0.002   137.274     0.994x
       40000    parallel        2       0.003   111.408     0.807x
       40000    parallel        4       0.002   156.761     1.135x
       40000    parallel        6       0.002   185.322     1.342x
       40000    parallel        8       0.002   152.831     1.107x
       40000    parallel       12       0.002   151.991     1.101x

       50000  sequential        -       0.003   128.993     1.000x
       50000    parallel        1       0.003   122.326     0.948x
       50000    parallel        2       0.003   116.122     0.900x
       50000    parallel        4       0.002   173.048     1.342x
       50000    parallel        6       0.003   143.392     1.112x
       50000    parallel        8       0.002   162.899     1.263x
       50000    parallel       12       0.003   153.120     1.187x

       75000  sequential        -       0.005   122.512     1.000x
       75000    parallel        1       0.006   105.734     0.863x
       75000    parallel        2       0.005   121.386     0.991x
       75000    parallel        4       0.003   199.140     1.625x
       75000    parallel        6       0.003   238.645     1.948x
       75000    parallel        8       0.003   183.951     1.501x
       75000    parallel       12       0.003   172.043     1.404x

      100000  sequential        -       0.006   123.601     1.000x
      100000    parallel        1       0.007   119.546     0.967x
      100000    parallel        2       0.007   120.939     0.978x
      100000    parallel        4       0.004   199.193     1.612x
      100000    parallel        6       0.005   148.341     1.200x
      100000    parallel        8       0.004   202.989     1.642x
      100000    parallel       12       0.003   280.210     2.267x

      200000  sequential        -       0.013   124.152     1.000x
      200000    parallel        1       0.013   120.771     0.973x
      200000    parallel        2       0.013   125.580     1.011x
      200000    parallel        4       0.008   208.774     1.682x
      200000    parallel        6       0.006   261.200     2.104x
      200000    parallel        8       0.007   215.966     1.740x
      200000    parallel       12       0.006   255.129     2.055x

      500000  sequential        -       0.035   115.778     1.000x
      500000    parallel        1       0.034   118.322     1.022x
      500000    parallel        2       0.033   122.043     1.054x
      500000    parallel        4       0.019   211.073     1.823x
      500000    parallel        6       0.015   268.151     2.316x
      500000    parallel        8       0.018   217.458     1.878x
      500000    parallel       12       0.019   213.709     1.846x

     1000000  sequential        -       0.100    79.675     1.000x
     1000000    parallel        1       0.072   111.614     1.401x
     1000000    parallel        2       0.061   131.011     1.644x
     1000000    parallel        4       0.040   202.103     2.537x
     1000000    parallel        6       0.029   272.908     3.425x
     1000000    parallel        8       0.036   219.396     2.754x
     1000000    parallel       12       0.028   281.517     3.533x

     2000000  sequential        -       0.498    32.101     1.000x
     2000000    parallel        1       0.499    32.057     0.999x
     2000000    parallel        2       0.306    52.247     1.628x
     2000000    parallel        4       0.223    71.718     2.234x
     2000000    parallel        6       0.178    89.827     2.798x
     2000000    parallel        8       0.181    88.630     2.761x
     2000000    parallel       12       0.218    73.380     2.286x

     5000000  sequential        -       2.259    17.703     1.000x
     5000000    parallel        1       2.144    18.661     1.054x
     5000000    parallel        2       1.983    20.176     1.140x
     5000000    parallel        4       2.037    19.637     1.109x
     5000000    parallel        6       2.031    19.691     1.112x
     5000000    parallel        8       2.057    19.445     1.098x
     5000000    parallel       12       2.074    19.288     1.090x

    10000000  sequential        -       4.929    16.231     1.000x
    10000000    parallel        1       4.649    17.208     1.060x
    10000000    parallel        2       4.423    18.088     1.114x
    10000000    parallel        4       4.525    17.680     1.089x
    10000000    parallel        6       4.719    16.953     1.044x
    10000000    parallel        8       4.476    17.873     1.101x
    10000000    parallel       12       5.432    14.728     0.907x

    33554432  sequential        -      17.749    15.124     1.000x
    33554432    parallel        1      18.368    14.614     0.966x
    33554432    parallel        2      15.845    16.942     1.120x
    33554432    parallel        4      15.940    16.841     1.114x
    33554432    parallel        6      16.117    16.656     1.101x
    33554432    parallel        8      16.055    16.720     1.106x
    33554432    parallel       12      16.048    16.728     1.106x


== abs ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    68.384     1.000x
        1000    parallel        1       0.000   128.866     1.884x
        1000    parallel        2       0.000   129.622     1.896x
        1000    parallel        4       0.000   129.642     1.896x
        1000    parallel        6       0.000   128.317     1.876x
        1000    parallel        8       0.000   128.672     1.882x
        1000    parallel       12       0.000   128.635     1.881x

       10000  sequential        -       0.001    70.906     1.000x
       10000    parallel        1       0.001   139.479     1.967x
       10000    parallel        2       0.001   139.519     1.968x
       10000    parallel        4       0.001   139.818     1.972x
       10000    parallel        6       0.001   139.492     1.967x
       10000    parallel        8       0.001   139.752     1.971x
       10000    parallel       12       0.001   139.226     1.964x

       32768  sequential        -       0.004    71.732     1.000x
       32768    parallel        1       0.002   140.760     1.962x
       32768    parallel        2       0.002   140.627     1.960x
       32768    parallel        4       0.002   140.100     1.953x
       32768    parallel        6       0.002   140.606     1.960x
       32768    parallel        8       0.002   140.675     1.961x
       32768    parallel       12       0.002   139.254     1.941x

       32769  sequential        -       0.004    71.642     1.000x
       32769    parallel        1       0.002   135.558     1.892x
       32769    parallel        2       0.002   166.969     2.331x
       32769    parallel        4       0.001   227.881     3.181x
       32769    parallel        6       0.001   232.886     3.251x
       32769    parallel        8       0.001   193.629     2.703x
       32769    parallel       12       0.001   176.535     2.464x

       40000  sequential        -       0.004    71.439     1.000x
       40000    parallel        1       0.002   140.187     1.962x
       40000    parallel        2       0.002   183.723     2.572x
       40000    parallel        4       0.001   252.144     3.529x
       40000    parallel        6       0.001   266.754     3.734x
       40000    parallel        8       0.002   211.520     2.961x
       40000    parallel       12       0.002   184.629     2.584x

       50000  sequential        -       0.006    70.643     1.000x
       50000    parallel        1       0.003   138.621     1.962x
       50000    parallel        2       0.002   196.493     2.781x
       50000    parallel        4       0.001   278.519     3.943x
       50000    parallel        6       0.001   298.810     4.230x
       50000    parallel        8       0.002   234.154     3.315x
       50000    parallel       12       0.002   209.249     2.962x

       75000  sequential        -       0.008    70.603     1.000x
       75000    parallel        1       0.005   123.473     1.749x
       75000    parallel        2       0.003   213.741     3.027x
       75000    parallel        4       0.002   320.478     4.539x
       75000    parallel        6       0.002   358.708     5.081x
       75000    parallel        8       0.002   278.910     3.950x
       75000    parallel       12       0.002   252.651     3.578x

      100000  sequential        -       0.011    71.087     1.000x
      100000    parallel        1       0.006   124.953     1.758x
      100000    parallel        2       0.004   212.849     2.994x
      100000    parallel        4       0.002   360.251     5.068x
      100000    parallel        6       0.002   416.125     5.854x
      100000    parallel        8       0.003   304.295     4.281x
      100000    parallel       12       0.002   326.496     4.593x

      200000  sequential        -       0.023    70.474     1.000x
      200000    parallel        1       0.013   124.292     1.764x
      200000    parallel        2       0.008   190.726     2.706x
      200000    parallel        4       0.004   360.652     5.118x
      200000    parallel        6       0.003   471.758     6.694x
      200000    parallel        8       0.005   339.344     4.815x
      200000    parallel       12       0.004   376.023     5.336x

      500000  sequential        -       0.057    70.152     1.000x
      500000    parallel        1       0.035   113.138     1.613x
      500000    parallel        2       0.019   206.142     2.938x
      500000    parallel        4       0.012   325.995     4.647x
      500000    parallel        6       0.010   399.353     5.693x
      500000    parallel        8       0.013   313.493     4.469x
      500000    parallel       12       0.016   253.135     3.608x

     1000000  sequential        -       0.128    62.704     1.000x
     1000000    parallel        1       0.101    79.523     1.268x
     1000000    parallel        2       0.086    92.725     1.479x
     1000000    parallel        4       0.032   252.520     4.027x
     1000000    parallel        6       0.032   250.231     3.991x
     1000000    parallel        8       0.030   270.626     4.316x
     1000000    parallel       12       4.186     1.911     0.030x

     2000000  sequential        -       0.732    21.858     1.000x
     2000000    parallel        1       0.794    20.161     0.922x
     2000000    parallel        2       0.383    41.762     1.911x
     2000000    parallel        4       0.290    55.153     2.523x
     2000000    parallel        6       0.332    48.215     2.206x
     2000000    parallel        8       0.299    53.450     2.445x
     2000000    parallel       12       0.278    57.646     2.637x

     5000000  sequential        -       2.783    14.372     1.000x
     5000000    parallel        1       2.338    17.110     1.190x
     5000000    parallel        2       2.177    18.373     1.278x
     5000000    parallel        4       2.124    18.836     1.311x
     5000000    parallel        6       2.217    18.045     1.256x
     5000000    parallel        8       2.173    18.411     1.281x
     5000000    parallel       12       5.990     6.677     0.465x

    10000000  sequential        -       5.023    15.926     1.000x
    10000000    parallel        1       4.776    16.749     1.052x
    10000000    parallel        2       5.112    15.649     0.983x
    10000000    parallel        4       4.570    17.506     1.099x
    10000000    parallel        6       4.455    17.957     1.128x
    10000000    parallel        8       4.501    17.772     1.116x
    10000000    parallel       12       4.658    17.174     1.078x

    33554432  sequential        -      17.397    15.430     1.000x
    33554432    parallel        1      18.059    14.864     0.963x
    33554432    parallel        2      16.277    16.491     1.069x
    33554432    parallel        4      16.427    16.341     1.059x
    33554432    parallel        6      16.558    16.212     1.051x
    33554432    parallel        8      16.110    16.663     1.080x
    33554432    parallel       12      17.374    15.450     1.001x


== square ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   124.809     1.000x
        1000    parallel        1       0.000   123.648     0.991x
        1000    parallel        2       0.000   123.541     0.990x
        1000    parallel        4       0.000   123.358     0.988x
        1000    parallel        6       0.000   122.826     0.984x
        1000    parallel        8       0.000   125.098     1.002x
        1000    parallel       12       0.000   124.520     0.998x

       10000  sequential        -       0.001   133.030     1.000x
       10000    parallel        1       0.001   134.803     1.013x
       10000    parallel        2       0.001   133.681     1.005x
       10000    parallel        4       0.001   132.699     0.998x
       10000    parallel        6       0.001   132.249     0.994x
       10000    parallel        8       0.001   133.215     1.001x
       10000    parallel       12       0.001   133.160     1.001x

       32768  sequential        -       0.002   134.366     1.000x
       32768    parallel        1       0.002   135.346     1.007x
       32768    parallel        2       0.002   134.203     0.999x
       32768    parallel        4       0.002   135.997     1.012x
       32768    parallel        6       0.002   135.133     1.006x
       32768    parallel        8       0.002   135.585     1.009x
       32768    parallel       12       0.002   134.035     0.998x

       32769  sequential        -       0.002   133.403     1.000x
       32769    parallel        1       0.002   133.382     1.000x
       32769    parallel        2       0.002   165.264     1.239x
       32769    parallel        4       0.001   224.789     1.685x
       32769    parallel        6       0.001   229.663     1.722x
       32769    parallel        8       0.001   191.340     1.434x
       32769    parallel       12       0.001   175.325     1.314x

       40000  sequential        -       0.002   134.497     1.000x
       40000    parallel        1       0.002   135.638     1.008x
       40000    parallel        2       0.002   181.028     1.346x
       40000    parallel        4       0.001   251.716     1.872x
       40000    parallel        6       0.001   267.053     1.986x
       40000    parallel        8       0.002   203.955     1.516x
       40000    parallel       12       0.002   209.704     1.559x

       50000  sequential        -       0.003   130.308     1.000x
       50000    parallel        1       0.003   128.118     0.983x
       50000    parallel        2       0.002   192.132     1.474x
       50000    parallel        4       0.001   278.227     2.135x
       50000    parallel        6       0.001   285.105     2.188x
       50000    parallel        8       0.002   233.317     1.791x
       50000    parallel       12       0.002   225.559     1.731x

       75000  sequential        -       0.005   121.637     1.000x
       75000    parallel        1       0.005   119.407     0.982x
       75000    parallel        2       0.003   203.800     1.675x
       75000    parallel        4       0.002   319.652     2.628x
       75000    parallel        6       0.002   346.538     2.849x
       75000    parallel        8       0.002   278.052     2.286x
       75000    parallel       12       0.002   268.397     2.207x

      100000  sequential        -       0.007   120.484     1.000x
      100000    parallel        1       0.007   118.607     0.984x
      100000    parallel        2       0.004   210.761     1.749x
      100000    parallel        4       0.002   359.725     2.986x
      100000    parallel        6       0.003   270.940     2.249x
      100000    parallel        8       0.003   306.261     2.542x
      100000    parallel       12       0.003   304.702     2.529x

      200000  sequential        -       0.015   110.031     1.000x
      200000    parallel        1       0.014   116.940     1.063x
      200000    parallel        2       0.008   188.249     1.711x
      200000    parallel        4       0.004   368.118     3.346x
      200000    parallel        6       0.006   270.452     2.458x
      200000    parallel        8       0.005   350.229     3.183x
      200000    parallel       12       0.004   423.112     3.845x

      500000  sequential        -       0.038   106.007     1.000x
      500000    parallel        1       0.040   100.900     0.952x
      500000    parallel        2       0.021   189.175     1.785x
      500000    parallel        4       0.014   289.807     2.734x
      500000    parallel        6       0.010   394.882     3.725x
      500000    parallel        8       0.012   328.747     3.101x
      500000    parallel       12       0.011   352.412     3.324x

     1000000  sequential        -       0.096    83.455     1.000x
     1000000    parallel        1       0.089    89.601     1.074x
     1000000    parallel        2       0.051   157.566     1.888x
     1000000    parallel        4       0.027   300.062     3.595x
     1000000    parallel        6       0.023   346.763     4.155x
     1000000    parallel        8       0.027   299.834     3.593x
     1000000    parallel       12       0.025   315.162     3.776x

     2000000  sequential        -       0.512    31.223     1.000x
     2000000    parallel        1       0.513    31.181     0.999x
     2000000    parallel        2       0.337    47.469     1.520x
     2000000    parallel        4       0.225    71.205     2.281x
     2000000    parallel        6       0.203    78.764     2.523x
     2000000    parallel        8       0.213    75.140     2.407x
     2000000    parallel       12       0.284    56.365     1.805x

     5000000  sequential        -       2.324    17.215     1.000x
     5000000    parallel        1       2.190    18.268     1.061x
     5000000    parallel        2       1.923    20.799     1.208x
     5000000    parallel        4       2.053    19.480     1.132x
     5000000    parallel        6       1.986    20.146     1.170x
     5000000    parallel        8       1.989    20.108     1.168x
     5000000    parallel       12       2.082    19.215     1.116x

    10000000  sequential        -       5.602    14.280     1.000x
    10000000    parallel        1       5.390    14.843     1.039x
    10000000    parallel        2       4.596    17.408     1.219x
    10000000    parallel        4       5.089    15.719     1.101x
    10000000    parallel        6       5.060    15.812     1.107x
    10000000    parallel        8       4.623    17.304     1.212x
    10000000    parallel       12       6.114    13.085     0.916x

    33554432  sequential        -      17.932    14.969     1.000x
    33554432    parallel        1      18.505    14.506     0.969x
    33554432    parallel        2      15.718    17.079     1.141x
    33554432    parallel        4      15.497    17.321     1.157x
    33554432    parallel        6      15.732    17.063     1.140x
    33554432    parallel        8      15.696    17.102     1.142x
    33554432    parallel       12      16.481    16.288     1.088x


== relu ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   110.040     1.000x
        1000    parallel        1       0.000    66.563     0.605x
        1000    parallel        2       0.000    66.691     0.606x
        1000    parallel        4       0.000    66.660     0.606x
        1000    parallel        6       0.000    66.694     0.606x
        1000    parallel        8       0.000    66.680     0.606x
        1000    parallel       12       0.000    66.444     0.604x

       10000  sequential        -       0.001   106.651     1.000x
       10000    parallel        1       0.001    71.526     0.671x
       10000    parallel        2       0.001    71.617     0.672x
       10000    parallel        4       0.001    71.820     0.673x
       10000    parallel        6       0.001    71.840     0.674x
       10000    parallel        8       0.001    71.627     0.672x
       10000    parallel       12       0.001    71.623     0.672x

       32768  sequential        -       0.003   101.568     1.000x
       32768    parallel        1       0.004    70.861     0.698x
       32768    parallel        2       0.004    71.142     0.700x
       32768    parallel        4       0.004    70.955     0.699x
       32768    parallel        6       0.004    69.922     0.688x
       32768    parallel        8       0.004    69.559     0.685x
       32768    parallel       12       0.004    69.355     0.683x

       32769  sequential        -       0.003    98.745     1.000x
       32769    parallel        1       0.004    67.481     0.683x
       32769    parallel        2       0.002   105.641     1.070x
       32769    parallel        4       0.002   158.246     1.603x
       32769    parallel        6       0.002   114.543     1.160x
       32769    parallel        8       0.002   140.630     1.424x
       32769    parallel       12       0.002   139.716     1.415x

       40000  sequential        -       0.003   100.335     1.000x
       40000    parallel        1       0.005    69.693     0.695x
       40000    parallel        2       0.003   110.981     1.106x
       40000    parallel        4       0.002   167.452     1.669x
       40000    parallel        6       0.002   187.985     1.874x
       40000    parallel        8       0.002   149.052     1.486x
       40000    parallel       12       0.002   151.716     1.512x

       50000  sequential        -       0.004   105.130     1.000x
       50000    parallel        1       0.006    70.584     0.671x
       50000    parallel        2       0.004   113.914     1.084x
       50000    parallel        4       0.002   179.524     1.708x
       50000    parallel        6       0.002   208.934     1.987x
       50000    parallel        8       0.002   164.793     1.568x
       50000    parallel       12       0.002   174.753     1.662x

       75000  sequential        -       0.006    94.336     1.000x
       75000    parallel        1       0.008    71.056     0.753x
       75000    parallel        2       0.005   115.029     1.219x
       75000    parallel        4       0.003   202.080     2.142x
       75000    parallel        6       0.003   238.442     2.528x
       75000    parallel        8       0.003   181.989     1.929x
       75000    parallel       12       0.003   207.327     2.198x

      100000  sequential        -       0.008   102.004     1.000x
      100000    parallel        1       0.011    70.590     0.692x
      100000    parallel        2       0.007   121.696     1.193x
      100000    parallel        4       0.004   205.077     2.010x
      100000    parallel        6       0.003   252.565     2.476x
      100000    parallel        8       0.004   189.638     1.859x
      100000    parallel       12       0.004   220.835     2.165x

      200000  sequential        -       0.016   102.836     1.000x
      200000    parallel        1       0.024    66.279     0.645x
      200000    parallel        2       0.014   110.722     1.077x
      200000    parallel        4       0.009   182.697     1.777x
      200000    parallel        6       0.006   272.698     2.652x
      200000    parallel        8       0.008   209.078     2.033x
      200000    parallel       12       0.005   325.269     3.163x

      500000  sequential        -       0.042    94.571     1.000x
      500000    parallel        1       0.059    67.971     0.719x
      500000    parallel        2       0.033   120.481     1.274x
      500000    parallel        4       0.021   188.282     1.991x
      500000    parallel        6       0.016   254.374     2.690x
      500000    parallel        8       0.019   208.115     2.201x
      500000    parallel       12       0.018   223.424     2.362x

     1000000  sequential        -       0.109    73.095     1.000x
     1000000    parallel        1       0.128    62.560     0.856x
     1000000    parallel        2       0.070   114.020     1.560x
     1000000    parallel        4       0.041   193.422     2.646x
     1000000    parallel        6       0.031   259.900     3.556x
     1000000    parallel        8       0.039   204.649     2.800x
     1000000    parallel       12       0.034   236.241     3.232x

     2000000  sequential        -       0.569    28.108     1.000x
     2000000    parallel        1       0.551    29.059     1.034x
     2000000    parallel        2       0.365    43.792     1.558x
     2000000    parallel        4       0.232    69.044     2.456x
     2000000    parallel        6       0.212    75.523     2.687x
     2000000    parallel        8       0.259    61.765     2.197x
     2000000    parallel       12       0.244    65.674     2.337x

     5000000  sequential        -       3.412    11.723     1.000x
     5000000    parallel        1       2.770    14.443     1.232x
     5000000    parallel        2       2.329    17.177     1.465x
     5000000    parallel        4       2.448    16.339     1.394x
     5000000    parallel        6       2.230    17.933     1.530x
     5000000    parallel        8       2.249    17.790     1.517x
     5000000    parallel       12       5.995     6.672     0.569x

    10000000  sequential        -       5.182    15.438     1.000x
    10000000    parallel        1       5.138    15.570     1.009x
    10000000    parallel        2       4.782    16.730     1.084x
    10000000    parallel        4       4.947    16.170     1.047x
    10000000    parallel        6       4.808    16.638     1.078x
    10000000    parallel        8       5.974    13.392     0.867x
    10000000    parallel       12      10.106     7.916     0.513x

    33554432  sequential        -      18.009    14.905     1.000x
    33554432    parallel        1      18.346    14.632     0.982x
    33554432    parallel        2      15.330    17.511     1.175x
    33554432    parallel        4      16.200    16.570     1.112x
    33554432    parallel        6      16.060    16.714     1.121x
    33554432    parallel        8      15.783    17.008     1.141x
    33554432    parallel       12      16.489    16.280     1.092x


== sqrt ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001     7.234     1.000x
        1000    parallel        1       0.001     7.214     0.997x
        1000    parallel        2       0.001     7.234     1.000x
        1000    parallel        4       0.001     7.231     1.000x
        1000    parallel        6       0.001     7.221     0.998x
        1000    parallel        8       0.001     7.237     1.000x
        1000    parallel       12       0.001     7.233     1.000x

       10000  sequential        -       0.011     7.235     1.000x
       10000    parallel        1       0.011     7.226     0.999x
       10000    parallel        2       0.011     7.223     0.998x
       10000    parallel        4       0.011     7.229     0.999x
       10000    parallel        6       0.011     7.232     0.999x
       10000    parallel        8       0.011     7.232     1.000x
       10000    parallel       12       0.011     7.228     0.999x

       32768  sequential        -       0.036     7.230     1.000x
       32768    parallel        1       0.036     7.231     1.000x
       32768    parallel        2       0.036     7.228     1.000x
       32768    parallel        4       0.036     7.217     0.998x
       32768    parallel        6       0.036     7.209     0.997x
       32768    parallel        8       0.036     7.226     0.999x
       32768    parallel       12       0.036     7.219     0.998x

       32769  sequential        -       0.036     7.225     1.000x
       32769    parallel        1       0.036     7.222     1.000x
       32769    parallel        2       0.019    13.990     1.936x
       32769    parallel        4       0.010    26.856     3.717x
       32769    parallel        6       0.013    20.724     2.868x
       32769    parallel        8       0.010    25.922     3.588x
       32769    parallel       12       0.009    28.599     3.958x

       40000  sequential        -       0.044     7.221     1.000x
       40000    parallel        1       0.044     7.221     1.000x
       40000    parallel        2       0.023    14.050     1.946x
       40000    parallel        4       0.012    27.148     3.760x
       40000    parallel        6       0.008    38.504     5.333x
       40000    parallel        8       0.012    26.369     3.652x
       40000    parallel       12       0.011    29.361     4.066x

       50000  sequential        -       0.056     7.140     1.000x
       50000    parallel        1       0.055     7.224     1.012x
       50000    parallel        2       0.028    14.121     1.978x
       50000    parallel        4       0.015    27.527     3.855x
       50000    parallel        6       0.019    20.951     2.934x
       50000    parallel        8       0.015    27.122     3.798x
       50000    parallel       12       0.012    33.052     4.629x

       75000  sequential        -       0.083     7.213     1.000x
       75000    parallel        1       0.083     7.214     1.000x
       75000    parallel        2       0.042    14.219     1.971x
       75000    parallel        4       0.021    28.047     3.888x
       75000    parallel        6       0.015    39.253     5.442x
       75000    parallel        8       0.022    27.056     3.751x
       75000    parallel       12       0.020    29.495     4.089x

      100000  sequential        -       0.111     7.216     1.000x
      100000    parallel        1       0.111     7.216     1.000x
      100000    parallel        2       0.056    14.193     1.967x
      100000    parallel        4       0.028    28.213     3.910x
      100000    parallel        6       0.020    40.860     5.662x
      100000    parallel        8       0.029    27.577     3.821x
      100000    parallel       12       0.022    36.717     5.088x

      200000  sequential        -       0.222     7.221     1.000x
      200000    parallel        1       0.222     7.215     0.999x
      200000    parallel        2       0.112    14.323     1.984x
      200000    parallel        4       0.056    28.502     3.947x
      200000    parallel        6       0.075    21.473     2.974x
      200000    parallel        8       0.057    28.044     3.884x
      200000    parallel       12       0.042    38.309     5.305x

      500000  sequential        -       0.553     7.228     1.000x
      500000    parallel        1       0.555     7.203     0.997x
      500000    parallel        2       0.278    14.390     1.991x
      500000    parallel        4       0.143    27.882     3.858x
      500000    parallel        6       0.099    40.422     5.593x
      500000    parallel        8       0.144    27.864     3.855x
      500000    parallel       12       0.109    36.683     5.075x

     1000000  sequential        -       1.109     7.213     1.000x
     1000000    parallel        1       1.112     7.197     0.998x
     1000000    parallel        2       0.557    14.368     1.992x
     1000000    parallel        4       0.281    28.458     3.945x
     1000000    parallel        6       0.197    40.669     5.638x
     1000000    parallel        8       0.294    27.235     3.776x
     1000000    parallel       12       0.209    38.220     5.299x

     2000000  sequential        -       2.256     7.093     1.000x
     2000000    parallel        1       2.260     7.079     0.998x
     2000000    parallel        2       1.122    14.258     2.010x
     2000000    parallel        4       0.598    26.744     3.770x
     2000000    parallel        6       0.431    37.144     5.237x
     2000000    parallel        8       0.598    26.765     3.773x
     2000000    parallel       12       0.485    32.993     4.652x

     5000000  sequential        -       5.764     6.940     1.000x
     5000000    parallel        1       5.701     7.016     1.011x
     5000000    parallel        2       3.081    12.983     1.871x
     5000000    parallel        4       2.096    19.088     2.750x
     5000000    parallel        6       2.066    19.360     2.790x
     5000000    parallel        8       2.063    19.388     2.794x
     5000000    parallel       12       2.281    17.533     2.526x

    10000000  sequential        -      11.587     6.904     1.000x
    10000000    parallel        1      11.672     6.854     0.993x
    10000000    parallel        2       6.132    13.046     1.889x
    10000000    parallel        4       4.542    17.615     2.551x
    10000000    parallel        6       4.697    17.031     2.467x
    10000000    parallel        8       4.602    17.385     2.518x
    10000000    parallel       12       6.730    11.887     1.722x

    33554432  sequential        -      38.631     6.949     1.000x
    33554432    parallel        1      39.873     6.732     0.969x
    33554432    parallel        2      21.719    12.359     1.779x
    33554432    parallel        4      17.223    15.586     2.243x
    33554432    parallel        6      16.041    16.734     2.408x
    33554432    parallel        8      17.171    15.633     2.250x
    33554432    parallel       12      17.888    15.007     2.160x


== exp ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.002     3.260     1.000x
        1000    parallel        1       0.003     2.996     0.919x
        1000    parallel        2       0.003     2.992     0.918x
        1000    parallel        4       0.003     2.990     0.917x
        1000    parallel        6       0.003     3.001     0.920x
        1000    parallel        8       0.003     2.993     0.918x
        1000    parallel       12       0.003     2.991     0.917x

       10000  sequential        -       0.024     3.268     1.000x
       10000    parallel        1       0.027     3.008     0.921x
       10000    parallel        2       0.027     2.996     0.917x
       10000    parallel        4       0.027     3.000     0.918x
       10000    parallel        6       0.027     2.988     0.914x
       10000    parallel        8       0.027     2.987     0.914x
       10000    parallel       12       0.027     3.001     0.918x

       32768  sequential        -       0.081     3.249     1.000x
       32768    parallel        1       0.088     2.994     0.922x
       32768    parallel        2       0.087     2.997     0.923x
       32768    parallel        4       0.088     2.988     0.920x
       32768    parallel        6       0.088     2.991     0.921x
       32768    parallel        8       0.088     2.987     0.920x
       32768    parallel       12       0.087     3.000     0.923x

       32769  sequential        -       0.080     3.266     1.000x
       32769    parallel        1       0.087     3.004     0.920x
       32769    parallel        2       0.041     6.383     1.954x
       32769    parallel        4       0.022    11.773     3.604x
       32769    parallel        6       0.023    11.495     3.519x
       32769    parallel        8       0.019    14.039     4.298x
       32769    parallel       12       0.016    16.046     4.913x

       40000  sequential        -       0.097     3.283     1.000x
       40000    parallel        1       0.107     2.994     0.912x
       40000    parallel        2       0.051     6.270     1.910x
       40000    parallel        4       0.027    11.803     3.595x
       40000    parallel        6       0.020    15.651     4.768x
       40000    parallel        8       0.023    14.220     4.332x
       40000    parallel       12       0.018    17.567     5.351x

       50000  sequential        -       0.122     3.272     1.000x
       50000    parallel        1       0.134     2.987     0.913x
       50000    parallel        2       0.064     6.233     1.905x
       50000    parallel        4       0.034    11.633     3.555x
       50000    parallel        6       0.025    15.736     4.809x
       50000    parallel        8       0.028    14.236     4.351x
       50000    parallel       12       0.022    18.022     5.508x

       75000  sequential        -       0.184     3.262     1.000x
       75000    parallel        1       0.201     2.992     0.917x
       75000    parallel        2       0.095     6.342     1.944x
       75000    parallel        4       0.052    11.593     3.554x
       75000    parallel        6       0.038    15.884     4.870x
       75000    parallel        8       0.042    14.319     4.390x
       75000    parallel       12       0.032    18.804     5.765x

      100000  sequential        -       0.245     3.265     1.000x
      100000    parallel        1       0.266     3.006     0.920x
      100000    parallel        2       0.126     6.338     1.941x
      100000    parallel        4       0.065    12.235     3.747x
      100000    parallel        6       0.051    15.548     4.761x
      100000    parallel        8       0.056    14.401     4.410x
      100000    parallel       12       0.046    17.437     5.340x

      200000  sequential        -       0.489     3.270     1.000x
      200000    parallel        1       0.532     3.006     0.919x
      200000    parallel        2       0.261     6.135     1.876x
      200000    parallel        4       0.138    11.636     3.558x
      200000    parallel        6       0.100    15.977     4.886x
      200000    parallel        8       0.109    14.630     4.474x
      200000    parallel       12       0.095    16.835     5.149x

      500000  sequential        -       1.221     3.277     1.000x
      500000    parallel        1       1.339     2.988     0.912x
      500000    parallel        2       0.631     6.344     1.936x
      500000    parallel        4       0.335    11.932     3.641x
      500000    parallel        6       0.250    15.990     4.880x
      500000    parallel        8       0.277    14.435     4.405x
      500000    parallel       12       0.223    17.907     5.465x

     1000000  sequential        -       2.453     3.262     1.000x
     1000000    parallel        1       2.685     2.979     0.913x
     1000000    parallel        2       1.236     6.474     1.985x
     1000000    parallel        4       0.675    11.856     3.635x
     1000000    parallel        6       0.524    15.276     4.683x
     1000000    parallel        8       0.595    13.447     4.123x
     1000000    parallel       12       0.447    17.904     5.489x

     2000000  sequential        -       5.024     3.185     1.000x
     2000000    parallel        1       5.484     2.918     0.916x
     2000000    parallel        2       2.872     5.570     1.749x
     2000000    parallel        4       1.629     9.825     3.085x
     2000000    parallel        6       1.507    10.618     3.334x
     2000000    parallel        8       1.281    12.492     3.923x
     2000000    parallel       12       2.191     7.302     2.293x

     5000000  sequential        -      12.646     3.163     1.000x
     5000000    parallel        1      13.631     2.935     0.928x
     5000000    parallel        2       6.463     6.189     1.957x
     5000000    parallel        4       3.868    10.342     3.270x
     5000000    parallel        6       2.929    13.655     4.317x
     5000000    parallel        8       2.986    13.398     4.236x
     5000000    parallel       12       5.826     6.865     2.170x

    10000000  sequential        -      24.927     3.209     1.000x
    10000000    parallel        1      27.170     2.944     0.917x
    10000000    parallel        2      13.089     6.112     1.904x
    10000000    parallel        4       7.294    10.967     3.417x
    10000000    parallel        6       5.518    14.498     4.517x
    10000000    parallel        8       5.971    13.397     4.174x
    10000000    parallel       12       5.937    13.474     4.198x

    33554432  sequential        -      84.234     3.187     1.000x
    33554432    parallel        1      91.861     2.922     0.917x
    33554432    parallel        2      45.502     5.899     1.851x
    33554432    parallel        4      25.417    10.561     3.314x
    33554432    parallel        6      24.705    10.866     3.410x
    33554432    parallel        8      30.444     8.817     2.767x
    33554432    parallel       12      17.807    15.075     4.730x


== sigmoid ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.003     2.995     1.000x
        1000    parallel        1       0.003     2.881     0.962x
        1000    parallel        2       0.003     2.873     0.959x
        1000    parallel        4       0.003     2.879     0.961x
        1000    parallel        6       0.003     2.880     0.962x
        1000    parallel        8       0.003     2.878     0.961x
        1000    parallel       12       0.003     2.887     0.964x

       10000  sequential        -       0.027     3.000     1.000x
       10000    parallel        1       0.028     2.870     0.957x
       10000    parallel        2       0.028     2.814     0.938x
       10000    parallel        4       0.028     2.835     0.945x
       10000    parallel        6       0.028     2.837     0.946x
       10000    parallel        8       0.028     2.836     0.945x
       10000    parallel       12       0.028     2.837     0.945x

       32768  sequential        -       0.089     2.941     1.000x
       32768    parallel        1       0.093     2.833     0.963x
       32768    parallel        2       0.093     2.834     0.963x
       32768    parallel        4       0.093     2.832     0.963x
       32768    parallel        6       0.093     2.829     0.962x
       32768    parallel        8       0.093     2.827     0.961x
       32768    parallel       12       0.093     2.833     0.963x

       32769  sequential        -       0.090     2.922     1.000x
       32769    parallel        1       0.093     2.828     0.968x
       32769    parallel        2       0.048     5.508     1.885x
       32769    parallel        4       0.026    10.261     3.511x
       32769    parallel        6       0.026     9.949     3.405x
       32769    parallel        8       0.023    11.396     3.900x
       32769    parallel       12       0.020    13.163     4.504x

       40000  sequential        -       0.109     2.941     1.000x
       40000    parallel        1       0.113     2.829     0.962x
       40000    parallel        2       0.058     5.558     1.890x
       40000    parallel        4       0.030    10.558     3.590x
       40000    parallel        6       0.024    13.520     4.598x
       40000    parallel        8       0.028    11.362     3.864x
       40000    parallel       12       0.041     7.848     2.669x

       50000  sequential        -       0.136     2.940     1.000x
       50000    parallel        1       0.141     2.831     0.963x
       50000    parallel        2       0.072     5.542     1.885x
       50000    parallel        4       0.038    10.563     3.593x
       50000    parallel        6       0.025    16.182     5.504x
       50000    parallel        8       0.034    11.617     3.951x
       50000    parallel       12       0.034    11.797     4.012x

       75000  sequential        -       0.204     2.939     1.000x
       75000    parallel        1       0.212     2.829     0.962x
       75000    parallel        2       0.108     5.577     1.898x
       75000    parallel        4       0.058    10.333     3.515x
       75000    parallel        6       0.044    13.726     4.670x
       75000    parallel        8       0.054    11.198     3.810x
       75000    parallel       12       0.042    14.128     4.806x

      100000  sequential        -       0.272     2.940     1.000x
      100000    parallel        1       0.283     2.829     0.962x
      100000    parallel        2       0.143     5.578     1.897x
      100000    parallel        4       0.077    10.448     3.554x
      100000    parallel        6       0.059    13.501     4.592x
      100000    parallel        8       0.070    11.440     3.891x
      100000    parallel       12       0.057    14.048     4.778x

      200000  sequential        -       0.545     2.938     1.000x
      200000    parallel        1       0.565     2.831     0.964x
      200000    parallel        2       0.287     5.575     1.897x
      200000    parallel        4       0.153    10.482     3.567x
      200000    parallel        6       0.116    13.779     4.689x
      200000    parallel        8       0.138    11.615     3.953x
      200000    parallel       12       0.109    14.658     4.989x

      500000  sequential        -       1.348     2.968     1.000x
      500000    parallel        1       1.391     2.876     0.969x
      500000    parallel        2       0.785     5.097     1.717x
      500000    parallel        4       0.392    10.192     3.434x
      500000    parallel        6       0.294    13.625     4.591x
      500000    parallel        8       0.360    11.105     3.742x
      500000    parallel       12       0.287    13.938     4.696x

     1000000  sequential        -       2.752     2.907     1.000x
     1000000    parallel        1       2.876     2.782     0.957x
     1000000    parallel        2       1.435     5.577     1.918x
     1000000    parallel        4       0.804     9.952     3.423x
     1000000    parallel        6       0.585    13.679     4.705x
     1000000    parallel        8       0.722    11.077     3.810x
     1000000    parallel       12       0.663    12.073     4.153x

     2000000  sequential        -       5.526     2.895     1.000x
     2000000    parallel        1       5.739     2.788     0.963x
     2000000    parallel        2       2.924     5.472     1.890x
     2000000    parallel        4       1.654     9.672     3.341x
     2000000    parallel        6       1.290    12.398     4.282x
     2000000    parallel        8       1.449    11.042     3.814x
     2000000    parallel       12       1.377    11.623     4.014x

     5000000  sequential        -      13.712     2.917     1.000x
     5000000    parallel        1      14.514     2.756     0.945x
     5000000    parallel        2       7.422     5.389     1.847x
     5000000    parallel        4       4.092     9.775     3.351x
     5000000    parallel        6       3.074    13.013     4.461x
     5000000    parallel        8       3.682    10.864     3.724x
     5000000    parallel       12       3.593    11.132     3.816x

    10000000  sequential        -      27.674     2.891     1.000x
    10000000    parallel        1      29.888     2.677     0.926x
    10000000    parallel        2      16.403     4.877     1.687x
    10000000    parallel        4       9.447     8.468     2.929x
    10000000    parallel        6      10.086     7.932     2.744x
    10000000    parallel        8       8.388     9.538     3.299x
    10000000    parallel       12      13.425     5.959     2.061x

    33554432  sequential        -      93.958     2.857     1.000x
    33554432    parallel        1      96.528     2.781     0.973x
    33554432    parallel        2      50.090     5.359     1.876x
    33554432    parallel        4      28.324     9.477     3.317x
    33554432    parallel        6      21.243    12.636     4.423x
    33554432    parallel        8      24.695    10.870     3.805x
    33554432    parallel       12      27.154     9.886     3.460x


== dot ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001    12.129     1.000x
        1000    parallel        1       0.001    12.088     0.997x
        1000    parallel        2       0.001    12.082     0.996x
        1000    parallel        4       0.001    12.069     0.995x
        1000    parallel        6       0.001    12.052     0.994x
        1000    parallel        8       0.001    12.076     0.996x
        1000    parallel       12       0.001    12.036     0.992x

       10000  sequential        -       0.007    12.049     1.000x
       10000    parallel        1       0.007    11.956     0.992x
       10000    parallel        2       0.007    11.962     0.993x
       10000    parallel        4       0.007    11.950     0.992x
       10000    parallel        6       0.007    11.952     0.992x
       10000    parallel        8       0.007    12.001     0.996x
       10000    parallel       12       0.007    11.950     0.992x

       32768  sequential        -       0.022    12.039     1.000x
       32768    parallel        1       0.022    11.847     0.984x
       32768    parallel        2       0.022    11.950     0.993x
       32768    parallel        4       0.022    11.948     0.992x
       32768    parallel        6       0.022    11.911     0.989x
       32768    parallel        8       0.022    11.989     0.996x
       32768    parallel       12       0.022    12.015     0.998x

       32769  sequential        -       0.022    12.037     1.000x
       32769    parallel        1       0.022    12.011     0.998x
       32769    parallel        2       0.012    22.574     1.875x
       32769    parallel        4       0.006    41.003     3.406x
       32769    parallel        6       0.005    52.642     4.373x
       32769    parallel        8       0.005    49.812     4.138x
       32769    parallel       12       0.004    59.526     4.945x

       40000  sequential        -       0.027    12.026     1.000x
       40000    parallel        1       0.027    11.958     0.994x
       40000    parallel        2       0.014    22.738     1.891x
       40000    parallel        4       0.008    41.693     3.467x
       40000    parallel        6       0.006    54.249     4.511x
       40000    parallel        8       0.006    50.691     4.215x
       40000    parallel       12       0.005    62.483     5.196x

       50000  sequential        -       0.033    12.017     1.000x
       50000    parallel        1       0.033    12.030     1.001x
       50000    parallel        2       0.017    22.990     1.913x
       50000    parallel        4       0.010    41.432     3.448x
       50000    parallel        6       0.007    55.321     4.604x
       50000    parallel        8       0.008    51.747     4.306x
       50000    parallel       12       0.009    44.383     3.693x

       75000  sequential        -       0.050    11.976     1.000x
       75000    parallel        1       0.050    11.937     0.997x
       75000    parallel        2       0.026    23.084     1.928x
       75000    parallel        4       0.014    42.058     3.512x
       75000    parallel        6       0.010    58.127     4.854x
       75000    parallel        8       0.011    53.997     4.509x
       75000    parallel       12       0.009    65.077     5.434x

      100000  sequential        -       0.069    11.622     1.000x
      100000    parallel        1       0.067    11.948     1.028x
      100000    parallel        2       0.035    22.637     1.948x
      100000    parallel        4       0.018    44.279     3.810x
      100000    parallel        6       0.014    56.890     4.895x
      100000    parallel        8       0.015    54.873     4.722x
      100000    parallel       12       0.011    69.602     5.989x

      200000  sequential        -       0.133    12.003     1.000x
      200000    parallel        1       0.134    11.960     0.996x
      200000    parallel        2       0.069    23.229     1.935x
      200000    parallel        4       0.047    34.398     2.866x
      200000    parallel        6       0.028    57.856     4.820x
      200000    parallel        8       0.029    56.068     4.671x
      200000    parallel       12       0.023    69.399     5.782x

      500000  sequential        -       0.333    11.996     1.000x
      500000    parallel        1       0.334    11.991     1.000x
      500000    parallel        2       0.169    23.721     1.977x
      500000    parallel        4       0.089    44.917     3.744x
      500000    parallel        6       0.068    58.719     4.895x
      500000    parallel        8       0.072    55.750     4.647x
      500000    parallel       12       0.054    74.111     6.178x

     1000000  sequential        -       0.667    11.993     1.000x
     1000000    parallel        1       0.676    11.831     0.987x
     1000000    parallel        2       0.342    23.387     1.950x
     1000000    parallel        4       0.181    44.315     3.695x
     1000000    parallel        6       0.137    58.446     4.873x
     1000000    parallel        8       0.140    57.099     4.761x
     1000000    parallel       12       0.117    68.514     5.713x

     2000000  sequential        -       1.427    11.216     1.000x
     2000000    parallel        1       1.397    11.450     1.021x
     2000000    parallel        2       0.714    22.416     1.999x
     2000000    parallel        4       0.406    39.416     3.514x
     2000000    parallel        6       0.311    51.396     4.582x
     2000000    parallel        8       0.300    53.318     4.754x
     2000000    parallel       12       0.226    70.749     6.308x

     5000000  sequential        -       3.452    11.586     1.000x
     5000000    parallel        1       3.477    11.503     0.993x
     5000000    parallel        2       1.857    21.540     1.859x
     5000000    parallel        4       1.378    29.034     2.506x
     5000000    parallel        6       1.404    28.498     2.460x
     5000000    parallel        8       1.320    30.314     2.616x
     5000000    parallel       12       1.832    21.837     1.885x

    10000000  sequential        -       7.029    11.381     1.000x
    10000000    parallel        1       7.074    11.309     0.994x
    10000000    parallel        2       3.689    21.686     1.905x
    10000000    parallel        4       2.737    29.230     2.568x
    10000000    parallel        6       2.883    27.748     2.438x
    10000000    parallel        8       2.801    28.559     2.509x
    10000000    parallel       12       3.302    24.225     2.128x

    33554432  sequential        -      25.429    10.556     1.000x
    33554432    parallel        1      25.537    10.512     0.996x
    33554432    parallel        2      13.374    20.071     1.901x
    33554432    parallel        4      10.205    26.305     2.492x
    33554432    parallel        6      10.317    26.020     2.465x
    33554432    parallel        8       9.135    29.386     2.784x
    33554432    parallel       12      12.089    22.205     2.104x
```

</details>

<details>
<summary>Earlier results with a four-thread limit</summary>

These earlier measurements retain their original metadata and results. The main table above
uses two new complete runs with 1, 2, 4, 6, 8, and 12 threads for comparison within the same runs.

### Earlier measurements with a four-thread limit

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
| Power state | Power plan, charging state, and background activity were not recorded |
| Thread limit | `OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE`; measured with 1, 2, and 4 threads out of 12 hardware threads |
| Measurement date | 2026-10-31; CUDA, ROCm, and Metal backends disabled |

#### Results (two runs, sequential implementation as the baseline)

The baseline is the plain sequential kernel (`src/backends/cpu/sequential.cpp`: one loop, no OpenMP).
Cells show run 1 / run 2 from the two most recent complete runs, which include all 12 operations.
"4 threads" is the parallel kernel with the configured four-thread limit; "best" is the fastest
measured thread count among 1, 2, and 4, in brackets. These runs do not measure all 12 hardware threads.
The table uses the same four input sizes as Machine 1; the full outputs below also include the
smaller sizes and cutoff checks.

| Op | n | Sequential (ms) | Speedup, 4 threads | Speedup, best (threads) |
|---|---|---|---|---|
| add | 100,000 | 0.011 / 0.010 | 2.232x / 2.430x | 2.232x (4) / 2.430x (4) |
| add | 1,000,000 | 0.222 / 0.165 | 2.064x / 2.716x | 2.064x (4) / 2.716x (4) |
| add | 10,000,000 | 7.792 / 6.100 | 0.858x / 0.976x | 0.915x (1) / 1.018x (2) |
| add | 33,554,432 | 26.601 / 23.978 | 1.088x / 1.175x | 1.189x (2) / 1.175x (4) |
| sub | 100,000 | 0.010 / 0.011 | 2.318x / 2.749x | 2.318x (4) / 2.749x (4) |
| sub | 1,000,000 | 0.182 / 0.401 | 3.732x / 5.148x | 3.732x (4) / 5.148x (4) |
| sub | 10,000,000 | 7.557 / 6.756 | 1.124x / 1.108x | 1.140x (2) / 1.134x (2) |
| sub | 33,554,432 | 25.144 / 23.128 | 1.061x / 1.049x | 1.061x (4) / 1.104x (2) |
| mul | 100,000 | 0.011 / 0.011 | 2.664x / 2.777x | 2.664x (4) / 2.777x (4) |
| mul | 1,000,000 | 0.252 / 0.228 | 3.990x / 2.960x | 3.990x (4) / 2.960x (4) |
| mul | 10,000,000 | 6.451 / 6.745 | 0.870x / 1.110x | 0.996x (2) / 1.110x (4) |
| mul | 33,554,432 | 24.025 / 23.347 | 1.069x / 0.973x | 1.130x (2) / 1.083x (2) |
| div | 100,000 | 0.020 / 0.019 | 3.474x / 3.127x | 3.474x (4) / 3.127x (4) |
| div | 1,000,000 | 0.267 / 0.346 | 3.316x / 3.818x | 3.316x (4) / 3.818x (4) |
| div | 10,000,000 | 8.502 / 6.836 | 1.237x / 1.068x | 1.237x (4) / 1.144x (2) |
| div | 33,554,432 | 33.921 / 23.670 | 1.345x / 0.902x | 1.404x (1) / 1.116x (2) |
| neg | 100,000 | 0.007 / 0.007 | 1.701x / 1.296x | 1.701x (4) / 1.296x (4) |
| neg | 1,000,000 | 0.124 / 0.081 | 2.682x / 1.946x | 2.682x (4) / 1.946x (4) |
| neg | 10,000,000 | 4.819 / 5.097 | 1.060x / 1.123x | 1.060x (4) / 1.142x (2) |
| neg | 33,554,432 | 16.738 / 17.906 | 1.056x / 1.093x | 1.093x (2) / 1.129x (2) |
| abs | 100,000 | 0.013 / 0.012 | 5.526x / 4.513x | 5.526x (4) / 4.513x (4) |
| abs | 1,000,000 | 0.164 / 0.265 | 5.133x / 5.004x | 5.133x (4) / 5.004x (4) |
| abs | 10,000,000 | 5.200 / 7.460 | 1.162x / 1.454x | 1.179x (2) / 1.492x (2) |
| abs | 33,554,432 | 17.686 / 18.304 | 1.098x / 1.022x | 1.098x (4) / 1.030x (2) |
| square | 100,000 | 0.007 / 0.007 | 2.919x / 2.844x | 2.919x (4) / 2.844x (4) |
| square | 1,000,000 | 0.113 / 0.129 | 4.290x / 4.609x | 4.290x (4) / 4.609x (4) |
| square | 10,000,000 | 4.626 / 5.214 | 1.046x / 1.079x | 1.058x (2) / 1.112x (2) |
| square | 33,554,432 | 23.210 / 17.810 | 1.222x / 1.041x | 1.222x (4) / 1.135x (2) |
| relu | 100,000 | 0.009 / 0.008 | 2.339x / 2.066x | 2.339x (4) / 2.066x (4) |
| relu | 1,000,000 | 0.103 / 0.093 | 2.601x / 2.312x | 2.601x (4) / 2.312x (4) |
| relu | 10,000,000 | 5.006 / 5.631 | 1.104x / 1.115x | 1.104x (4) / 1.115x (4) |
| relu | 33,554,432 | 17.627 / 21.014 | 0.929x / 1.301x | 0.929x (4) / 1.301x (4) |
| sqrt | 100,000 | 0.112 / 0.111 | 3.817x / 3.910x | 3.817x (4) / 3.910x (4) |
| sqrt | 1,000,000 | 1.112 / 1.119 | 3.707x / 4.023x | 3.707x (4) / 4.023x (4) |
| sqrt | 10,000,000 | 11.405 / 11.524 | 2.297x / 2.433x | 2.297x (4) / 2.433x (4) |
| sqrt | 33,554,432 | 39.141 / 40.082 | 2.486x / 2.322x | 2.486x (4) / 2.322x (4) |
| exp | 100,000 | 0.295 / 0.255 | 2.712x / 3.303x | 2.712x (4) / 3.303x (4) |
| exp | 1,000,000 | 2.700 / 2.514 | 3.218x / 3.119x | 3.218x (4) / 3.119x (4) |
| exp | 10,000,000 | 26.086 / 25.884 | 3.262x / 2.997x | 3.262x (4) / 2.997x (4) |
| exp | 33,554,432 | 90.810 / 84.488 | 2.855x / 3.391x | 2.855x (4) / 3.391x (4) |
| sigmoid | 100,000 | 0.274 / 0.280 | 3.280x / 3.334x | 3.280x (4) / 3.334x (4) |
| sigmoid | 1,000,000 | 2.787 / 2.746 | 3.247x / 3.263x | 3.247x (4) / 3.263x (4) |
| sigmoid | 10,000,000 | 30.067 / 27.505 | 3.351x / 3.207x | 3.351x (4) / 3.207x (4) |
| sigmoid | 33,554,432 | 95.447 / 91.080 | 3.071x / 3.000x | 3.071x (4) / 3.000x (4) |
| dot | 100,000 | 0.068 / 0.070 | 2.963x / 3.515x | 2.963x (4) / 3.515x (4) |
| dot | 1,000,000 | 0.743 / 0.692 | 2.602x / 2.820x | 2.602x (4) / 2.820x (4) |
| dot | 10,000,000 | 7.204 / 7.089 | 2.202x / 2.024x | 2.202x (4) / 2.024x (4) |
| dot | 33,554,432 | 24.847 / 25.807 | 2.574x / 2.325x | 2.574x (4) / 2.325x (4) |

What it shows:

- Simple elementwise operations gain from threads on many smaller inputs, but gains often shrink
  on large arrays. At 33,554,432 elements, four-thread speedups for `add`, `sub`, `mul`, `div`,
  `neg`, `abs`, `square`, and `relu` range from about 0.90x to 1.35x across the two runs.
  This is consistent with data movement becoming the main cost.
- `sqrt`, `exp`, and `sigmoid` keep larger gains at 33,554,432 elements: about 2.32-2.49x,
  2.85-3.39x, and 3.00-3.07x respectively with four threads. They do more computation per element
  than the simple unary operations.
- `dot` reaches about 2.33-2.57x with four threads at the largest size.
- The existing cutoff remains 32,768 elements. Rows at or below it use the serial fallback,
  regardless of the requested thread count. Most operations benefit at the first parallel size,
  32,769, but `neg` varies from 0.983x to 2.481x with four threads. These results support keeping
  the cutoff as an initial setting, rather than treating it as a measured optimum.
- Every measured elementwise result matched the sequential output bit for bit. `dot` passed the
  benchmark's tolerance check, since parallel summation can change the rounding order.

Reliability of these numbers: some timings differ by more than 10% between runs, and power and
background conditions were not recorded. Both runs are retained, including slower results.
Comparisons with Machine 1 also involve a different compiler and thread limit. Read the broad
trends before drawing conclusions from an individual ratio.

Benchmark method note: kernels write into preallocated buffers whose starting offsets are
staggered by 1 KB, using the same allocation arrangement as Machine 1's corrected benchmark.
Each measurement starts with a warmup, followed by nine timed samples whose median is reported,
with repeated calls for fast operations. Tensor allocation, dispatch, packing, and input preparation
are outside the timed calls. Unary operations and `dot` count 8 logical bytes per element; binary elementwise operations
count 12. `sqrt` uses nonnegative inputs; the other unary operations use values in [-0.75, 0.75].
Times are printed to three decimal places; 0.000 ms means below print precision, and speedups
are calculated from unrounded timings.

Run with the Release CPU build and `MENSURFLOW_NATIVE=OFF`:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark
```

<details>
<summary>Full output: cpu_parallel_benchmark, Machine 3, run 1</summary>

```text
Sequential vs parallel on the CPU (float32)
compiler: gcc 13.3.0, OpenMP: yes, max threads: 4, hardware threads: 12, parallel cutoff: 32768 elements
sequential = plain loop, no OpenMP. parallel = parallel kernel with N threads.
Speedup = sequential time / this time. Median of 9 samples.

== add ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   186.264     1.000x
        1000    parallel        1       0.000   187.207     1.005x
        1000    parallel        2       0.000   187.048     1.004x
        1000    parallel        4       0.000   184.595     0.991x

       10000  sequential        -       0.001   133.063     1.000x
       10000    parallel        1       0.001   133.533     1.004x
       10000    parallel        2       0.001   136.231     1.024x
       10000    parallel        4       0.001   136.203     1.024x

       32768  sequential        -       0.003   131.092     1.000x
       32768    parallel        1       0.003   129.621     0.989x
       32768    parallel        2       0.003   131.541     1.003x
       32768    parallel        4       0.003   130.802     0.998x

       32769  sequential        -       0.003   131.799     1.000x
       32769    parallel        1       0.003   129.745     0.984x
       32769    parallel        2       0.003   157.074     1.192x
       32769    parallel        4       0.002   227.171     1.724x

       40000  sequential        -       0.004   127.525     1.000x
       40000    parallel        1       0.004   125.959     0.988x
       40000    parallel        2       0.003   148.391     1.164x
       40000    parallel        4       0.002   224.938     1.764x

       50000  sequential        -       0.006   102.122     1.000x
       50000    parallel        1       0.006   105.716     1.035x
       50000    parallel        2       0.004   139.924     1.370x
       50000    parallel        4       0.004   139.971     1.371x

       75000  sequential        -       0.009    96.865     1.000x
       75000    parallel        1       0.009    99.046     1.023x
       75000    parallel        2       0.008   112.051     1.157x
       75000    parallel        4       0.004   229.555     2.370x

      100000  sequential        -       0.011   109.090     1.000x
      100000    parallel        1       0.013    94.994     0.871x
      100000    parallel        2       0.009   131.332     1.204x
      100000    parallel        4       0.005   243.529     2.232x

      200000  sequential        -       0.022   108.672     1.000x
      200000    parallel        1       0.022   107.338     0.988x
      200000    parallel        2       0.016   152.780     1.406x
      200000    parallel        4       0.010   234.710     2.160x

      500000  sequential        -       0.065    92.541     1.000x
      500000    parallel        1       0.060   100.468     1.086x
      500000    parallel        2       0.040   149.481     1.615x
      500000    parallel        4       0.024   250.147     2.703x

     1000000  sequential        -       0.222    53.940     1.000x
     1000000    parallel        1       0.251    47.778     0.886x
     1000000    parallel        2       0.149    80.419     1.491x
     1000000    parallel        4       0.108   111.314     2.064x

     2000000  sequential        -       1.296    18.517     1.000x
     2000000    parallel        1       1.380    17.387     0.939x
     2000000    parallel        2       1.345    17.848     0.964x
     2000000    parallel        4       1.229    19.526     1.054x

     5000000  sequential        -       3.449    17.396     1.000x
     5000000    parallel        1       3.681    16.298     0.937x
     5000000    parallel        2       2.916    20.577     1.183x
     5000000    parallel        4       3.372    17.792     1.023x

    10000000  sequential        -       7.792    15.401     1.000x
    10000000    parallel        1       8.517    14.090     0.915x
    10000000    parallel        2       9.241    12.985     0.843x
    10000000    parallel        4       9.081    13.214     0.858x

    33554432  sequential        -      26.601    15.137     1.000x
    33554432    parallel        1      24.563    16.393     1.083x
    33554432    parallel        2      22.379    17.992     1.189x
    33554432    parallel        4      24.455    16.465     1.088x


== sub ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   188.160     1.000x
        1000    parallel        1       0.000   184.466     0.980x
        1000    parallel        2       0.000   161.138     0.856x
        1000    parallel        4       0.000   160.092     0.851x

       10000  sequential        -       0.001   127.134     1.000x
       10000    parallel        1       0.001   134.670     1.059x
       10000    parallel        2       0.001   137.588     1.082x
       10000    parallel        4       0.001   140.291     1.103x

       32768  sequential        -       0.003   136.781     1.000x
       32768    parallel        1       0.003   134.023     0.980x
       32768    parallel        2       0.003   133.942     0.979x
       32768    parallel        4       0.003   134.330     0.982x

       32769  sequential        -       0.003   136.152     1.000x
       32769    parallel        1       0.003   132.878     0.976x
       32769    parallel        2       0.002   158.892     1.167x
       32769    parallel        4       0.002   216.815     1.592x

       40000  sequential        -       0.004   130.554     1.000x
       40000    parallel        1       0.004   131.902     1.010x
       40000    parallel        2       0.003   146.414     1.121x
       40000    parallel        4       0.002   223.713     1.714x

       50000  sequential        -       0.005   114.485     1.000x
       50000    parallel        1       0.005   121.751     1.063x
       50000    parallel        2       0.004   164.317     1.435x
       50000    parallel        4       0.003   233.806     2.042x

       75000  sequential        -       0.009   105.796     1.000x
       75000    parallel        1       0.008   116.764     1.104x
       75000    parallel        2       0.005   174.254     1.647x
       75000    parallel        4       0.003   280.854     2.655x

      100000  sequential        -       0.010   126.023     1.000x
      100000    parallel        1       0.010   124.512     0.988x
      100000    parallel        2       0.007   169.817     1.348x
      100000    parallel        4       0.004   292.173     2.318x

      200000  sequential        -       0.022   111.142     1.000x
      200000    parallel        1       0.021   115.184     1.036x
      200000    parallel        2       0.015   156.431     1.407x
      200000    parallel        4       0.009   261.699     2.355x

      500000  sequential        -       0.057   105.679     1.000x
      500000    parallel        1       0.055   109.178     1.033x
      500000    parallel        2       0.035   170.072     1.609x
      500000    parallel        4       0.022   272.267     2.576x

     1000000  sequential        -       0.182    65.849     1.000x
     1000000    parallel        1       0.159    75.308     1.144x
     1000000    parallel        2       0.096   124.722     1.894x
     1000000    parallel        4       0.049   245.778     3.732x

     2000000  sequential        -       0.855    28.056     1.000x
     2000000    parallel        1       0.869    27.625     0.985x
     2000000    parallel        2       0.695    34.526     1.231x
     2000000    parallel        4       0.694    34.572     1.232x

     5000000  sequential        -       3.133    19.154     1.000x
     5000000    parallel        1       3.262    18.394     0.960x
     5000000    parallel        2       2.892    20.747     1.083x
     5000000    parallel        4       2.908    20.635     1.077x

    10000000  sequential        -       7.557    15.879     1.000x
    10000000    parallel        1       7.768    15.448     0.973x
    10000000    parallel        2       6.629    18.103     1.140x
    10000000    parallel        4       6.725    17.845     1.124x

    33554432  sequential        -      25.144    16.014     1.000x
    33554432    parallel        1      24.081    16.721     1.044x
    33554432    parallel        2      24.430    16.482     1.029x
    33554432    parallel        4      23.708    16.984     1.061x


== mul ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   188.031     1.000x
        1000    parallel        1       0.000   185.363     0.986x
        1000    parallel        2       0.000   184.445     0.981x
        1000    parallel        4       0.000   185.217     0.985x

       10000  sequential        -       0.001   126.499     1.000x
       10000    parallel        1       0.001   131.225     1.037x
       10000    parallel        2       0.001   134.465     1.063x
       10000    parallel        4       0.001   134.380     1.062x

       32768  sequential        -       0.003   135.000     1.000x
       32768    parallel        1       0.003   134.792     0.998x
       32768    parallel        2       0.003   135.243     1.002x
       32768    parallel        4       0.003   132.234     0.980x

       32769  sequential        -       0.003   130.827     1.000x
       32769    parallel        1       0.003   130.221     0.995x
       32769    parallel        2       0.003   147.382     1.127x
       32769    parallel        4       0.002   230.377     1.761x

       40000  sequential        -       0.004   131.264     1.000x
       40000    parallel        1       0.004   128.351     0.978x
       40000    parallel        2       0.003   157.422     1.199x
       40000    parallel        4       0.002   238.200     1.815x

       50000  sequential        -       0.005   125.418     1.000x
       50000    parallel        1       0.005   121.369     0.968x
       50000    parallel        2       0.004   168.793     1.346x
       50000    parallel        4       0.002   266.653     2.126x

       75000  sequential        -       0.007   121.820     1.000x
       75000    parallel        1       0.008   112.657     0.925x
       75000    parallel        2       0.005   167.845     1.378x
       75000    parallel        4       0.003   263.722     2.165x

      100000  sequential        -       0.011   107.539     1.000x
      100000    parallel        1       0.011   109.736     1.020x
      100000    parallel        2       0.008   143.574     1.335x
      100000    parallel        4       0.004   286.474     2.664x

      200000  sequential        -       0.023   103.415     1.000x
      200000    parallel        1       0.019   124.289     1.202x
      200000    parallel        2       0.014   168.047     1.625x
      200000    parallel        4       0.009   274.022     2.650x

      500000  sequential        -       0.058   104.312     1.000x
      500000    parallel        1       0.057   106.076     1.017x
      500000    parallel        2       0.037   161.684     1.550x
      500000    parallel        4       0.024   253.838     2.433x

     1000000  sequential        -       0.252    47.534     1.000x
     1000000    parallel        1       0.179    66.879     1.407x
     1000000    parallel        2       0.097   123.649     2.601x
     1000000    parallel        4       0.063   189.665     3.990x

     2000000  sequential        -       0.959    25.022     1.000x
     2000000    parallel        1       0.864    27.792     1.111x
     2000000    parallel        2       0.723    33.193     1.327x
     2000000    parallel        4       0.797    30.105     1.203x

     5000000  sequential        -       2.926    20.508     1.000x
     5000000    parallel        1       3.005    19.969     0.974x
     5000000    parallel        2       2.824    21.246     1.036x
     5000000    parallel        4       2.942    20.396     0.995x

    10000000  sequential        -       6.451    18.602     1.000x
    10000000    parallel        1       6.848    17.524     0.942x
    10000000    parallel        2       6.475    18.532     0.996x
    10000000    parallel        4       7.413    16.188     0.870x

    33554432  sequential        -      24.025    16.760     1.000x
    33554432    parallel        1      24.414    16.493     0.984x
    33554432    parallel        2      21.266    18.934     1.130x
    33554432    parallel        4      22.481    17.911     1.069x


== div ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    62.156     1.000x
        1000    parallel        1       0.000    61.836     0.995x
        1000    parallel        2       0.000    62.313     1.003x
        1000    parallel        4       0.000    62.039     0.998x

       10000  sequential        -       0.002    61.980     1.000x
       10000    parallel        1       0.002    62.638     1.011x
       10000    parallel        2       0.002    63.009     1.017x
       10000    parallel        4       0.002    63.126     1.018x

       32768  sequential        -       0.006    61.973     1.000x
       32768    parallel        1       0.006    62.006     1.001x
       32768    parallel        2       0.006    62.174     1.003x
       32768    parallel        4       0.006    62.387     1.007x

       32769  sequential        -       0.006    62.349     1.000x
       32769    parallel        1       0.006    63.127     1.012x
       32769    parallel        2       0.004   110.728     1.776x
       32769    parallel        4       0.002   182.703     2.930x

       40000  sequential        -       0.008    61.641     1.000x
       40000    parallel        1       0.008    62.725     1.018x
       40000    parallel        2       0.004   112.519     1.825x
       40000    parallel        4       0.003   188.543     3.059x

       50000  sequential        -       0.010    62.109     1.000x
       50000    parallel        1       0.010    62.295     1.003x
       50000    parallel        2       0.005   110.747     1.783x
       50000    parallel        4       0.003   173.887     2.800x

       75000  sequential        -       0.015    61.952     1.000x
       75000    parallel        1       0.014    65.105     1.051x
       75000    parallel        2       0.008   114.634     1.850x
       75000    parallel        4       0.004   207.308     3.346x

      100000  sequential        -       0.020    61.459     1.000x
      100000    parallel        1       0.019    62.217     1.012x
      100000    parallel        2       0.011   112.696     1.834x
      100000    parallel        4       0.006   213.526     3.474x

      200000  sequential        -       0.039    61.454     1.000x
      200000    parallel        1       0.038    63.538     1.034x
      200000    parallel        2       0.021   111.981     1.822x
      200000    parallel        4       0.012   193.037     3.141x

      500000  sequential        -       0.102    58.847     1.000x
      500000    parallel        1       0.099    60.882     1.035x
      500000    parallel        2       0.051   118.023     2.006x
      500000    parallel        4       0.034   174.640     2.968x

     1000000  sequential        -       0.267    44.988     1.000x
     1000000    parallel        1       0.261    45.970     1.022x
     1000000    parallel        2       0.166    72.493     1.611x
     1000000    parallel        4       0.080   149.199     3.316x

     2000000  sequential        -       0.997    24.075     1.000x
     2000000    parallel        1       0.962    24.940     1.036x
     2000000    parallel        2       1.020    23.532     0.977x
     2000000    parallel        4       0.956    25.097     1.042x

     5000000  sequential        -       3.167    18.946     1.000x
     5000000    parallel        1       3.713    16.161     0.853x
     5000000    parallel        2       3.330    18.020     0.951x
     5000000    parallel        4       3.206    18.712     0.988x

    10000000  sequential        -       8.502    14.114     1.000x
    10000000    parallel        1       7.854    15.278     1.083x
    10000000    parallel        2       7.214    16.633     1.178x
    10000000    parallel        4       6.875    17.454     1.237x

    33554432  sequential        -      33.921    11.870     1.000x
    33554432    parallel        1      24.154    16.670     1.404x
    33554432    parallel        2      25.314    15.906     1.340x
    33554432    parallel        4      25.219    15.966     1.345x


== neg ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   124.381     1.000x
        1000    parallel        1       0.000   124.245     0.999x
        1000    parallel        2       0.000   121.757     0.979x
        1000    parallel        4       0.000   121.559     0.977x

       10000  sequential        -       0.001   132.287     1.000x
       10000    parallel        1       0.001   127.759     0.966x
       10000    parallel        2       0.001   128.876     0.974x
       10000    parallel        4       0.001   130.059     0.983x

       32768  sequential        -       0.002   135.563     1.000x
       32768    parallel        1       0.002   135.935     1.003x
       32768    parallel        2       0.002   134.506     0.992x
       32768    parallel        4       0.002   135.200     0.997x

       32769  sequential        -       0.002   134.222     1.000x
       32769    parallel        1       0.002   133.502     0.995x
       32769    parallel        2       0.003   103.026     0.768x
       32769    parallel        4       0.002   131.937     0.983x

       40000  sequential        -       0.002   135.012     1.000x
       40000    parallel        1       0.003   127.419     0.944x
       40000    parallel        2       0.003   110.013     0.815x
       40000    parallel        4       0.002   162.747     1.205x

       50000  sequential        -       0.003   130.783     1.000x
       50000    parallel        1       0.003   130.307     0.996x
       50000    parallel        2       0.004   113.409     0.867x
       50000    parallel        4       0.002   176.690     1.351x

       75000  sequential        -       0.005   117.787     1.000x
       75000    parallel        1       0.005   117.108     0.994x
       75000    parallel        2       0.005   121.241     1.029x
       75000    parallel        4       0.003   202.206     1.717x

      100000  sequential        -       0.007   122.126     1.000x
      100000    parallel        1       0.007   119.679     0.980x
      100000    parallel        2       0.007   115.243     0.944x
      100000    parallel        4       0.004   207.775     1.701x

      200000  sequential        -       0.014   116.430     1.000x
      200000    parallel        1       0.014   116.942     1.004x
      200000    parallel        2       0.013   119.525     1.027x
      200000    parallel        4       0.008   201.662     1.732x

      500000  sequential        -       0.047    85.679     1.000x
      500000    parallel        1       0.053    74.787     0.873x
      500000    parallel        2       0.039   101.830     1.189x
      500000    parallel        4       0.021   188.548     2.201x

     1000000  sequential        -       0.124    64.478     1.000x
     1000000    parallel        1       0.116    68.990     1.070x
     1000000    parallel        2       0.079   101.285     1.571x
     1000000    parallel        4       0.046   172.926     2.682x

     2000000  sequential        -       0.533    30.030     1.000x
     2000000    parallel        1       0.525    30.484     1.015x
     2000000    parallel        2       0.486    32.922     1.096x
     2000000    parallel        4       0.272    58.928     1.962x

     5000000  sequential        -       2.291    17.459     1.000x
     5000000    parallel        1       2.146    18.639     1.068x
     5000000    parallel        2       2.112    18.936     1.085x
     5000000    parallel        4       2.286    17.498     1.002x

    10000000  sequential        -       4.819    16.600     1.000x
    10000000    parallel        1       5.088    15.724     0.947x
    10000000    parallel        2       4.734    16.898     1.018x
    10000000    parallel        4       4.546    17.599     1.060x

    33554432  sequential        -      16.738    16.037     1.000x
    33554432    parallel        1      16.404    16.364     1.020x
    33554432    parallel        2      15.307    17.536     1.093x
    33554432    parallel        4      15.858    16.928     1.056x


== abs ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    66.993     1.000x
        1000    parallel        1       0.000   125.643     1.875x
        1000    parallel        2       0.000   125.441     1.872x
        1000    parallel        4       0.000   126.172     1.883x

       10000  sequential        -       0.001    70.100     1.000x
       10000    parallel        1       0.001   131.079     1.870x
       10000    parallel        2       0.001   128.835     1.838x
       10000    parallel        4       0.001   127.628     1.821x

       32768  sequential        -       0.004    69.848     1.000x
       32768    parallel        1       0.002   135.739     1.943x
       32768    parallel        2       0.002   135.095     1.934x
       32768    parallel        4       0.002   135.927     1.946x

       32769  sequential        -       0.004    70.188     1.000x
       32769    parallel        1       0.002   136.039     1.938x
       32769    parallel        2       0.002   166.394     2.371x
       32769    parallel        4       0.001   228.379     3.254x

       40000  sequential        -       0.005    70.020     1.000x
       40000    parallel        1       0.002   131.207     1.874x
       40000    parallel        2       0.002   182.562     2.607x
       40000    parallel        4       0.001   246.679     3.523x

       50000  sequential        -       0.006    70.174     1.000x
       50000    parallel        1       0.003   130.323     1.857x
       50000    parallel        2       0.002   194.057     2.765x
       50000    parallel        4       0.002   240.964     3.434x

       75000  sequential        -       0.009    65.918     1.000x
       75000    parallel        1       0.006   107.517     1.631x
       75000    parallel        2       0.003   186.389     2.828x
       75000    parallel        4       0.002   253.224     3.841x

      100000  sequential        -       0.013    63.672     1.000x
      100000    parallel        1       0.008   106.613     1.674x
      100000    parallel        2       0.004   211.217     3.317x
      100000    parallel        4       0.002   351.852     5.526x

      200000  sequential        -       0.023    69.102     1.000x
      200000    parallel        1       0.014   117.375     1.699x
      200000    parallel        2       0.009   178.415     2.582x
      200000    parallel        4       0.005   323.616     4.683x

      500000  sequential        -       0.063    63.031     1.000x
      500000    parallel        1       0.039   101.374     1.608x
      500000    parallel        2       0.025   160.940     2.553x
      500000    parallel        4       0.015   272.823     4.328x

     1000000  sequential        -       0.164    48.645     1.000x
     1000000    parallel        1       0.105    76.299     1.568x
     1000000    parallel        2       0.053   150.284     3.089x
     1000000    parallel        4       0.032   249.681     5.133x

     2000000  sequential        -       0.667    23.996     1.000x
     2000000    parallel        1       0.533    30.017     1.251x
     2000000    parallel        2       0.351    45.623     1.901x
     2000000    parallel        4       0.224    71.381     2.975x

     5000000  sequential        -       2.516    15.901     1.000x
     5000000    parallel        1       2.315    17.279     1.087x
     5000000    parallel        2       2.218    18.038     1.134x
     5000000    parallel        4       2.171    18.423     1.159x

    10000000  sequential        -       5.200    15.384     1.000x
    10000000    parallel        1       4.666    17.147     1.115x
    10000000    parallel        2       4.412    18.132     1.179x
    10000000    parallel        4       4.476    17.875     1.162x

    33554432  sequential        -      17.686    15.178     1.000x
    33554432    parallel        1      18.081    14.847     0.978x
    33554432    parallel        2      16.624    16.147     1.064x
    33554432    parallel        4      16.101    16.672     1.098x


== square ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   124.758     1.000x
        1000    parallel        1       0.000   121.485     0.974x
        1000    parallel        2       0.000   122.826     0.985x
        1000    parallel        4       0.000   122.910     0.985x

       10000  sequential        -       0.001   115.333     1.000x
       10000    parallel        1       0.001   125.586     1.089x
       10000    parallel        2       0.001    96.398     0.836x
       10000    parallel        4       0.001    94.848     0.822x

       32768  sequential        -       0.003    85.500     1.000x
       32768    parallel        1       0.003    96.889     1.133x
       32768    parallel        2       0.002   117.239     1.371x
       32768    parallel        4       0.002   130.552     1.527x

       32769  sequential        -       0.002   122.380     1.000x
       32769    parallel        1       0.002   124.233     1.015x
       32769    parallel        2       0.002   162.506     1.328x
       32769    parallel        4       0.001   211.651     1.729x

       40000  sequential        -       0.003   123.146     1.000x
       40000    parallel        1       0.002   128.740     1.045x
       40000    parallel        2       0.002   175.092     1.422x
       40000    parallel        4       0.001   241.767     1.963x

       50000  sequential        -       0.004   109.135     1.000x
       50000    parallel        1       0.004   114.072     1.045x
       50000    parallel        2       0.002   173.202     1.587x
       50000    parallel        4       0.002   241.452     2.212x

       75000  sequential        -       0.005   116.242     1.000x
       75000    parallel        1       0.005   117.739     1.013x
       75000    parallel        2       0.003   207.469     1.785x
       75000    parallel        4       0.002   308.367     2.653x

      100000  sequential        -       0.007   119.179     1.000x
      100000    parallel        1       0.008   105.857     0.888x
      100000    parallel        2       0.004   203.538     1.708x
      100000    parallel        4       0.002   347.944     2.919x

      200000  sequential        -       0.015   105.483     1.000x
      200000    parallel        1       0.017    94.956     0.900x
      200000    parallel        2       0.009   168.978     1.602x
      200000    parallel        4       0.004   375.170     3.557x

      500000  sequential        -       0.039   101.883     1.000x
      500000    parallel        1       0.046    87.727     0.861x
      500000    parallel        2       0.025   161.679     1.587x
      500000    parallel        4       0.014   286.905     2.816x

     1000000  sequential        -       0.113    70.533     1.000x
     1000000    parallel        1       0.164    48.637     0.690x
     1000000    parallel        2       0.072   111.333     1.578x
     1000000    parallel        4       0.026   302.586     4.290x

     2000000  sequential        -       0.856    18.682     1.000x
     2000000    parallel        1       0.498    32.144     1.721x
     2000000    parallel        2       0.303    52.797     2.826x
     2000000    parallel        4       0.210    76.127     4.075x

     5000000  sequential        -       2.249    17.789     1.000x
     5000000    parallel        1       2.204    18.149     1.020x
     5000000    parallel        2       1.964    20.365     1.145x
     5000000    parallel        4       2.148    18.621     1.047x

    10000000  sequential        -       4.626    17.292     1.000x
    10000000    parallel        1       4.945    16.177     0.936x
    10000000    parallel        2       4.372    18.300     1.058x
    10000000    parallel        4       4.425    18.079     1.046x

    33554432  sequential        -      23.210    11.566     1.000x
    33554432    parallel        1      22.051    12.173     1.053x
    33554432    parallel        2      19.397    13.839     1.197x
    33554432    parallel        4      18.994    14.133     1.222x


== relu ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   108.823     1.000x
        1000    parallel        1       0.000    66.202     0.608x
        1000    parallel        2       0.000    66.196     0.608x
        1000    parallel        4       0.000    66.234     0.609x

       10000  sequential        -       0.001    95.615     1.000x
       10000    parallel        1       0.001    67.407     0.705x
       10000    parallel        2       0.001    67.076     0.702x
       10000    parallel        4       0.001    66.121     0.692x

       32768  sequential        -       0.003   101.880     1.000x
       32768    parallel        1       0.004    68.531     0.673x
       32768    parallel        2       0.004    67.204     0.660x
       32768    parallel        4       0.004    68.639     0.674x

       32769  sequential        -       0.003   101.697     1.000x
       32769    parallel        1       0.004    67.792     0.667x
       32769    parallel        2       0.002   105.404     1.036x
       32769    parallel        4       0.002   145.795     1.434x

       40000  sequential        -       0.003    98.786     1.000x
       40000    parallel        1       0.005    67.108     0.679x
       40000    parallel        2       0.003    95.351     0.965x
       40000    parallel        4       0.002   129.645     1.312x

       50000  sequential        -       0.004    97.346     1.000x
       50000    parallel        1       0.006    67.153     0.690x
       50000    parallel        2       0.004   113.922     1.170x
       50000    parallel        4       0.003   153.904     1.581x

       75000  sequential        -       0.007    80.855     1.000x
       75000    parallel        1       0.010    62.951     0.779x
       75000    parallel        2       0.006   104.228     1.289x
       75000    parallel        4       0.003   180.783     2.236x

      100000  sequential        -       0.009    85.167     1.000x
      100000    parallel        1       0.013    62.813     0.738x
      100000    parallel        2       0.007   117.557     1.380x
      100000    parallel        4       0.004   199.220     2.339x

      200000  sequential        -       0.023    69.916     1.000x
      200000    parallel        1       0.026    62.411     0.893x
      200000    parallel        2       0.015   107.119     1.532x
      200000    parallel        4       0.008   212.697     3.042x

      500000  sequential        -       0.041    97.979     1.000x
      500000    parallel        1       0.060    66.213     0.676x
      500000    parallel        2       0.035   114.931     1.173x
      500000    parallel        4       0.020   201.218     2.054x

     1000000  sequential        -       0.103    77.469     1.000x
     1000000    parallel        1       0.147    54.497     0.703x
     1000000    parallel        2       0.065   123.116     1.589x
     1000000    parallel        4       0.040   201.526     2.601x

     2000000  sequential        -       0.529    30.232     1.000x
     2000000    parallel        1       0.524    30.543     1.010x
     2000000    parallel        2       0.424    37.715     1.248x
     2000000    parallel        4       0.403    39.727     1.314x

     5000000  sequential        -       2.416    16.559     1.000x
     5000000    parallel        1       2.299    17.401     1.051x
     5000000    parallel        2       1.944    20.576     1.243x
     5000000    parallel        4       2.196    18.211     1.100x

    10000000  sequential        -       5.006    15.981     1.000x
    10000000    parallel        1       5.339    14.985     0.938x
    10000000    parallel        2       4.976    16.076     1.006x
    10000000    parallel        4       4.535    17.639     1.104x

    33554432  sequential        -      17.627    15.228     1.000x
    33554432    parallel        1      21.741    12.347     0.811x
    33554432    parallel        2      19.499    13.767     0.904x
    33554432    parallel        4      18.966    14.154     0.929x


== sqrt ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001     7.057     1.000x
        1000    parallel        1       0.001     7.001     0.992x
        1000    parallel        2       0.001     6.807     0.965x
        1000    parallel        4       0.001     6.998     0.992x

       10000  sequential        -       0.011     7.052     1.000x
       10000    parallel        1       0.011     7.022     0.996x
       10000    parallel        2       0.011     7.024     0.996x
       10000    parallel        4       0.011     7.055     1.000x

       32768  sequential        -       0.037     7.084     1.000x
       32768    parallel        1       0.037     7.178     1.013x
       32768    parallel        2       0.038     6.959     0.982x
       32768    parallel        4       0.038     6.961     0.983x

       32769  sequential        -       0.038     6.948     1.000x
       32769    parallel        1       0.037     7.023     1.011x
       32769    parallel        2       0.019    13.703     1.972x
       32769    parallel        4       0.010    25.066     3.608x

       40000  sequential        -       0.045     7.093     1.000x
       40000    parallel        1       0.045     7.054     0.995x
       40000    parallel        2       0.024    13.297     1.875x
       40000    parallel        4       0.013    25.069     3.535x

       50000  sequential        -       0.057     7.022     1.000x
       50000    parallel        1       0.058     6.881     0.980x
       50000    parallel        2       0.030    13.528     1.926x
       50000    parallel        4       0.016    24.577     3.500x

       75000  sequential        -       0.086     6.950     1.000x
       75000    parallel        1       0.085     7.096     1.021x
       75000    parallel        2       0.043    14.075     2.025x
       75000    parallel        4       0.022    27.484     3.955x

      100000  sequential        -       0.112     7.158     1.000x
      100000    parallel        1       0.111     7.207     1.007x
      100000    parallel        2       0.056    14.196     1.983x
      100000    parallel        4       0.029    27.325     3.817x

      200000  sequential        -       0.223     7.186     1.000x
      200000    parallel        1       0.223     7.172     0.998x
      200000    parallel        2       0.112    14.272     1.986x
      200000    parallel        4       0.057    27.850     3.875x

      500000  sequential        -       0.558     7.170     1.000x
      500000    parallel        1       0.556     7.194     1.003x
      500000    parallel        2       0.279    14.318     1.997x
      500000    parallel        4       0.148    27.025     3.769x

     1000000  sequential        -       1.112     7.191     1.000x
     1000000    parallel        1       1.152     6.942     0.965x
     1000000    parallel        2       0.685    11.677     1.624x
     1000000    parallel        4       0.300    26.662     3.707x

     2000000  sequential        -       2.226     7.188     1.000x
     2000000    parallel        1       2.226     7.188     1.000x
     2000000    parallel        2       1.174    13.628     1.896x
     2000000    parallel        4       0.576    27.778     3.864x

     5000000  sequential        -       5.712     7.003     1.000x
     5000000    parallel        1       5.608     7.133     1.019x
     5000000    parallel        2       2.968    13.478     1.925x
     5000000    parallel        4       2.107    18.984     2.711x

    10000000  sequential        -      11.405     7.015     1.000x
    10000000    parallel        1      11.509     6.951     0.991x
    10000000    parallel        2       6.202    12.900     1.839x
    10000000    parallel        4       4.964    16.115     2.297x

    33554432  sequential        -      39.141     6.858     1.000x
    33554432    parallel        1      40.329     6.656     0.971x
    33554432    parallel        2      21.446    12.517     1.825x
    33554432    parallel        4      15.742    17.052     2.486x


== exp ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.002     3.202     1.000x
        1000    parallel        1       0.003     2.933     0.916x
        1000    parallel        2       0.003     2.934     0.916x
        1000    parallel        4       0.003     2.913     0.910x

       10000  sequential        -       0.025     3.203     1.000x
       10000    parallel        1       0.027     2.940     0.918x
       10000    parallel        2       0.027     2.923     0.913x
       10000    parallel        4       0.027     2.930     0.915x

       32768  sequential        -       0.085     3.088     1.000x
       32768    parallel        1       0.093     2.816     0.912x
       32768    parallel        2       0.094     2.796     0.906x
       32768    parallel        4       0.092     2.844     0.921x

       32769  sequential        -       0.083     3.159     1.000x
       32769    parallel        1       0.092     2.849     0.902x
       32769    parallel        2       0.047     5.549     1.757x
       32769    parallel        4       0.036     7.266     2.300x

       40000  sequential        -       0.117     2.744     1.000x
       40000    parallel        1       0.128     2.509     0.914x
       40000    parallel        2       0.063     5.114     1.864x
       40000    parallel        4       0.050     6.429     2.343x

       50000  sequential        -       0.178     2.242     1.000x
       50000    parallel        1       0.176     2.274     1.015x
       50000    parallel        2       0.087     4.624     2.063x
       50000    parallel        4       0.056     7.203     3.213x

       75000  sequential        -       0.211     2.850     1.000x
       75000    parallel        1       0.232     2.590     0.909x
       75000    parallel        2       0.130     4.607     1.617x
       75000    parallel        4       0.082     7.325     2.571x

      100000  sequential        -       0.295     2.716     1.000x
      100000    parallel        1       0.301     2.656     0.978x
      100000    parallel        2       0.145     5.508     2.028x
      100000    parallel        4       0.109     7.365     2.712x

      200000  sequential        -       0.512     3.124     1.000x
      200000    parallel        1       0.606     2.640     0.845x
      200000    parallel        2       0.280     5.712     1.828x
      200000    parallel        4       0.207     7.719     2.470x

      500000  sequential        -       1.401     2.855     1.000x
      500000    parallel        1       1.494     2.677     0.938x
      500000    parallel        2       0.874     4.577     1.603x
      500000    parallel        4       0.558     7.170     2.511x

     1000000  sequential        -       2.700     2.963     1.000x
     1000000    parallel        1       2.902     2.757     0.930x
     1000000    parallel        2       1.385     5.775     1.949x
     1000000    parallel        4       0.839     9.536     3.218x

     2000000  sequential        -       5.103     3.135     1.000x
     2000000    parallel        1       5.601     2.857     0.911x
     2000000    parallel        2       2.786     5.744     1.832x
     2000000    parallel        4       1.579    10.131     3.231x

     5000000  sequential        -      12.857     3.111     1.000x
     5000000    parallel        1      13.939     2.870     0.922x
     5000000    parallel        2       6.732     5.942     1.910x
     5000000    parallel        4       4.039     9.904     3.183x

    10000000  sequential        -      26.086     3.067     1.000x
    10000000    parallel        1      28.383     2.819     0.919x
    10000000    parallel        2      13.522     5.916     1.929x
    10000000    parallel        4       7.997    10.004     3.262x

    33554432  sequential        -      90.810     2.956     1.000x
    33554432    parallel        1      93.945     2.857     0.967x
    33554432    parallel        2      47.298     5.675     1.920x
    33554432    parallel        4      31.802     8.441     2.855x


== sigmoid ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.003     2.822     1.000x
        1000    parallel        1       0.003     2.621     0.929x
        1000    parallel        2       0.003     2.542     0.901x
        1000    parallel        4       0.003     2.294     0.813x

       10000  sequential        -       0.033     2.424     1.000x
       10000    parallel        1       0.035     2.255     0.931x
       10000    parallel        2       0.031     2.620     1.081x
       10000    parallel        4       0.030     2.688     1.109x

       32768  sequential        -       0.095     2.756     1.000x
       32768    parallel        1       0.100     2.618     0.950x
       32768    parallel        2       0.100     2.629     0.954x
       32768    parallel        4       0.102     2.571     0.933x

       32769  sequential        -       0.094     2.775     1.000x
       32769    parallel        1       0.100     2.625     0.946x
       32769    parallel        2       0.059     4.409     1.589x
       32769    parallel        4       0.033     7.848     2.828x

       40000  sequential        -       0.114     2.797     1.000x
       40000    parallel        1       0.125     2.567     0.918x
       40000    parallel        2       0.068     4.698     1.680x
       40000    parallel        4       0.055     5.777     2.065x

       50000  sequential        -       0.155     2.582     1.000x
       50000    parallel        1       0.170     2.359     0.914x
       50000    parallel        2       0.098     4.085     1.582x
       50000    parallel        4       0.068     5.878     2.277x

       75000  sequential        -       0.222     2.706     1.000x
       75000    parallel        1       0.243     2.466     0.911x
       75000    parallel        2       0.137     4.380     1.619x
       75000    parallel        4       0.066     9.127     3.373x

      100000  sequential        -       0.274     2.919     1.000x
      100000    parallel        1       0.289     2.772     0.950x
      100000    parallel        2       0.152     5.266     1.804x
      100000    parallel        4       0.084     9.573     3.280x

      200000  sequential        -       0.549     2.914     1.000x
      200000    parallel        1       0.582     2.747     0.943x
      200000    parallel        2       0.319     5.012     1.720x
      200000    parallel        4       0.175     9.118     3.129x

      500000  sequential        -       1.416     2.824     1.000x
      500000    parallel        1       1.493     2.680     0.949x
      500000    parallel        2       0.735     5.444     1.928x
      500000    parallel        4       0.631     6.340     2.245x

     1000000  sequential        -       2.787     2.871     1.000x
     1000000    parallel        1       2.872     2.786     0.971x
     1000000    parallel        2       1.477     5.418     1.887x
     1000000    parallel        4       0.858     9.321     3.247x

     2000000  sequential        -       5.602     2.856     1.000x
     2000000    parallel        1       5.806     2.756     0.965x
     2000000    parallel        2       2.999     5.334     1.868x
     2000000    parallel        4       1.799     8.895     3.114x

     5000000  sequential        -      14.096     2.838     1.000x
     5000000    parallel        1      14.498     2.759     0.972x
     5000000    parallel        2       7.678     5.210     1.836x
     5000000    parallel        4       4.416     9.057     3.192x

    10000000  sequential        -      30.067     2.661     1.000x
    10000000    parallel        1      29.747     2.689     1.011x
    10000000    parallel        2      15.428     5.186     1.949x
    10000000    parallel        4       8.972     8.917     3.351x

    33554432  sequential        -      95.447     2.812     1.000x
    33554432    parallel        1      98.270     2.732     0.971x
    33554432    parallel        2      53.850     4.985     1.772x
    33554432    parallel        4      31.080     8.637     3.071x


== dot ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001    11.858     1.000x
        1000    parallel        1       0.001    10.873     0.917x
        1000    parallel        2       0.001    11.811     0.996x
        1000    parallel        4       0.001    11.527     0.972x

       10000  sequential        -       0.007    11.699     1.000x
       10000    parallel        1       0.007    11.496     0.983x
       10000    parallel        2       0.007    11.507     0.984x
       10000    parallel        4       0.007    11.658     0.997x

       32768  sequential        -       0.022    11.717     1.000x
       32768    parallel        1       0.022    11.722     1.000x
       32768    parallel        2       0.023    11.629     0.993x
       32768    parallel        4       0.023    11.588     0.989x

       32769  sequential        -       0.022    11.739     1.000x
       32769    parallel        1       0.022    11.655     0.993x
       32769    parallel        2       0.013    20.819     1.774x
       32769    parallel        4       0.007    38.667     3.294x

       40000  sequential        -       0.027    11.780     1.000x
       40000    parallel        1       0.027    11.655     0.989x
       40000    parallel        2       0.016    19.685     1.671x
       40000    parallel        4       0.008    37.655     3.196x

       50000  sequential        -       0.034    11.783     1.000x
       50000    parallel        1       0.034    11.680     0.991x
       50000    parallel        2       0.018    22.098     1.875x
       50000    parallel        4       0.011    37.290     3.165x

       75000  sequential        -       0.051    11.735     1.000x
       75000    parallel        1       0.052    11.569     0.986x
       75000    parallel        2       0.028    21.499     1.832x
       75000    parallel        4       0.017    35.155     2.996x

      100000  sequential        -       0.068    11.690     1.000x
      100000    parallel        1       0.070    11.393     0.975x
      100000    parallel        2       0.037    21.854     1.869x
      100000    parallel        4       0.023    34.633     2.963x

      200000  sequential        -       0.138    11.570     1.000x
      200000    parallel        1       0.144    11.125     0.961x
      200000    parallel        2       0.105    15.268     1.320x
      200000    parallel        4       0.058    27.373     2.366x

      500000  sequential        -       0.356    11.224     1.000x
      500000    parallel        1       0.359    11.133     0.992x
      500000    parallel        2       0.201    19.868     1.770x
      500000    parallel        4       0.142    28.263     2.518x

     1000000  sequential        -       0.743    10.763     1.000x
     1000000    parallel        1       0.747    10.709     0.995x
     1000000    parallel        2       0.443    18.046     1.677x
     1000000    parallel        4       0.286    28.009     2.602x

     2000000  sequential        -       1.448    11.050     1.000x
     2000000    parallel        1       2.249     7.114     0.644x
     2000000    parallel        2       1.183    13.529     1.224x
     2000000    parallel        4       0.731    21.899     1.982x

     5000000  sequential        -       4.324     9.252     1.000x
     5000000    parallel        1       3.988    10.030     1.084x
     5000000    parallel        2       2.149    18.615     2.012x
     5000000    parallel        4       1.501    26.650     2.881x

    10000000  sequential        -       7.204    11.105     1.000x
    10000000    parallel        1       7.431    10.766     0.969x
    10000000    parallel        2       3.948    20.263     1.825x
    10000000    parallel        4       3.271    24.456     2.202x

    33554432  sequential        -      24.847    10.804     1.000x
    33554432    parallel        1      25.190    10.656     0.986x
    33554432    parallel        2      13.252    20.256     1.875x
    33554432    parallel        4       9.654    27.806     2.574x
```

</details>

<details>
<summary>Full output: cpu_parallel_benchmark, Machine 3, run 2</summary>

```text
Sequential vs parallel on the CPU (float32)
compiler: gcc 13.3.0, OpenMP: yes, max threads: 4, hardware threads: 12, parallel cutoff: 32768 elements
sequential = plain loop, no OpenMP. parallel = parallel kernel with N threads.
Speedup = sequential time / this time. Median of 9 samples.

== add ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   188.429     1.000x
        1000    parallel        1       0.000   186.629     0.990x
        1000    parallel        2       0.000   185.745     0.986x
        1000    parallel        4       0.000   187.767     0.996x

       10000  sequential        -       0.001   135.109     1.000x
       10000    parallel        1       0.001   120.068     0.889x
       10000    parallel        2       0.001   120.023     0.888x
       10000    parallel        4       0.001   120.063     0.889x

       32768  sequential        -       0.003   133.923     1.000x
       32768    parallel        1       0.003   127.580     0.953x
       32768    parallel        2       0.003   127.868     0.955x
       32768    parallel        4       0.003   114.114     0.852x

       32769  sequential        -       0.003   128.043     1.000x
       32769    parallel        1       0.003   129.749     1.013x
       32769    parallel        2       0.002   158.542     1.238x
       32769    parallel        4       0.002   231.316     1.807x

       40000  sequential        -       0.004   130.282     1.000x
       40000    parallel        1       0.004   113.500     0.871x
       40000    parallel        2       0.003   159.977     1.228x
       40000    parallel        4       0.002   246.627     1.893x

       50000  sequential        -       0.005   125.096     1.000x
       50000    parallel        1       0.005   113.773     0.909x
       50000    parallel        2       0.004   167.898     1.342x
       50000    parallel        4       0.002   279.335     2.233x

       75000  sequential        -       0.007   123.253     1.000x
       75000    parallel        1       0.007   121.233     0.984x
       75000    parallel        2       0.005   176.422     1.431x
       75000    parallel        4       0.003   303.803     2.465x

      100000  sequential        -       0.010   122.546     1.000x
      100000    parallel        1       0.010   121.243     0.989x
      100000    parallel        2       0.007   178.354     1.455x
      100000    parallel        4       0.004   297.737     2.430x

      200000  sequential        -       0.020   123.026     1.000x
      200000    parallel        1       0.020   121.061     0.984x
      200000    parallel        2       0.013   181.358     1.474x
      200000    parallel        4       0.008   304.459     2.475x

      500000  sequential        -       0.051   116.812     1.000x
      500000    parallel        1       0.052   116.037     0.993x
      500000    parallel        2       0.033   183.915     1.574x
      500000    parallel        4       0.020   300.823     2.575x

     1000000  sequential        -       0.165    72.548     1.000x
     1000000    parallel        1       0.186    64.641     0.891x
     1000000    parallel        2       0.114   105.677     1.457x
     1000000    parallel        4       0.061   197.056     2.716x

     2000000  sequential        -       0.825    29.081     1.000x
     2000000    parallel        1       0.847    28.343     0.975x
     2000000    parallel        2       0.691    34.742     1.195x
     2000000    parallel        4       0.699    34.350     1.181x

     5000000  sequential        -       3.240    18.516     1.000x
     5000000    parallel        1       2.928    20.492     1.107x
     5000000    parallel        2       2.823    21.251     1.148x
     5000000    parallel        4       2.955    20.305     1.097x

    10000000  sequential        -       6.100    19.671     1.000x
    10000000    parallel        1       6.387    18.787     0.955x
    10000000    parallel        2       5.991    20.030     1.018x
    10000000    parallel        4       6.251    19.198     0.976x

    33554432  sequential        -      23.978    16.792     1.000x
    33554432    parallel        1      23.979    16.792     1.000x
    33554432    parallel        2      21.151    19.037     1.134x
    33554432    parallel        4      20.407    19.731     1.175x


== sub ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   185.579     1.000x
        1000    parallel        1       0.000   178.415     0.961x
        1000    parallel        2       0.000   176.502     0.951x
        1000    parallel        4       0.000   186.099     1.003x

       10000  sequential        -       0.001   130.942     1.000x
       10000    parallel        1       0.001   112.756     0.861x
       10000    parallel        2       0.001   118.219     0.903x
       10000    parallel        4       0.001   113.285     0.865x

       32768  sequential        -       0.003   122.997     1.000x
       32768    parallel        1       0.003   125.173     1.018x
       32768    parallel        2       0.003   127.046     1.033x
       32768    parallel        4       0.003   125.328     1.019x

       32769  sequential        -       0.003   124.081     1.000x
       32769    parallel        1       0.003   123.571     0.996x
       32769    parallel        2       0.003   152.900     1.232x
       32769    parallel        4       0.002   231.637     1.867x

       40000  sequential        -       0.004   116.021     1.000x
       40000    parallel        1       0.004   114.550     0.987x
       40000    parallel        2       0.003   159.510     1.375x
       40000    parallel        4       0.002   234.731     2.023x

       50000  sequential        -       0.005   110.962     1.000x
       50000    parallel        1       0.005   109.739     0.989x
       50000    parallel        2       0.004   158.979     1.433x
       50000    parallel        4       0.002   253.095     2.281x

       75000  sequential        -       0.008   108.344     1.000x
       75000    parallel        1       0.008   106.810     0.986x
       75000    parallel        2       0.006   145.527     1.343x
       75000    parallel        4       0.003   276.056     2.548x

      100000  sequential        -       0.011   109.814     1.000x
      100000    parallel        1       0.011   107.761     0.981x
      100000    parallel        2       0.008   152.249     1.386x
      100000    parallel        4       0.004   301.876     2.749x

      200000  sequential        -       0.022   108.578     1.000x
      200000    parallel        1       0.022   107.934     0.994x
      200000    parallel        2       0.016   149.825     1.380x
      200000    parallel        4       0.010   234.302     2.158x

      500000  sequential        -       0.065    92.919     1.000x
      500000    parallel        1       0.113    53.314     0.574x
      500000    parallel        2       0.044   136.551     1.470x
      500000    parallel        4       0.026   231.307     2.489x

     1000000  sequential        -       0.401    29.923     1.000x
     1000000    parallel        1       0.215    55.707     1.862x
     1000000    parallel        2       0.136    87.934     2.939x
     1000000    parallel        4       0.078   154.034     5.148x

     2000000  sequential        -       1.386    17.322     1.000x
     2000000    parallel        1       1.235    19.437     1.122x
     2000000    parallel        2       0.982    24.441     1.411x
     2000000    parallel        4       1.017    23.593     1.362x

     5000000  sequential        -       3.118    19.244     1.000x
     5000000    parallel        1       2.938    20.424     1.061x
     5000000    parallel        2       2.890    20.759     1.079x
     5000000    parallel        4       2.945    20.373     1.059x

    10000000  sequential        -       6.756    17.761     1.000x
    10000000    parallel        1       6.284    19.095     1.075x
    10000000    parallel        2       5.959    20.137     1.134x
    10000000    parallel        4       6.097    19.682     1.108x

    33554432  sequential        -      23.128    17.410     1.000x
    33554432    parallel        1      24.141    16.679     0.958x
    33554432    parallel        2      20.942    19.227     1.104x
    33554432    parallel        4      22.049    18.262     1.049x


== mul ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   186.559     1.000x
        1000    parallel        1       0.000   183.626     0.984x
        1000    parallel        2       0.000   176.869     0.948x
        1000    parallel        4       0.000   184.154     0.987x

       10000  sequential        -       0.001   129.508     1.000x
       10000    parallel        1       0.001   115.199     0.890x
       10000    parallel        2       0.001   117.345     0.906x
       10000    parallel        4       0.001   117.674     0.909x

       32768  sequential        -       0.003   126.282     1.000x
       32768    parallel        1       0.003   120.117     0.951x
       32768    parallel        2       0.005    76.516     0.606x
       32768    parallel        4       0.005    76.413     0.605x

       32769  sequential        -       0.004    98.822     1.000x
       32769    parallel        1       0.004    91.836     0.929x
       32769    parallel        2       0.003   132.438     1.340x
       32769    parallel        4       0.002   189.202     1.915x

       40000  sequential        -       0.004   114.763     1.000x
       40000    parallel        1       0.004   112.649     0.982x
       40000    parallel        2       0.003   148.772     1.296x
       40000    parallel        4       0.002   252.424     2.200x

       50000  sequential        -       0.006   109.084     1.000x
       50000    parallel        1       0.005   109.537     1.004x
       50000    parallel        2       0.004   152.211     1.395x
       50000    parallel        4       0.002   248.342     2.277x

       75000  sequential        -       0.008   107.635     1.000x
       75000    parallel        1       0.008   107.403     0.998x
       75000    parallel        2       0.006   149.804     1.392x
       75000    parallel        4       0.003   268.013     2.490x

      100000  sequential        -       0.011   109.393     1.000x
      100000    parallel        1       0.011   107.134     0.979x
      100000    parallel        2       0.008   148.664     1.359x
      100000    parallel        4       0.004   303.738     2.777x

      200000  sequential        -       0.022   109.653     1.000x
      200000    parallel        1       0.022   107.798     0.983x
      200000    parallel        2       0.033    73.135     0.667x
      200000    parallel        4       0.018   132.312     1.207x

      500000  sequential        -       0.144    41.727     1.000x
      500000    parallel        1       0.067    89.398     2.142x
      500000    parallel        2       0.039   153.348     3.675x
      500000    parallel        4       0.025   241.916     5.798x

     1000000  sequential        -       0.228    52.532     1.000x
     1000000    parallel        1       0.248    48.426     0.922x
     1000000    parallel        2       0.162    74.242     1.413x
     1000000    parallel        4       0.077   155.505     2.960x

     2000000  sequential        -       1.272    18.874     1.000x
     2000000    parallel        1       1.176    20.409     1.081x
     2000000    parallel        2       0.930    25.814     1.368x
     2000000    parallel        4       0.834    28.789     1.525x

     5000000  sequential        -       3.684    16.288     1.000x
     5000000    parallel        1       3.279    18.298     1.123x
     5000000    parallel        2       3.046    19.698     1.209x
     5000000    parallel        4       3.144    19.086     1.172x

    10000000  sequential        -       6.745    17.791     1.000x
    10000000    parallel        1       6.963    17.233     0.969x
    10000000    parallel        2       6.442    18.627     1.047x
    10000000    parallel        4       6.078    19.742     1.110x

    33554432  sequential        -      23.347    17.247     1.000x
    33554432    parallel        1      22.905    17.579     1.019x
    33554432    parallel        2      21.563    18.673     1.083x
    33554432    parallel        4      23.996    16.780     0.973x


== div ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    62.151     1.000x
        1000    parallel        1       0.000    61.857     0.995x
        1000    parallel        2       0.000    61.425     0.988x
        1000    parallel        4       0.000    61.341     0.987x

       10000  sequential        -       0.002    62.023     1.000x
       10000    parallel        1       0.002    62.177     1.002x
       10000    parallel        2       0.002    61.276     0.988x
       10000    parallel        4       0.002    60.850     0.981x

       32768  sequential        -       0.007    58.758     1.000x
       32768    parallel        1       0.006    61.983     1.055x
       32768    parallel        2       0.006    62.649     1.066x
       32768    parallel        4       0.006    62.381     1.062x

       32769  sequential        -       0.006    61.541     1.000x
       32769    parallel        1       0.006    64.211     1.043x
       32769    parallel        2       0.004   102.080     1.659x
       32769    parallel        4       0.002   167.994     2.730x

       40000  sequential        -       0.008    61.849     1.000x
       40000    parallel        1       0.008    63.697     1.030x
       40000    parallel        2       0.004   110.844     1.792x
       40000    parallel        4       0.003   179.114     2.896x

       50000  sequential        -       0.010    61.581     1.000x
       50000    parallel        1       0.009    63.673     1.034x
       50000    parallel        2       0.006   105.978     1.721x
       50000    parallel        4       0.004   166.464     2.703x

       75000  sequential        -       0.015    60.820     1.000x
       75000    parallel        1       0.015    61.594     1.013x
       75000    parallel        2       0.008   112.278     1.846x
       75000    parallel        4       0.005   182.586     3.002x

      100000  sequential        -       0.019    62.075     1.000x
      100000    parallel        1       0.018    65.172     1.050x
      100000    parallel        2       0.010   115.117     1.854x
      100000    parallel        4       0.006   194.089     3.127x

      200000  sequential        -       0.039    60.937     1.000x
      200000    parallel        1       0.039    60.873     0.999x
      200000    parallel        2       0.026    92.589     1.519x
      200000    parallel        4       0.022   108.952     1.788x

      500000  sequential        -       0.180    33.395     1.000x
      500000    parallel        1       0.130    46.262     1.385x
      500000    parallel        2       0.061    98.019     2.935x
      500000    parallel        4       0.035   169.531     5.077x

     1000000  sequential        -       0.346    34.641     1.000x
     1000000    parallel        1       0.306    39.196     1.131x
     1000000    parallel        2       0.147    81.696     2.358x
     1000000    parallel        4       0.091   132.253     3.818x

     2000000  sequential        -       1.138    21.089     1.000x
     2000000    parallel        1       0.968    24.792     1.176x
     2000000    parallel        2       0.803    29.888     1.417x
     2000000    parallel        4       0.766    31.320     1.485x

     5000000  sequential        -       3.153    19.027     1.000x
     5000000    parallel        1       3.149    19.052     1.001x
     5000000    parallel        2       3.108    19.305     1.015x
     5000000    parallel        4       3.141    19.101     1.004x

    10000000  sequential        -       6.836    17.554     1.000x
    10000000    parallel        1       6.641    18.070     1.029x
    10000000    parallel        2       5.978    20.074     1.144x
    10000000    parallel        4       6.402    18.744     1.068x

    33554432  sequential        -      23.670    17.011     1.000x
    33554432    parallel        1      22.781    17.675     1.039x
    33554432    parallel        2      21.211    18.983     1.116x
    33554432    parallel        4      26.254    15.337     0.902x


== neg ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   109.567     1.000x
        1000    parallel        1       0.000   122.139     1.115x
        1000    parallel        2       0.000   111.934     1.022x
        1000    parallel        4       0.000   122.490     1.118x

       10000  sequential        -       0.001   125.607     1.000x
       10000    parallel        1       0.001   124.856     0.994x
       10000    parallel        2       0.001   126.757     1.009x
       10000    parallel        4       0.001   127.076     1.012x

       32768  sequential        -       0.002   132.650     1.000x
       32768    parallel        1       0.002   135.205     1.019x
       32768    parallel        2       0.002   124.538     0.939x
       32768    parallel        4       0.002   129.833     0.979x

       32769  sequential        -       0.005    57.219     1.000x
       32769    parallel        1       0.002   107.994     1.887x
       32769    parallel        2       0.003    99.843     1.745x
       32769    parallel        4       0.002   141.940     2.481x

       40000  sequential        -       0.003   126.418     1.000x
       40000    parallel        1       0.003   123.848     0.980x
       40000    parallel        2       0.003    92.920     0.735x
       40000    parallel        4       0.003   127.623     1.010x

       50000  sequential        -       0.003   119.147     1.000x
       50000    parallel        1       0.003   123.956     1.040x
       50000    parallel        2       0.004   107.577     0.903x
       50000    parallel        4       0.002   173.435     1.456x

       75000  sequential        -       0.005   115.080     1.000x
       75000    parallel        1       0.006    99.191     0.862x
       75000    parallel        2       0.005   117.822     1.024x
       75000    parallel        4       0.003   183.329     1.593x

      100000  sequential        -       0.007   110.213     1.000x
      100000    parallel        1       0.007   107.640     0.977x
      100000    parallel        2       0.008    95.066     0.863x
      100000    parallel        4       0.006   142.793     1.296x

      200000  sequential        -       0.021    76.608     1.000x
      200000    parallel        1       0.018    88.684     1.158x
      200000    parallel        2       0.016    99.457     1.298x
      200000    parallel        4       0.010   162.093     2.116x

      500000  sequential        -       0.036   109.607     1.000x
      500000    parallel        1       0.036   111.375     1.016x
      500000    parallel        2       0.033   122.403     1.117x
      500000    parallel        4       0.020   195.557     1.784x

     1000000  sequential        -       0.081    98.374     1.000x
     1000000    parallel        1       0.094    85.310     0.867x
     1000000    parallel        2       0.077   104.320     1.060x
     1000000    parallel        4       0.042   191.425     1.946x

     2000000  sequential        -       0.561    28.529     1.000x
     2000000    parallel        1       0.524    30.521     1.070x
     2000000    parallel        2       0.324    49.368     1.730x
     2000000    parallel        4       0.239    67.052     2.350x

     5000000  sequential        -       2.331    17.163     1.000x
     5000000    parallel        1       2.339    17.102     0.996x
     5000000    parallel        2       1.897    21.085     1.228x
     5000000    parallel        4       1.981    20.188     1.176x

    10000000  sequential        -       5.097    15.695     1.000x
    10000000    parallel        1       4.938    16.202     1.032x
    10000000    parallel        2       4.463    17.926     1.142x
    10000000    parallel        4       4.540    17.619     1.123x

    33554432  sequential        -      17.906    14.992     1.000x
    33554432    parallel        1      18.460    14.541     0.970x
    33554432    parallel        2      15.866    16.919     1.129x
    33554432    parallel        4      16.381    16.387     1.093x


== abs ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000    68.618     1.000x
        1000    parallel        1       0.000   126.559     1.844x
        1000    parallel        2       0.000   126.780     1.848x
        1000    parallel        4       0.000   126.803     1.848x

       10000  sequential        -       0.001    70.240     1.000x
       10000    parallel        1       0.001   130.340     1.856x
       10000    parallel        2       0.001   128.776     1.833x
       10000    parallel        4       0.001   128.327     1.827x

       32768  sequential        -       0.004    70.295     1.000x
       32768    parallel        1       0.002   135.727     1.931x
       32768    parallel        2       0.002   135.365     1.926x
       32768    parallel        4       0.002   135.325     1.925x

       32769  sequential        -       0.004    70.317     1.000x
       32769    parallel        1       0.002   134.198     1.908x
       32769    parallel        2       0.002   167.998     2.389x
       32769    parallel        4       0.001   225.287     3.204x

       40000  sequential        -       0.005    70.372     1.000x
       40000    parallel        1       0.002   128.521     1.826x
       40000    parallel        2       0.002   182.744     2.597x
       40000    parallel        4       0.001   253.499     3.602x

       50000  sequential        -       0.006    69.598     1.000x
       50000    parallel        1       0.003   130.539     1.876x
       50000    parallel        2       0.002   190.170     2.732x
       50000    parallel        4       0.001   272.023     3.908x

       75000  sequential        -       0.009    69.506     1.000x
       75000    parallel        1       0.005   116.897     1.682x
       75000    parallel        2       0.003   211.214     3.039x
       75000    parallel        4       0.002   319.503     4.597x

      100000  sequential        -       0.012    68.891     1.000x
      100000    parallel        1       0.007   118.049     1.714x
      100000    parallel        2       0.004   193.782     2.813x
      100000    parallel        4       0.003   310.887     4.513x

      200000  sequential        -       0.024    67.384     1.000x
      200000    parallel        1       0.014   115.469     1.714x
      200000    parallel        2       0.009   175.502     2.605x
      200000    parallel        4       0.005   345.984     5.135x

      500000  sequential        -       0.061    65.964     1.000x
      500000    parallel        1       0.052    76.601     1.161x
      500000    parallel        2       0.026   153.223     2.323x
      500000    parallel        4       0.014   285.209     4.324x

     1000000  sequential        -       0.265    30.178     1.000x
     1000000    parallel        1       0.272    29.430     0.975x
     1000000    parallel        2       0.114    70.148     2.324x
     1000000    parallel        4       0.053   151.000     5.004x

     2000000  sequential        -       1.107    14.449     1.000x
     2000000    parallel        1       1.078    14.838     1.027x
     2000000    parallel        2       0.752    21.279     1.473x
     2000000    parallel        4       0.607    26.341     1.823x

     5000000  sequential        -       2.981    13.419     1.000x
     5000000    parallel        1       2.583    15.488     1.154x
     5000000    parallel        2       2.315    17.279     1.288x
     5000000    parallel        4       2.208    18.117     1.350x

    10000000  sequential        -       7.460    10.724     1.000x
    10000000    parallel        1       8.481     9.433     0.880x
    10000000    parallel        2       5.000    15.999     1.492x
    10000000    parallel        4       5.130    15.595     1.454x

    33554432  sequential        -      18.304    14.665     1.000x
    33554432    parallel        1      18.634    14.406     0.982x
    33554432    parallel        2      17.770    15.106     1.030x
    33554432    parallel        4      17.910    14.988     1.022x


== square ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   114.830     1.000x
        1000    parallel        1       0.000   120.980     1.054x
        1000    parallel        2       0.000   123.000     1.071x
        1000    parallel        4       0.000   119.869     1.044x

       10000  sequential        -       0.001   109.887     1.000x
       10000    parallel        1       0.001   112.922     1.028x
       10000    parallel        2       0.001   107.800     0.981x
       10000    parallel        4       0.001   118.302     1.077x

       32768  sequential        -       0.002   125.977     1.000x
       32768    parallel        1       0.002   123.175     0.978x
       32768    parallel        2       0.002   121.381     0.964x
       32768    parallel        4       0.002   123.294     0.979x

       32769  sequential        -       0.002   133.700     1.000x
       32769    parallel        1       0.002   124.092     0.928x
       32769    parallel        2       0.002   163.106     1.220x
       32769    parallel        4       0.001   228.849     1.712x

       40000  sequential        -       0.002   129.154     1.000x
       40000    parallel        1       0.003   127.111     0.984x
       40000    parallel        2       0.002   182.230     1.411x
       40000    parallel        4       0.001   253.090     1.960x

       50000  sequential        -       0.003   128.105     1.000x
       50000    parallel        1       0.003   124.765     0.974x
       50000    parallel        2       0.002   193.775     1.513x
       50000    parallel        4       0.001   276.132     2.156x

       75000  sequential        -       0.005   120.741     1.000x
       75000    parallel        1       0.005   116.718     0.967x
       75000    parallel        2       0.003   204.003     1.690x
       75000    parallel        4       0.002   320.450     2.654x

      100000  sequential        -       0.007   119.385     1.000x
      100000    parallel        1       0.007   117.540     0.985x
      100000    parallel        2       0.004   205.402     1.720x
      100000    parallel        4       0.002   339.562     2.844x

      200000  sequential        -       0.014   116.208     1.000x
      200000    parallel        1       0.014   118.127     1.017x
      200000    parallel        2       0.008   202.640     1.744x
      200000    parallel        4       0.004   359.520     3.094x

      500000  sequential        -       0.034   118.224     1.000x
      500000    parallel        1       0.037   107.408     0.909x
      500000    parallel        2       0.022   178.621     1.511x
      500000    parallel        4       0.014   278.072     2.352x

     1000000  sequential        -       0.129    61.816     1.000x
     1000000    parallel        1       0.088    90.537     1.465x
     1000000    parallel        2       0.045   176.794     2.860x
     1000000    parallel        4       0.028   284.899     4.609x

     2000000  sequential        -       0.512    31.270     1.000x
     2000000    parallel        1       0.505    31.656     1.012x
     2000000    parallel        2       0.351    45.564     1.457x
     2000000    parallel        4       0.288    55.499     1.775x

     5000000  sequential        -       2.293    17.447     1.000x
     5000000    parallel        1       2.495    16.033     0.919x
     5000000    parallel        2       2.322    17.230     0.988x
     5000000    parallel        4       2.032    19.687     1.128x

    10000000  sequential        -       5.214    15.344     1.000x
    10000000    parallel        1       5.199    15.388     1.003x
    10000000    parallel        2       4.689    17.060     1.112x
    10000000    parallel        4       4.834    16.550     1.079x

    33554432  sequential        -      17.810    15.072     1.000x
    33554432    parallel        1      17.701    15.165     1.006x
    33554432    parallel        2      15.687    17.112     1.135x
    33554432    parallel        4      17.108    15.690     1.041x


== relu ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.000   112.164     1.000x
        1000    parallel        1       0.000    67.581     0.603x
        1000    parallel        2       0.000    66.778     0.595x
        1000    parallel        4       0.000    66.882     0.596x

       10000  sequential        -       0.001   101.168     1.000x
       10000    parallel        1       0.001    69.206     0.684x
       10000    parallel        2       0.001    69.672     0.689x
       10000    parallel        4       0.001    69.837     0.690x

       32768  sequential        -       0.002   106.789     1.000x
       32768    parallel        1       0.004    70.930     0.664x
       32768    parallel        2       0.004    71.347     0.668x
       32768    parallel        4       0.004    71.508     0.670x

       32769  sequential        -       0.002   106.353     1.000x
       32769    parallel        1       0.004    70.644     0.664x
       32769    parallel        2       0.003   103.506     0.973x
       32769    parallel        4       0.002   158.166     1.487x

       40000  sequential        -       0.003   104.067     1.000x
       40000    parallel        1       0.005    70.974     0.682x
       40000    parallel        2       0.003   109.618     1.053x
       40000    parallel        4       0.002   166.767     1.602x

       50000  sequential        -       0.004   103.931     1.000x
       50000    parallel        1       0.006    71.043     0.684x
       50000    parallel        2       0.004   112.796     1.085x
       50000    parallel        4       0.002   183.385     1.764x

       75000  sequential        -       0.006    98.123     1.000x
       75000    parallel        1       0.009    67.345     0.686x
       75000    parallel        2       0.005   119.459     1.217x
       75000    parallel        4       0.003   203.095     2.070x

      100000  sequential        -       0.008   100.620     1.000x
      100000    parallel        1       0.012    67.654     0.672x
      100000    parallel        2       0.007   122.568     1.218x
      100000    parallel        4       0.004   207.902     2.066x

      200000  sequential        -       0.016   100.710     1.000x
      200000    parallel        1       0.023    70.400     0.699x
      200000    parallel        2       0.013   124.335     1.235x
      200000    parallel        4       0.007   216.979     2.154x

      500000  sequential        -       0.040   101.235     1.000x
      500000    parallel        1       0.058    68.645     0.678x
      500000    parallel        2       0.035   112.872     1.115x
      500000    parallel        4       0.021   194.718     1.923x

     1000000  sequential        -       0.093    85.646     1.000x
     1000000    parallel        1       0.132    60.506     0.706x
     1000000    parallel        2       0.068   117.531     1.372x
     1000000    parallel        4       0.040   197.973     2.312x

     2000000  sequential        -       0.547    29.276     1.000x
     2000000    parallel        1       0.519    30.843     1.054x
     2000000    parallel        2       0.364    43.923     1.500x
     2000000    parallel        4       0.272    58.770     2.007x

     5000000  sequential        -       2.549    15.691     1.000x
     5000000    parallel        1       2.481    16.125     1.028x
     5000000    parallel        2       2.164    18.488     1.178x
     5000000    parallel        4       2.067    19.353     1.233x

    10000000  sequential        -       5.631    14.207     1.000x
    10000000    parallel        1       5.786    13.827     0.973x
    10000000    parallel        2       5.942    13.463     0.948x
    10000000    parallel        4       5.048    15.846     1.115x

    33554432  sequential        -      21.014    12.774     1.000x
    33554432    parallel        1      21.537    12.464     0.976x
    33554432    parallel        2      19.732    13.604     1.065x
    33554432    parallel        4      16.154    16.618     1.301x


== sqrt ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001     7.223     1.000x
        1000    parallel        1       0.001     7.232     1.001x
        1000    parallel        2       0.001     7.239     1.002x
        1000    parallel        4       0.001     7.233     1.001x

       10000  sequential        -       0.011     7.229     1.000x
       10000    parallel        1       0.011     7.234     1.001x
       10000    parallel        2       0.011     7.232     1.000x
       10000    parallel        4       0.011     7.232     1.000x

       32768  sequential        -       0.036     7.225     1.000x
       32768    parallel        1       0.036     7.233     1.001x
       32768    parallel        2       0.036     7.229     1.001x
       32768    parallel        4       0.036     7.224     1.000x

       32769  sequential        -       0.036     7.228     1.000x
       32769    parallel        1       0.036     7.224     0.999x
       32769    parallel        2       0.019    14.001     1.937x
       32769    parallel        4       0.010    27.004     3.736x

       40000  sequential        -       0.044     7.221     1.000x
       40000    parallel        1       0.044     7.227     1.001x
       40000    parallel        2       0.023    14.093     1.951x
       40000    parallel        4       0.012    27.355     3.788x

       50000  sequential        -       0.055     7.226     1.000x
       50000    parallel        1       0.055     7.224     1.000x
       50000    parallel        2       0.028    14.145     1.958x
       50000    parallel        4       0.014    27.600     3.820x

       75000  sequential        -       0.083     7.223     1.000x
       75000    parallel        1       0.083     7.227     1.001x
       75000    parallel        2       0.042    14.226     1.970x
       75000    parallel        4       0.021    28.062     3.885x

      100000  sequential        -       0.111     7.220     1.000x
      100000    parallel        1       0.111     7.217     1.000x
      100000    parallel        2       0.056    14.256     1.975x
      100000    parallel        4       0.028    28.230     3.910x

      200000  sequential        -       0.222     7.223     1.000x
      200000    parallel        1       0.222     7.222     1.000x
      200000    parallel        2       0.112    14.326     1.983x
      200000    parallel        4       0.056    28.483     3.944x

      500000  sequential        -       0.555     7.210     1.000x
      500000    parallel        1       0.554     7.218     1.001x
      500000    parallel        2       0.278    14.409     1.998x
      500000    parallel        4       0.139    28.710     3.982x

     1000000  sequential        -       1.119     7.149     1.000x
     1000000    parallel        1       1.109     7.216     1.009x
     1000000    parallel        2       0.560    14.287     1.998x
     1000000    parallel        4       0.278    28.764     4.023x

     2000000  sequential        -       2.236     7.156     1.000x
     2000000    parallel        1       2.227     7.184     1.004x
     2000000    parallel        2       1.155    13.848     1.935x
     2000000    parallel        4       0.599    26.714     3.733x

     5000000  sequential        -       5.680     7.042     1.000x
     5000000    parallel        1       5.679     7.044     1.000x
     5000000    parallel        2       3.022    13.238     1.880x
     5000000    parallel        4       2.044    19.572     2.779x

    10000000  sequential        -      11.524     6.942     1.000x
    10000000    parallel        1      11.732     6.819     0.982x
    10000000    parallel        2       6.256    12.787     1.842x
    10000000    parallel        4       4.736    16.892     2.433x

    33554432  sequential        -      40.082     6.697     1.000x
    33554432    parallel        1      40.667     6.601     0.986x
    33554432    parallel        2      23.290    11.526     1.721x
    33554432    parallel        4      17.261    15.551     2.322x


== exp ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.003     3.069     1.000x
        1000    parallel        1       0.003     2.896     0.944x
        1000    parallel        2       0.003     2.832     0.923x
        1000    parallel        4       0.003     2.850     0.929x

       10000  sequential        -       0.025     3.197     1.000x
       10000    parallel        1       0.028     2.895     0.905x
       10000    parallel        2       0.027     2.928     0.916x
       10000    parallel        4       0.027     2.925     0.915x

       32768  sequential        -       0.082     3.189     1.000x
       32768    parallel        1       0.089     2.935     0.920x
       32768    parallel        2       0.090     2.926     0.917x
       32768    parallel        4       0.089     2.929     0.918x

       32769  sequential        -       0.082     3.191     1.000x
       32769    parallel        1       0.089     2.943     0.922x
       32769    parallel        2       0.042     6.223     1.950x
       32769    parallel        4       0.023    11.556     3.622x

       40000  sequential        -       0.100     3.203     1.000x
       40000    parallel        1       0.109     2.935     0.916x
       40000    parallel        2       0.051     6.331     1.976x
       40000    parallel        4       0.028    11.446     3.573x

       50000  sequential        -       0.124     3.214     1.000x
       50000    parallel        1       0.136     2.947     0.917x
       50000    parallel        2       0.077     5.222     1.625x
       50000    parallel        4       0.056     7.093     2.207x

       75000  sequential        -       0.297     2.023     1.000x
       75000    parallel        1       0.231     2.602     1.286x
       75000    parallel        2       0.101     5.944     2.938x
       75000    parallel        4       0.066     9.115     4.505x

      100000  sequential        -       0.255     3.133     1.000x
      100000    parallel        1       0.278     2.874     0.917x
      100000    parallel        2       0.132     6.069     1.937x
      100000    parallel        4       0.077    10.349     3.303x

      200000  sequential        -       0.517     3.095     1.000x
      200000    parallel        1       0.557     2.872     0.928x
      200000    parallel        2       0.257     6.228     2.012x
      200000    parallel        4       0.145    11.070     3.577x

      500000  sequential        -       1.248     3.204     1.000x
      500000    parallel        1       1.363     2.935     0.916x
      500000    parallel        2       0.646     6.189     1.932x
      500000    parallel        4       0.363    11.018     3.439x

     1000000  sequential        -       2.514     3.182     1.000x
     1000000    parallel        1       2.731     2.929     0.921x
     1000000    parallel        2       1.346     5.945     1.868x
     1000000    parallel        4       0.806     9.922     3.119x

     2000000  sequential        -       5.080     3.150     1.000x
     2000000    parallel        1       5.486     2.916     0.926x
     2000000    parallel        2       2.615     6.118     1.942x
     2000000    parallel        4       1.492    10.722     3.404x

     5000000  sequential        -      12.620     3.170     1.000x
     5000000    parallel        1      13.730     2.913     0.919x
     5000000    parallel        2       6.426     6.225     1.964x
     5000000    parallel        4       3.544    11.288     3.561x

    10000000  sequential        -      25.884     3.091     1.000x
    10000000    parallel        1      27.931     2.864     0.927x
    10000000    parallel        2      13.276     6.026     1.950x
    10000000    parallel        4       8.636     9.264     2.997x

    33554432  sequential        -      84.488     3.177     1.000x
    33554432    parallel        1      93.867     2.860     0.900x
    33554432    parallel        2      43.678     6.146     1.934x
    33554432    parallel        4      24.918    10.773     3.391x


== sigmoid ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.003     2.937     1.000x
        1000    parallel        1       0.003     2.810     0.957x
        1000    parallel        2       0.003     2.798     0.952x
        1000    parallel        4       0.003     2.786     0.948x

       10000  sequential        -       0.028     2.900     1.000x
       10000    parallel        1       0.029     2.752     0.949x
       10000    parallel        2       0.031     2.567     0.885x
       10000    parallel        4       0.029     2.720     0.938x

       32768  sequential        -       0.092     2.855     1.000x
       32768    parallel        1       0.095     2.761     0.967x
       32768    parallel        2       0.093     2.804     0.982x
       32768    parallel        4       0.093     2.807     0.983x

       32769  sequential        -       0.090     2.906     1.000x
       32769    parallel        1       0.093     2.816     0.969x
       32769    parallel        2       0.047     5.549     1.910x
       32769    parallel        4       0.026    10.120     3.483x

       40000  sequential        -       0.110     2.905     1.000x
       40000    parallel        1       0.114     2.818     0.970x
       40000    parallel        2       0.058     5.562     1.915x
       40000    parallel        4       0.032    10.060     3.463x

       50000  sequential        -       0.136     2.933     1.000x
       50000    parallel        1       0.142     2.823     0.962x
       50000    parallel        2       0.072     5.546     1.891x
       50000    parallel        4       0.040    10.004     3.410x

       75000  sequential        -       0.206     2.912     1.000x
       75000    parallel        1       0.213     2.813     0.966x
       75000    parallel        2       0.112     5.351     1.838x
       75000    parallel        4       0.066     9.089     3.121x

      100000  sequential        -       0.280     2.861     1.000x
      100000    parallel        1       0.293     2.729     0.954x
      100000    parallel        2       0.147     5.434     1.899x
      100000    parallel        4       0.084     9.539     3.334x

      200000  sequential        -       0.548     2.919     1.000x
      200000    parallel        1       0.573     2.792     0.957x
      200000    parallel        2       0.289     5.536     1.896x
      200000    parallel        4       0.164     9.734     3.334x

      500000  sequential        -       1.351     2.961     1.000x
      500000    parallel        1       1.401     2.855     0.964x
      500000    parallel        2       0.736     5.432     1.834x
      500000    parallel        4       0.432     9.270     3.130x

     1000000  sequential        -       2.746     2.914     1.000x
     1000000    parallel        1       2.804     2.853     0.979x
     1000000    parallel        2       1.537     5.206     1.787x
     1000000    parallel        4       0.841     9.507     3.263x

     2000000  sequential        -       5.518     2.900     1.000x
     2000000    parallel        1       5.674     2.820     0.972x
     2000000    parallel        2       3.104     5.155     1.778x
     2000000    parallel        4       1.698     9.424     3.250x

     5000000  sequential        -      14.715     2.718     1.000x
     5000000    parallel        1      14.656     2.729     1.004x
     5000000    parallel        2       8.391     4.767     1.754x
     5000000    parallel        4       4.866     8.219     3.024x

    10000000  sequential        -      27.505     2.909     1.000x
    10000000    parallel        1      28.513     2.806     0.965x
    10000000    parallel        2      15.184     5.269     1.811x
    10000000    parallel        4       8.575     9.329     3.207x

    33554432  sequential        -      91.080     2.947     1.000x
    33554432    parallel        1     103.412     2.596     0.881x
    33554432    parallel        2      52.261     5.136     1.743x
    33554432    parallel        4      30.364     8.841     3.000x


== dot ==
           n        impl  threads          ms      GB/s    speedup
        1000  sequential        -       0.001    11.926     1.000x
        1000    parallel        1       0.001    11.465     0.961x
        1000    parallel        2       0.001    11.330     0.950x
        1000    parallel        4       0.001    11.689     0.980x

       10000  sequential        -       0.007    11.685     1.000x
       10000    parallel        1       0.007    11.474     0.982x
       10000    parallel        2       0.007    11.505     0.985x
       10000    parallel        4       0.007    11.138     0.953x

       32768  sequential        -       0.022    11.654     1.000x
       32768    parallel        1       0.023    11.343     0.973x
       32768    parallel        2       0.023    11.198     0.961x
       32768    parallel        4       0.024    10.718     0.920x

       32769  sequential        -       0.023    11.615     1.000x
       32769    parallel        1       0.023    11.400     0.982x
       32769    parallel        2       0.012    21.599     1.860x
       32769    parallel        4       0.007    35.954     3.096x

       40000  sequential        -       0.027    11.672     1.000x
       40000    parallel        1       0.033     9.841     0.843x
       40000    parallel        2       0.015    20.827     1.784x
       40000    parallel        4       0.009    35.372     3.030x

       50000  sequential        -       0.035    11.417     1.000x
       50000    parallel        1       0.036    11.232     0.984x
       50000    parallel        2       0.018    21.640     1.895x
       50000    parallel        4       0.010    38.890     3.406x

       75000  sequential        -       0.052    11.489     1.000x
       75000    parallel        1       0.055    10.908     0.949x
       75000    parallel        2       0.027    22.191     1.932x
       75000    parallel        4       0.016    38.183     3.323x

      100000  sequential        -       0.070    11.485     1.000x
      100000    parallel        1       0.071    11.272     0.982x
      100000    parallel        2       0.036    22.300     1.942x
      100000    parallel        4       0.020    40.372     3.515x

      200000  sequential        -       0.136    11.776     1.000x
      200000    parallel        1       0.140    11.468     0.974x
      200000    parallel        2       0.077    20.718     1.759x
      200000    parallel        4       0.043    37.190     3.158x

      500000  sequential        -       0.346    11.572     1.000x
      500000    parallel        1       0.348    11.493     0.993x
      500000    parallel        2       0.179    22.357     1.932x
      500000    parallel        4       0.111    36.094     3.119x

     1000000  sequential        -       0.692    11.568     1.000x
     1000000    parallel        1       0.703    11.373     0.983x
     1000000    parallel        2       0.418    19.150     1.655x
     1000000    parallel        4       0.245    32.624     2.820x

     2000000  sequential        -       1.416    11.302     1.000x
     2000000    parallel        1       1.418    11.281     0.998x
     2000000    parallel        2       0.745    21.474     1.900x
     2000000    parallel        4       0.413    38.733     3.427x

     5000000  sequential        -       3.551    11.264     1.000x
     5000000    parallel        1       4.122     9.704     0.861x
     5000000    parallel        2       2.065    19.366     1.719x
     5000000    parallel        4       1.403    28.507     2.531x

    10000000  sequential        -       7.089    11.285     1.000x
    10000000    parallel        1       7.260    11.019     0.976x
    10000000    parallel        2       3.968    20.162     1.787x
    10000000    parallel        4       3.502    22.842     2.024x

    33554432  sequential        -      25.807    10.402     1.000x
    33554432    parallel        1      26.678    10.062     0.967x
    33554432    parallel        2      14.320    18.746     1.802x
    33554432    parallel        4      11.097    24.189     2.325x
```

</details>

<details>
<summary>Previous detailed tables and operation notes</summary>

These tables retain the previously documented measurements and their original code versions.
The results table above uses one pair of complete runs for consistent comparison across all operations.

#### Earlier elementwise cutoff checks (2026-10-31)

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

##### Negation (`neg`): shared elementwise implementation

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

##### Absolute value, square, and ReLU: shared unary operations

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

##### Square root, exponential, and sigmoid: CPU float32

`sqrt`, `exp`, and `sigmoid` use the existing shared unary loops, API, dispatcher,
reference file, test groups, and CPU benchmark. Code: `84b3cbf` plus these local changes.
The same Ryzen 5 6600H, GCC 13.3.0 Release build, `MENSURFLOW_NATIVE=OFF`, and
`OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE` configuration were used. Power and background
conditions were not recorded. Each operation reads and writes one float per element.

Numerical behavior:

- `sqrt` uses float32 square root: negative inputs produce NaN, +inf produces +inf,
  and signed zeros are preserved.
- `exp` uses float32 exponential: overflow produces +inf, and underflow may produce
  +0. NaNs propagate, exp(-inf) is +0, and exp(0) is 1.
- `sigmoid` uses `1 / (1 + exp(-x))` for nonnegative inputs and
  `exp(x) / (1 + exp(x))` for negative inputs. The exponential argument is never
  positive for finite inputs, avoiding overflow in the negative tail. Infinities
  map to 0 and 1; NaNs propagate. The tests check that sigmoid(-100) stays positive
  because its result is representable as a float32 subnormal.

Sequential and parallel outputs are checked bit for bit, including in-place calls.
Against independent float64 references rounded to float32, the tests allow 4e-7
relative error plus one float32 subnormal step. NaNs are compared by classification,
while infinity signs and signed zeros must agree. Earlier exact-reference checks
for basic operations remain exact. These are test tolerances, not an accuracy
promise for every math-library implementation and floating-point environment.

The benchmark uses nonnegative input for `sqrt` to time valid roots. Other unary
inputs follow the existing pattern in [-0.75, 0.75]. Timing excludes allocations and
input preparation; domain errors and extreme values are covered by tests separately.
Values separated by `/` below are run 1 / run 2. Times are milliseconds; speedups
use unrounded internal timings. All measured outputs matched the sequential baseline.

| Op | n | Sequential (ms) | Parallel, 4 threads (ms) | Speedup, 2 threads | Speedup, 4 threads |
|---|---:|---|---|---|---|
| sqrt | 1,000 | 0.001 / 0.001 | 0.001 / 0.001 | 0.965x / 1.002x | 0.992x / 1.001x |
| sqrt | 10,000 | 0.011 / 0.011 | 0.011 / 0.011 | 0.996x / 1.000x | 1.000x / 1.000x |
| sqrt | 32,768 | 0.037 / 0.036 | 0.038 / 0.036 | 0.982x / 1.001x | 0.983x / 1.000x |
| sqrt | 32,769 | 0.038 / 0.036 | 0.010 / 0.010 | 1.972x / 1.937x | 3.608x / 3.736x |
| sqrt | 40,000 | 0.045 / 0.044 | 0.013 / 0.012 | 1.875x / 1.951x | 3.535x / 3.788x |
| sqrt | 50,000 | 0.057 / 0.055 | 0.016 / 0.014 | 1.926x / 1.958x | 3.500x / 3.820x |
| sqrt | 75,000 | 0.086 / 0.083 | 0.022 / 0.021 | 2.025x / 1.970x | 3.955x / 3.885x |
| sqrt | 100,000 | 0.112 / 0.111 | 0.029 / 0.028 | 1.983x / 1.975x | 3.817x / 3.910x |
| sqrt | 200,000 | 0.223 / 0.222 | 0.057 / 0.056 | 1.986x / 1.983x | 3.875x / 3.944x |
| sqrt | 500,000 | 0.558 / 0.555 | 0.148 / 0.139 | 1.997x / 1.998x | 3.769x / 3.982x |
| sqrt | 1,000,000 | 1.112 / 1.119 | 0.300 / 0.278 | 1.624x / 1.998x | 3.707x / 4.023x |
| sqrt | 2,000,000 | 2.226 / 2.236 | 0.576 / 0.599 | 1.896x / 1.935x | 3.864x / 3.733x |
| sqrt | 5,000,000 | 5.712 / 5.680 | 2.107 / 2.044 | 1.925x / 1.880x | 2.711x / 2.779x |
| sqrt | 10,000,000 | 11.405 / 11.524 | 4.964 / 4.736 | 1.839x / 1.842x | 2.297x / 2.433x |
| sqrt | 33,554,432 | 39.141 / 40.082 | 15.742 / 17.261 | 1.825x / 1.721x | 2.486x / 2.322x |
| exp | 1,000 | 0.002 / 0.003 | 0.003 / 0.003 | 0.916x / 0.923x | 0.910x / 0.929x |
| exp | 10,000 | 0.025 / 0.025 | 0.027 / 0.027 | 0.913x / 0.916x | 0.915x / 0.915x |
| exp | 32,768 | 0.085 / 0.082 | 0.092 / 0.089 | 0.906x / 0.917x | 0.921x / 0.918x |
| exp | 32,769 | 0.083 / 0.082 | 0.036 / 0.023 | 1.757x / 1.950x | 2.300x / 3.622x |
| exp | 40,000 | 0.117 / 0.100 | 0.050 / 0.028 | 1.864x / 1.976x | 2.343x / 3.573x |
| exp | 50,000 | 0.178 / 0.124 | 0.056 / 0.056 | 2.063x / 1.625x | 3.213x / 2.207x |
| exp | 75,000 | 0.211 / 0.297 | 0.082 / 0.066 | 1.617x / 2.938x | 2.571x / 4.505x |
| exp | 100,000 | 0.295 / 0.255 | 0.109 / 0.077 | 2.028x / 1.937x | 2.712x / 3.303x |
| exp | 200,000 | 0.512 / 0.517 | 0.207 / 0.145 | 1.828x / 2.012x | 2.470x / 3.577x |
| exp | 500,000 | 1.401 / 1.248 | 0.558 / 0.363 | 1.603x / 1.932x | 2.511x / 3.439x |
| exp | 1,000,000 | 2.700 / 2.514 | 0.839 / 0.806 | 1.949x / 1.868x | 3.218x / 3.119x |
| exp | 2,000,000 | 5.103 / 5.080 | 1.579 / 1.492 | 1.832x / 1.942x | 3.231x / 3.404x |
| exp | 5,000,000 | 12.857 / 12.620 | 4.039 / 3.544 | 1.910x / 1.964x | 3.183x / 3.561x |
| exp | 10,000,000 | 26.086 / 25.884 | 7.997 / 8.636 | 1.929x / 1.950x | 3.262x / 2.997x |
| exp | 33,554,432 | 90.810 / 84.488 | 31.802 / 24.918 | 1.920x / 1.934x | 2.855x / 3.391x |
| sigmoid | 1,000 | 0.003 / 0.003 | 0.003 / 0.003 | 0.901x / 0.952x | 0.813x / 0.948x |
| sigmoid | 10,000 | 0.033 / 0.028 | 0.030 / 0.029 | 1.081x / 0.885x | 1.109x / 0.938x |
| sigmoid | 32,768 | 0.095 / 0.092 | 0.102 / 0.093 | 0.954x / 0.982x | 0.933x / 0.983x |
| sigmoid | 32,769 | 0.094 / 0.090 | 0.033 / 0.026 | 1.589x / 1.910x | 2.828x / 3.483x |
| sigmoid | 40,000 | 0.114 / 0.110 | 0.055 / 0.032 | 1.680x / 1.915x | 2.065x / 3.463x |
| sigmoid | 50,000 | 0.155 / 0.136 | 0.068 / 0.040 | 1.582x / 1.891x | 2.277x / 3.410x |
| sigmoid | 75,000 | 0.222 / 0.206 | 0.066 / 0.066 | 1.619x / 1.838x | 3.373x / 3.121x |
| sigmoid | 100,000 | 0.274 / 0.280 | 0.084 / 0.084 | 1.804x / 1.899x | 3.280x / 3.334x |
| sigmoid | 200,000 | 0.549 / 0.548 | 0.175 / 0.164 | 1.720x / 1.896x | 3.129x / 3.334x |
| sigmoid | 500,000 | 1.416 / 1.351 | 0.631 / 0.432 | 1.928x / 1.834x | 2.245x / 3.130x |
| sigmoid | 1,000,000 | 2.787 / 2.746 | 0.858 / 0.841 | 1.887x / 1.787x | 3.247x / 3.263x |
| sigmoid | 2,000,000 | 5.602 / 5.518 | 1.799 / 1.698 | 1.868x / 1.778x | 3.114x / 3.250x |
| sigmoid | 5,000,000 | 14.096 / 14.715 | 4.416 / 4.866 | 1.836x / 1.754x | 3.192x / 3.024x |
| sigmoid | 10,000,000 | 30.067 / 27.505 | 8.972 / 8.575 | 1.949x / 1.811x | 3.351x / 3.207x |
| sigmoid | 33,554,432 | 95.447 / 91.080 | 31.080 / 30.364 | 1.772x / 1.743x | 3.071x / 3.000x |

Cutoff recommendation: keep the existing 32,768-element threshold as a conservative
initial setting on this machine with up to four threads. All three operations gain
from four threads at the first tested parallel size, 32,769, in both runs. Four threads
also outperform two there. Inputs at or below the cutoff still use the serial fallback,
so these measurements do not establish an optimal threshold below it.

Performance interpretation: exp and sigmoid are compute-heavy compared with the simple
unary operations and retain substantial parallel gains on large arrays. Square root
shows a mixture of computation and memory costs: four-thread speedup falls at large
sizes as data movement becomes more significant. These classifications describe this
build and workload, rather than every CPU or math library.

Some timings vary by more than 10% between runs. Both results are retained, including
outliers; precise ratios remain provisional. Reproduce through the common benchmark:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/cpu_parallel_benchmark
```

</details>

</details>
