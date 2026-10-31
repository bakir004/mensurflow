# CPU benchmarks: sequential vs parallel

Results from the CPU kernels (`add`, `mul`, `dot`, float32), measured on each machine we have, so
the numbers can be compared. Keep the build and run conditions below the same on every machine,
otherwise the comparison is not fair.

Code: `src/backends/cpu/parallel.hpp` (`parallel_for`, `parallel_sum`),
`src/backends/cpu/elementwise.cpp`, `src/backends/cpu/blas.cpp`.
Benchmarks: `benchmarks/cpu_parallel_benchmark.cpp` (sequential vs parallel, kernels called
directly), `benchmarks/blas/dot_benchmark.cpp` (through the public `mensurflow::dot`).

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

`threads = 1` in the output is the sequential run. The thread counts tested are 1, 2, 4, 8, 16,
half the hardware threads (the physical core count on SMT CPUs) and all hardware threads.

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
| Code | branch `feat/dispatcher-dot-elementwise`, commit `bdaef21`, plus `cpu_parallel_benchmark.cpp` (not yet committed) |
| Power state | **On battery, "Balanced" plan** when the specs were collected. The runs were probably in the same state, so the CPU may have been running below its full speed |

### Results (single run)

Best parallel time at each size, compared with the sequential (1 thread) run.

| Op | n | Sequential | Best parallel | Speedup |
|---|---|---|---|---|
| add | 100,000 | 0.024 ms | 0.048 ms (2 threads) | 0.5x (slower) |
| add | 1,000,000 | 0.498 ms | 0.372 ms (2 threads) | 1.34x |
| add | 10,000,000 | 6.50 ms | 6.17 ms (4 threads) | 1.05x |
| add | 33,554,432 | 22.8 ms | 21.3 ms (8 threads) | 1.07x |
| mul | 100,000 | 0.021 ms | 0.040 ms (2 threads) | 0.5x (slower) |
| mul | 1,000,000 | 0.436 ms | 0.333 ms (2 threads) | 1.31x |
| mul | 10,000,000 | 6.65 ms | 6.12 ms (2 threads) | 1.09x |
| mul | 33,554,432 | 22.3 ms | 20.5 ms (4 threads) | 1.09x |
| dot | 100,000 | 0.144 ms | 0.077 ms (4 threads) | 1.86x |
| dot | 1,000,000 | 1.27 ms | 0.292 ms (8 threads) | 4.3x |
| dot | 10,000,000 | 12.7 ms | 2.65 ms (8 threads) | 4.8x |
| dot | 33,554,432 | 42.7 ms | 9.24 ms (8 threads) | 4.6x |

What it shows:

- `dot` is limited by computation (products are accumulated in double precision, about 6 GB/s with
  one thread), so it scales: 4.6-4.8x with 8 threads and about 3.6x with 4.
- `add` and `mul` are limited by memory bandwidth. One thread already reaches about 18 GB/s at
  large sizes, so extra threads barely help (at most about 1.1x at 10 million elements or more).
- At 100,000 elements the threaded `add` and `mul` are 2-3x slower than sequential. The data fits
  in cache, so starting the threads costs more than it saves. The current cutoff (32,768 elements)
  is too low for the elementwise ops.
- An earlier run showed 1.2-1.6x for `add`/`mul` at 10 million elements, so treat those small gains
  as noise.

<details>
<summary>Full output: cpu_parallel_benchmark</summary>

```
Sequential vs parallel on the CPU (float32)
compiler: gcc 16.1.0, OpenMP: yes, max threads: 8, hardware threads: 8, parallel cutoff: 32768 elements
threads = 1 is the sequential run. Median of 9 samples.

== add ==
           n  threads          ms      GB/s    speedup
        1000        1       0.000    65.766     1.000x
        1000        2       0.000    65.042     0.989x
        1000        4       0.000    59.493     0.905x
        1000        8       0.000    55.644     0.846x

       10000        1       0.003    41.381     1.000x
       10000        2       0.003    40.698     0.984x
       10000        4       0.003    42.147     1.019x
       10000        8       0.003    42.738     1.033x

      100000        1       0.024    50.083     1.000x
      100000        2       0.048    25.242     0.504x
      100000        4       0.050    23.834     0.476x
      100000        8       0.069    17.319     0.346x

     1000000        1       0.498    24.104     1.000x
     1000000        2       0.372    32.230     1.337x
     1000000        4       0.429    27.984     1.161x
     1000000        8       0.486    24.681     1.024x

    10000000        1       6.504    18.450     1.000x
    10000000        2       6.273    19.129     1.037x
    10000000        4       6.168    19.455     1.054x
    10000000        8       6.557    18.300     0.992x

    33554432        1      22.811    17.652     1.000x
    33554432        2      24.255    16.601     0.940x
    33554432        4      22.687    17.748     1.005x
    33554432        8      21.329    18.879     1.069x


== mul ==
           n  threads          ms      GB/s    speedup
        1000        1       0.000    95.963     1.000x
        1000        2       0.000    92.840     0.967x
        1000        4       0.000    86.899     0.906x
        1000        8       0.000    90.875     0.947x

       10000        1       0.002    70.534     1.000x
       10000        2       0.002    71.887     1.019x
       10000        4       0.002    76.525     1.085x
       10000        8       0.001    91.452     1.297x

      100000        1       0.021    57.446     1.000x
      100000        2       0.040    29.942     0.521x
      100000        4       0.044    27.062     0.471x
      100000        8       0.065    18.506     0.322x

     1000000        1       0.436    27.547     1.000x
     1000000        2       0.333    36.025     1.308x
     1000000        4       0.398    30.167     1.095x
     1000000        8       0.504    23.826     0.865x

    10000000        1       6.648    18.051     1.000x
    10000000        2       6.122    19.602     1.086x
    10000000        4       6.260    19.169     1.062x
    10000000        8       6.133    19.565     1.084x

    33554432        1      22.325    18.036     1.000x
    33554432        2      23.248    17.320     0.960x
    33554432        4      20.474    19.666     1.090x
    33554432        8      20.908    19.259     1.068x


== dot ==
           n  threads          ms      GB/s    speedup
        1000        1       0.001     6.636     1.000x
        1000        2       0.001     6.784     1.022x
        1000        4       0.001     5.612     0.846x
        1000        8       0.001     5.400     0.814x

       10000        1       0.013     6.182     1.000x
       10000        2       0.013     6.299     1.019x
       10000        4       0.013     6.365     1.030x
       10000        8       0.012     6.747     1.091x

      100000        1       0.144     5.561     1.000x
      100000        2       0.139     5.757     1.035x
      100000        4       0.077    10.328     1.857x
      100000        8       0.090     8.860     1.593x

     1000000        1       1.267     6.315     1.000x
     1000000        2       0.701    11.408     1.807x
     1000000        4       0.410    19.534     3.093x
     1000000        8       0.292    27.363     4.333x

    10000000        1      12.708     6.295     1.000x
    10000000        2       6.748    11.856     1.883x
    10000000        4       3.540    22.602     3.590x
    10000000        8       2.648    30.207     4.799x

    33554432        1      42.681     6.289     1.000x
    33554432        2      21.366    12.564     1.998x
    33554432        4      11.803    22.742     3.616x
    33554432        8       9.236    29.062     4.621x
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

## Machine 2: desktop (to fill in)

| | |
|---|---|
| CPU | AMD Ryzen 9 5900X (Zen 3, two core complexes) |
| Cores / threads | 12 / 24 |
| Clock | 3.7 GHz base, up to 4.8 GHz boost (published figures) |
| Cache | 64 MB L3 (published figure); L2: fill in |
| RAM | fill in: size, type, speed, number of sticks (dual channel?) |
| OS | fill in |
| Compiler | fill in (name and version; note if it differs from the laptop's GCC 16.1.0) |
| OpenMP | fill in (MSVC supports only OpenMP 2.0; GCC supports 5.x) |
| Code | commit hash: fill in |
| Power state | fill in (plugged in / High performance?) |

Results: paste the output of `cpu_parallel_benchmark` and `dot_benchmark` here, in the same
`<details>` blocks as above, and fill in the summary table.

### Comparison (to fill in)

Best speedup over sequential at 10,000,000 elements, and sequential throughput.

| Op | Laptop speedup | Desktop speedup | Laptop sequential GB/s | Desktop sequential GB/s |
|---|---|---|---|---|
| add | 1.05x | | 18.5 | |
| mul | 1.09x | | 18.1 | |
| dot | 4.8x | | 6.3 | |

Things to look for on the desktop:

- Does `add`/`mul` scale better? A desktop with dual-channel DDR4 has more memory bandwidth, so more
  threads may help where the laptop was already limited.
- Does `dot` keep scaling past 8 threads, and where does it stop (12 physical cores vs 24 threads)?
- Is the threading cutoff still wrong at 100,000 elements? With more cores, starting the threads
  costs more, so the slowdown may be bigger.
- Compiler effect: if the desktop uses MSVC and the laptop GCC, differences can come from the
  compiler, not the hardware. Building the desktop with MinGW as well separates the two.
