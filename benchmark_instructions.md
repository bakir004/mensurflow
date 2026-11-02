# Benchmark instructions

How to write CPU benchmarks in the same style as this project: a small harness, one runner function per kernel group, and a correctness check against a scalar reference before every timing.

## 1. File layout

```
include/bench.hpp        Config, Sample, measure(), Report, checksum/consume
src/bench.cpp            consume, checksum, fill_floats, Report::print_summary, CPU check
include/<kernel>.hpp     function-pointer types + declarations of each variant
src/<kernel>_scalar.cpp  reference (vectorization disabled)
src/<kernel>_auto.cpp    plain loop, compiler may vectorize
src/<kernel>_omp.cpp     #pragma omp simd
src/<kernel>_avx2.cpp    intrinsics (compiled with /arch:AVX2 or -mavx2 only for this file)
src/benchmark_runner.cpp one run_xxx() per kernel group
src/main.cpp             sets Config, calls runner.run()
```

Keep the variants in separate `.cpp` files so CMake can give each file its own flags (no vectorization for the scalar file, `/arch:AVX2` for the AVX2 file).

## 2. The harness (`bench.hpp`)

```cpp
struct Config { int warmups = 1; int repeats = 10; std::size_t n = 1u << 17; };
struct Sample { double median_s, min_s; std::vector<double> runs_s; };

PERF_NOINLINE void consume(double v);          // XOR bits into a global atomic
PERF_NOINLINE double checksum(const float*, std::size_t);

template <class Run, class Observe>
Sample measure(const Config& c, Run run, Observe observe) {
    for (int i = 0; i < c.warmups; ++i) { run(); consume(observe()); }
    std::vector<double> secs;
    for (int i = 0; i < c.repeats; ++i) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        auto t0 = std::chrono::steady_clock::now();
        run();
        auto t1 = std::chrono::steady_clock::now();
        std::atomic_signal_fence(std::memory_order_seq_cst);
        consume(observe());                    // outside the timed region
        secs.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    // sort a copy, take median + min, keep the raw runs
}
```

Rules this enforces:

- **Warmup first.** It primes caches, page faults and the CPU clock.
- **Time only `run()`.** `observe()` (the checksum) runs after the timer stops.
- **Defeat dead-code elimination.** `consume` and `checksum` are `noinline` and write to an atomic, so the compiler cannot drop the work.
- **Report median, min and max.** The median is robust against outliers.
- **Reset state** before or inside `run()` if the kernel modifies its input (accumulating kernels like SAXPY drift otherwise).

## 3. Writing one benchmark group

```cpp
void BenchmarkRunner::run_mykernel() {
    const Config& cfg = _config;
    const std::size_t n = cfg.n + 3;                 // odd size exercises vector tails
    std::vector<float> in(n), out(n), expected(n);
    fill_floats(in.data(), n, 1234);                 // fixed seed -> reproducible

    mykernel_scalar(in.data(), expected.data(), n);  // reference

    struct Entry { const char* name; MyFn fn; };
    std::vector<Entry> impls = {
        {"scalar", mykernel_scalar}, {"auto", mykernel_auto}, {"omp simd", mykernel_omp}};
    if (avx2_supported()) impls.push_back({"AVX2", mykernel_avx2});

    for (const auto& e : impls) {
        e.fn(in.data(), out.data(), n);
        check_vector(expected, out.data(), n);       // throws on mismatch
        auto s = measure(cfg,
            [&]{ e.fn(in.data(), out.data(), n); },
            [&]{ return checksum(out.data(), n); });
        _report.add("MyKernel / " + std::string(e.name), s,
                    n / s.median_s / 1e6, "Melems/s");
    }
}
```

Call it from `BenchmarkRunner::run()` and keep `print_summary()` at the end.

## 4. Choosing tolerances

| Kernel kind | Check |
|---|---|
| Elementwise, same operations | `check_vector`, `1e-4 * (1 + abs(expected))` |
| Exact (integer, comparisons) | tolerance `0` |
| Reductions (dot, sum) | compare with a `double` accumulation, relative error about `5e-3` (summation order legitimately changes the float result) |
| Structures (SoA vs AoS) | per-field `verify` that throws with the index |

## 5. Metrics per kernel

Pick a unit that makes variants comparable:

- Memory-bound (SAXPY): `bytes_moved / median / 1e9` -> GB/s. SAXPY reads x, reads y, writes y, so `12 * n` bytes.
- Compute-bound (matmul): `2*n^3 / median / 1e9` -> GFLOP/s.
- Simple per-element work: `n / median / 1e6` -> Melems/s.

## 6. CPU feature check

Never call an AVX2 function on a CPU without AVX2. `avx2_supported()` does:

1. Compile time: returns `false` unless `PERFLABS_HAS_AVX2_BINARY` is defined (x86 target and option enabled).
2. MSVC runtime: `__cpuid` leaf 0 (max leaf >= 7), leaf 1 (OSXSAVE bit 27, AVX bit 28), `_xgetbv(0) & 0x6 == 0x6` (OS saves YMM state), leaf 7 EBX bit 5 (AVX2).
3. GCC/Clang runtime: `__builtin_cpu_supports("avx2")`.

Only add the `*_avx2` entries to the variant tables when it returns `true`.

## 7. CMake

```cmake
# scalar reference: no auto-vectorization
set_source_files_properties(src/mykernel_scalar.cpp PROPERTIES COMPILE_OPTIONS
    "$<IF:$<CXX_COMPILER_ID:MSVC>,/Qvec-,-fno-tree-vectorize>")
# AVX2 file only
set_source_files_properties(src/mykernel_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
target_compile_definitions(app PRIVATE PERFLABS_HAS_AVX2_BINARY=1)
```

The scalar loops can additionally use `PERF_NO_VECTORIZE` (`__pragma(loop(no_vector))` on MSVC). `PERF_RESTRICT` and `PERF_NOINLINE` are macros in `bench.hpp`.

## 8. Wall time vs CPU time

`steady_clock` measures wall-clock time, which is the right choice for comparing vectorized variants on one thread. For OpenMP kernels also record CPU time (total across threads): `GetProcessTimes` on Windows, `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` on Linux. For a multithreaded kernel, CPU time can be several times the wall time.

## 9. Checklist for credible numbers

- Build in **Release** (`/O2` or `-O2`/`-O3`). Debug builds make the comparison meaningless.
- Make the working set match what you intend: `n = 2^17` floats is 512 KB (L2). Larger sizes measure memory bandwidth.
- Close other heavy programs and use a fixed power plan.
- Print the CPU model and AVX2 yes/no in the report header so results are traceable (CPUID leaves `0x80000002` to `0x80000004` give the CPU name).
- Never time the correctness check.
