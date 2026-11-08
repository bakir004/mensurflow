# Matmul optimization log

Goal: float32 CPU `matmul` as fast as OpenBLAS, then faster. This file records each improvement, what it
does and why, and the numbers measured. It is updated as the work goes on.

**All numbers are GFLOP/s (`2*n^3 / time`) for square matrices, from single runs on one machine:
Ryzen 9 5900X (12 cores, 24 threads, 512 KB L2 per core), MSVC Release, OpenMP, no
`MENSURFLOW_NATIVE`. They are noisy (a few percent, and much more for OpenBLAS at 24 threads).** The
i7-10510U laptop has not been measured yet. OpenBLAS is the prebuilt 0.3.34 x64 release, run through
`cblas_sgemm`.

## Summary of improvements

| # | Improvement | What it changes | Measured effect |
|---|---|---|---|
| 1 | OpenBLAS comparison benchmark | A reference to measure against, at the same thread counts | OpenBLAS: about 100 GFLOP/s on 1 thread, 650-1000 on 24 threads |
| 2 | Packed 6x16 AVX2/FMA kernel (1 thread) | Register tile plus packed A and B panels instead of the i-k-j loop | 22 -> 100 GFLOP/s at n = 1024 (4.5x), equal to OpenBLAS |
| 3 | Run-time AVX2/FMA detection | The kernel is chosen on the CPU it runs on, with a safe fallback | Same binary works on CPUs without AVX2 |
| 4 | Threaded version (tiles of C) | Packing and compute shared across threads, no locks | 1 -> 24 threads: 100 -> 880-1000 GFLOP/s at n = 2048 |
| 5 | Reusable packing buffers | No allocation and page faults on every call | n = 128, 1 thread: 42 -> 85 GFLOP/s |
| 6 | Fewer threads for small products | Each thread needs at least 512K multiply-adds | n = 128, 24 threads: 53 -> 111-131 GFLOP/s |
| 7 | Cache-aware block sizes | KC = 1024, MC = 240, NC from the L3 size and 16 tiles per thread when the L2 is 512 KB or more, all from CPUID cache sizes | +3% to +30% at 8-24 threads in the sweeps; n = 2048 on 1-8 threads moved from 0.88x-0.97x to 0.94x-0.99x of OpenBLAS |
| 8 | Wired into `mensurflow::matmul` | The CPU backend uses the packed kernel when AVX2 is available | Public `matmul` gets all of the above |
| 9 | Fair benchmark methodology | Pause before each timed call so spinning worker threads do not steal cores | Removed a false "behind OpenBLAS" gap (376 vs 620 GFLOP/s at n = 1024, 8 threads) |
| 10 | Tuning tools | `gemm_tune_benchmark` and `set_packed_gemm_blocking()` | Makes the block-size sweeps repeatable on any machine |
| 11 | Other micro-kernel tile shapes | 6x16 unrolled, 4x24 and 12x8 kernels, selectable per call, all tested | None is faster than 6x16 on this CPU (4x24: 0.92x-0.99x, 12x8: 0.79x-0.95x, unrolled: 0.75x-0.85x); kept as tile variants for the autotuner |
| 12 | Register-level analysis and measured FMA peak | Disassembly of every kernel plus a register-only FMA benchmark (`fma_peak_benchmark`) | 6x16 has no spills and runs at 90-93% of the measured 112 GFLOP/s per-core peak (the 135-145 estimate was wrong: the core holds about 3.5 GHz under AVX2); the unrolled variant spills, which explains its slowdown |
| 13 | Hand-vectorized packing | AVX2 `pack_a` (8 x 8 register transpose) and `pack_b` (256-bit copies) replace scalar loops | Packing 3.0-3.6x (A) and 1.4-2.3x (B) faster alone; 1-6% faster matmul end to end; 106 GFLOP/s on 1 thread at n = 1024 (95% of peak) |
| 14 | Threading: pinning, placement, loop order | Each thread pinned to its own physical core (adjacent cores first), strategy 1 (one parallel region), A panel kept in L1, 16 tiles per thread, no SMT threads for small products | Removes a random 40-45% slowdown (n = 2048, 12 threads: 597 -> 1036); placement on one chiplet +3-45% at 2-6 threads; loop order +4-7% at n = 2048; 4 strategies compared, 3 kept as options |
| 15 | Strided operands | The packing step reads transposed, sliced and padded matrices in place (new fast paths for transposed A and B); `matmul` no longer copies them to contiguous memory on the CPU | Transposed operands 2.5-3.8x faster (one operand), 4.0-6.6x (both, the backward pass); same speed as contiguous; leading dimension (n, n+16, n+1) makes no difference |
| 16 | Packing overlapped with compute (strategy 4, now the default), KC = 768 | The next K step is packed into a second buffer while this one is multiplied: one barrier per step instead of two | Barrier wait 4-7% -> 1.4-3%; +3-12% over strategy 1 at n = 1536-4096 (n = 4096, 24 threads: 1147 -> 1208 GFLOP/s in the thread benchmark) |
| 17 | Team split by chiplet for small products (strategy 5) | Each L3 domain computes its own columns of C from its own copy of A and its own B panels; used automatically below 2^29 multiply-adds | n = 256-768 on 12 threads: +11% to +23% (n = 512: 677 -> 811 GFLOP/s) |
| 18 | Multi-panel `pack_b` | B is packed 4 panels per pass over the rows, so each source row is read as a run of 4 cache lines | Packing B alone 1.5-1.9x faster; no measurable end-to-end change with threads (packing is DRAM-bound there and already hidden by 16) |
| 19 | Approaches measured and rejected | Tiles per thread re-swept for strategy 4; masked edge tiles considered | 16 tiles stays best; partial edge tiles cost at most 2% (n = 1025, 2047), not worth a kernel |
| 20 | Manual unrolling and broadcast placement in the 6x16 kernel | 8 new variants (unrolled x2/x4, broadcasts pipelined one row ahead in the 16th register, with and without a compiler fence), a kernel-only benchmark, assembly of each | The default kernel already runs at 97.7% of the core's FMA peak by itself. Unrolling without a fence spills (64-75% of peak); with the fence it is spill-free but equal (98.0%, noise); pipelining costs ~1%. Default unchanged |
| 21 | Prefetching, second round (write-intent C prefetch now the default) | 6 kernel prefetch modes, 4 between-tile modes, prefetch in the B packer; also fixed `gemm_tune_benchmark` (variants started from a blank configuration, not the defaults) | PREFETCHW of the C tile before the depth loop: 0 to +3.5% (n = 1024, 12 threads: 914 -> 942 GFLOP/s), never more than 0.7% behind. B prefetch, next-tile and next-A-panel prefetch: no gain or worse |

## The improvements, explained

### 1. OpenBLAS comparison benchmark

`benchmarks/blas/matmul_openblas_benchmark.cpp` runs our sequential loop, our old blocked-parallel kernel,
the new packed kernel and OpenBLAS side by side at the same thread counts and prints a `vs obl` column
(OpenBLAS time / this row's time, so 1.00 means as fast as OpenBLAS). Every result is checked against the
sequential reference before it is timed. CMake builds it when OpenBLAS is found (`-DOPENBLAS_ROOT=...`,
option `MENSURFLOW_OPENBLAS`). How to set it up: [BENCHMARKING.md](BENCHMARKING.md).

### 2. Packed 6x16 AVX2/FMA micro-kernel (the Goto/BLIS design)

The old kernel computed one row of C at a time, loading and storing C for every product. The new one
(`src/backends/cpu/ops/blas/gemm_packed.cpp`) works like BLIS and OpenBLAS:

| Piece | What it is | Why it is fast |
|---|---|---|
| Register tile | A 6 x 16 block of C held in 12 of the 16 AVX2 registers for the whole depth loop | C is read and written once per tile, not once per product; each loaded B vector feeds 6 FMAs |
| Packing | A is copied into 6-row panels and B into 16-column panels, in the exact order the kernel reads them | The kernel streams through memory in order: no strides, no cache conflicts |
| Zero padding | Panels are padded to full size | The kernel never branches on edges; partial tiles are computed in full and only the valid part is copied out |
| Five loops | NC columns, KC depth, MC rows, then the 16-wide and 6-tall panels | A block of A stays in L2, a block of B in L3, one B panel in L1 |
| FMA | `_mm256_fmadd_ps` | 2 FMA units x 8 floats x 2 flops = 32 flops per cycle per core at best |

Single-thread results (Ryzen 9 5900X):

| n | old sequential loop | packed | OpenBLAS, 1 thread | packed / OpenBLAS |
|---|---|---|---|---|
| 64 | 20.7 | 67 | 53 | 1.26x |
| 128 | 22.9 | 85 | 72 | 1.17x |
| 256 | 23.0 | 96 | 88 | 1.09x |
| 512 | 22.5 | 100 | 97 | 1.04x |
| 1024 | 22.2 | 102 | 101 | 1.00x |
| 2048 | 21.8 | 100 | 103 | 0.97x |

The measured peak of one core is 112 GFLOP/s (section 12), so the kernel reaches 90-93% of it.

### 3. Run-time CPU detection

The kernel is compiled with AVX2/FMA intrinsics but only used when `CPUID` and `XGETBV` say the CPU and
the OS support them (`packed_gemm_supported()`). On a CPU without them, or on a non-x86 build, the
backend keeps the old kernels. This keeps one portable binary, which is why `MENSURFLOW_NATIVE` stays off.
With GCC or Clang the kernel function carries a `target("avx2,fma")` attribute instead of a global flag.

### 4. Threaded version

`matmul_f32_packed_parallel` splits the work without locks. This is the first version (strategy 0 in section
14; the default is now strategy 1, the same work in a single parallel region). Per (NC, KC) step:

1. One parallel loop packs all panels of B and all panels of A.
2. A second parallel loop hands out tiles of C (an MC-row block times a group of B panels) with dynamic
   scheduling. Every thread reads the shared packed A and B and writes only its own tile of C.

Groups of B panels are sized to give about `tiles_per_thread` tiles per thread, so small matrices (a
single row block) are split along the columns instead. It uses the repository's
`MENSURFLOW_PARALLEL_FOR` macro (and, from section 14 on, `MENSURFLOW_PARALLEL_REGION`, `MENSURFLOW_FOR` and
`MENSURFLOW_BARRIER` from `threads.hpp`), so there are no raw pragmas in the kernel.

| n | threads | old blocked parallel | packed parallel | OpenBLAS |
|---|---|---|---|---|
| 512 | 8 | 25 | 514 | 494 |
| 512 | 24 | 63 | 647 | 481 |
| 1024 | 8 | 47 | 624 | 662 |
| 1024 | 24 | 55 | 880 | 965 |
| 2048 | 8 | 41 | 670 | 764 |
| 2048 | 24 | 60 | 857 | 645 |

(Old blocked parallel numbers are from the first OpenBLAS comparison run; the others from the final
run with the fair methodology below.)

### 5. Reusable packing buffers

The packed buffers were allocated and freed on every call. Fresh memory has to be page-faulted in, which
cost as much as the multiplication itself for small matrices. The buffers are now thread-local, grown on
demand and kept between calls (freed when the calling thread exits). Cost: some memory stays allocated
per calling thread after a large call.

| n = 128 | before | after |
|---|---|---|
| 1 thread | 42 GFLOP/s (0.58x OpenBLAS) | 85 GFLOP/s (1.17x OpenBLAS) |

### 6. Fewer threads for small products

Waking and joining threads costs more than the work they share when the product is small. At n = 128 the
kernel was fastest on 4 of 24 threads. Each thread now needs at least 512K multiply-adds
(`min_macs_per_thread`), so the thread count is `min(max_threads, multiply-adds / 512K)`.

| n = 128, 24 threads | before | after | OpenBLAS |
|---|---|---|---|
| GFLOP/s | 53 | 111-131 | 56-59 |

### 7. Cache-aware block sizes

The block sizes were swept in three rounds on the Ryzen 9 5900X (L1d 32 KB, L2 512 KB per core, L3 32 MB),
each time re-timing the best five against the default in interleaved rounds. CPUID gives the cache sizes
(`CacheSizes cpu_cache_sizes()`; Intel leaf 4, AMD leaf 0x8000001D) and the defaults are derived from them.

| Round | What was swept | Finding |
|---|---|---|
| 1 | 1 thread; MC 120-720, KC 256-1024, NC 2048-8160; n = 1024, 2048 | Flat: every setting within 2% (KC 1024 marginally best). One thread is limited by the micro-kernel, not by cache blocking |
| 2 | 8 and 24 threads; MC 240-720, KC 384-1024, NC 4080-8160, 4/8/16 tiles per thread; n = 2048, 4096 | KC = 1024 and 16 tiles per thread win everywhere (+3% to +30% over the old default); NC = 8160 helps at n = 4096; MC 240-720 about equal |
| 3 | 8 and 24 threads; KC 1024-2048, 16-64 tiles per thread | Plateau: KC 1024 to 2048 and 16 to 64 tiles per thread equal within noise |
| extra | Software prefetch (C tile, and A/B panels 8 steps ahead), 1 and 24 threads | No effect (1005-1008 GFLOP/s in all three modes); left off (`prefetch = 0`), the option stays for other CPUs. Revisited in section 21: a write-intent prefetch of the C tile is now on |

Parameter by parameter:

| Parameter | Chosen | Why |
|---|---|---|
| KC (depth step) | 1024 | Fewer pack-and-barrier steps in the threaded version and fewer passes over C. It works even though the 240 x 1024 block of A (960 KB) is larger than the L2: the panels are read in order and the hardware prefetcher keeps up. So "A block fits L2" is not the binding rule on this CPU |
| NC (columns of B) | L3 / (KC x 4 bytes), rounded down to a multiple of 16, clamped to 4080..8160 | NC = 8160 with KC = 1024 is a 33 MB block of B, about the size of the 32 MB L3. Measured NC 1024 to 4080 are flat; 8160 helps at n = 4096 |
| MC (rows of A) | 240 | 240 to 720 are equal within noise; the smaller value gives more row blocks and so better tile balance |
| Tiles per thread | 16 | 16 to 64 are equal; fewer (4, 8) were slower at 24 threads |

| L2 per core | MC | KC | NC | tiles per thread |
|---|---|---|---|---|
| 512 KB or more (this Ryzen) | 240 | 1024 | L3-derived, 8160 here | 16 |
| smaller, or not detected | 168 | 256 | 4080 | 4 (BLIS's Haswell values) |

The small-L2 row is a conservative default, not a measured optimum: a 240 x 1024 block is far too big
for a 256 KB L2 without being measured, so the laptop still needs its own sweep.

End to end (`matmul_openblas_benchmark`, pauses between measurements), the packed kernel against OpenBLAS
after this change:

| n | threads 1-8 | threads 12 | threads 24 |
|---|---|---|---|
| 128 | 1.08x-1.62x | 1.58x | 1.59x |
| 256 | 1.08x-1.22x | 1.68x | 2.23x |
| 512 | 1.01x-1.07x | 1.12x | 1.50x |
| 1024 | 1.00x-1.04x | 1.06x | 0.95x |
| 2048 | 0.94x-0.99x | 1.25x | 1.56x |

Before this change n = 2048 on 1-8 threads was 0.88x-0.97x.

### 8. Wired into `mensurflow::matmul`

`src/backends/cpu/backend.cpp` calls `matmul_f32_packed_parallel` (OpenMP build) or `matmul_f32_packed`
(no OpenMP) when AVX2/FMA is available, and the old kernels otherwise. All 18 tests pass. Results can
differ from the old kernels in the last bits because the packed kernel uses fused multiply-adds; the
tests compare against a float64 reference with a tolerance.

### 9. Fair benchmark methodology

The first comparison made the packed kernel look about 10% behind OpenBLAS at 4-8 threads. Part of that
was the benchmark: OpenBLAS's and OpenMP's worker threads keep spinning after a call and take cores and
SMT execution units from whatever is timed next. The same kernel read 376 GFLOP/s at n = 1024 with 8
threads right after an OpenBLAS call, and 620 after a pause. `settle()` now sleeps 500 ms before every
measurement so each library starts from idle threads. Results measured without the pause are not
comparable with the ones above.

A first guess for the slow n = 128 case (OpenBLAS threads spinning) was tested and rejected: the cause
was the per-call buffer allocation (improvement 5).

### 10. Tuning tools

| Tool | Use |
|---|---|
| `gemm_tune_benchmark key=list ...` (`threads=`, `sizes=`, `mc=`, `kc=`, `nc=`, `tiles=`, `prefetch=`, `shape=`, `pack=`); `gemm_thread_benchmark` for the threading options | Sweeps any combination of block sizes, re-times the best five and the default alternately, prints the confirmed best and the CPU's cache sizes |
| `set_packed_gemm_blocking()` / `packed_gemm_blocking()` | Sets and reads the block sizes at run time (not thread-safe; for tuning only) |
| `gemm_packed_test` | Checks the sequential and threaded kernels against a float64 reference on 22 shapes around every tile and block edge, at 1, 2, 3, 8 and 24 threads |

### 11. Micro-kernel tile shapes

Cache blocking is exhausted, so the next lever is the register tile itself. An AVX2 core has 16 vector
registers; a tile of MR x NR floats needs MR x NR / 8 accumulators, NR / 8 registers for the B row and one
for the broadcast A value. All shapes below keep 12 accumulators (96 floats of C) and differ in how many
loads each FMA needs. The shape is chosen with `PackedGemmBlocking::shape`; packing, edge handling and the
threaded driver work for any shape, and `gemm_packed_test` checks every shape (also with tiny block sizes
that force all block edges on small matrices).

| Shape | Registers | Loads per 12 FMAs | `shape` | n = 256..2048, 1 thread | 8-24 threads (n = 1024, 2048) |
|---|---|---|---|---|---|
| 6 x 16 (baseline) | 12 acc + 2 B + 1 = 15 | 8 | 0 | 95-104 GFLOP/s | best, 1.00x |
| 4 x 24 | 12 acc + 3 B + 1 = 16 | 7 | 2 | 0.97x-0.99x of the baseline | 0.92x-0.97x |
| 12 x 8 | 12 acc + 1 B + 1 = 14 | 13 | 3 | 0.89x-0.91x | 0.79x-0.95x |
| 6 x 16, depth loop unrolled x4 | 15 | 8 | 1 | 0.75x-0.79x | 0.79x-0.85x |

Findings:

- The baseline 6 x 16 is the fastest at every size and thread count. 4 x 24 is close (it has the fewest
  loads per FMA, but the 3-vector B row and 4-row A panel cost more elsewhere) and 12 x 8 pays for 13
  loads per 12 FMAs.
- Manual 4x unrolling made the kernel about 25% slower. Not investigated: the MSVC code generation for
  the unrolled body (instruction order or register allocation) is the suspect, and the disassembly has
  not been read.
- The other shapes are kept as tile variants. They are what the autotuner in the weekly plan can choose
  between on CPUs where the ranking may differ (for example a CPU with fewer load ports, where 4 x 24 could
  win). On this Ryzen no variant beats the baseline.
- The last runs happened while the machine was slower than earlier ones (the 8-thread n = 2048 rate was
  about 400 instead of 700 GFLOP/s), so the absolute multi-thread numbers of this experiment are not
  comparable with the other tables; the ranking was identical in every run.

### 12. Register-level analysis: the micro-kernel is at the hardware limit

Two measurements settle how much the micro-kernel can still give.

**The real FMA peak.** `fma_peak_benchmark` runs independent AVX2 FMAs on registers only (no loads or
stores) with the same compiler:

| Accumulators (1 thread) | GFLOP/s | Reading |
|---|---|---|
| 1 | 14.0 | One dependent chain: an FMA every 4 cycles, so the core runs at about 3.5 GHz under AVX2 load |
| 4 | 55.9 | Scales linearly: latency bound |
| 8 | 111.7 | Saturates: 2 FMA units x 4-cycle latency need 8 independent chains |
| 10, 12, 14 | 111.8 | Flat: more chains add nothing |

| Threads (12 accumulators each) | GFLOP/s | Per thread |
|---|---|---|
| 1 | 111.7 | 111.7 |
| 4 | 443.9 | 111.0 |
| 12 (one per physical core) | 1318.6 | 109.9 |
| 24 (SMT) | 1333.0 | 55.5 |

So the peak is **112 GFLOP/s per core and about 1320 GFLOP/s for the chip**; the second SMT thread of a
core adds nothing to a kernel that is already FMA-bound. The earlier estimate of 135-145 GFLOP/s per core
assumed a boost clock this machine does not hold under AVX2 load.

**The generated code** (`dumpbin /disasm` of `gemm_packed.obj`):

| Kernel | Registers in the loop | Spills | Loop body |
|---|---|---|---|
| 6 x 16 | 12 accumulators `ymm3..ymm14`, B in `ymm1/2`, broadcast in `ymm0` | none | 2 loads, 6 broadcasts, 12 FMAs, 2 pointer `lea`, `sub`, `jne` (23 instructions) |
| 4 x 24 | 12 accumulators, 3 B, 1 broadcast: all 16 registers | none | 3 loads, 4 broadcasts, 12 FMAs, 4 bookkeeping |
| 12 x 8 | 12 accumulators, B, broadcast | none | 1 load, 12 broadcasts, 12 FMAs, 4 bookkeeping |
| 6 x 16 unrolled x4 | needs more than 16 | **yes: about a dozen accumulators stored to and reloaded from the stack every iteration** | spill traffic explains its 25% loss |

A depth step of the 6 x 16 kernel needs 6 cycles on the two FMA units and about 4 cycles to issue its 23
instructions, so it is FMA bound: loop unrolling and prefetching cannot help (and measured no gain). The
fixed per-call costs (saving `xmm6..xmm14` as the Windows ABI requires, the per-row `accumulate` test)
are below 1% at KC = 1024.

**Efficiency against the measured peak (n = 1024-2048):**

| Threads | Packed | OpenBLAS | Peak | Packed efficiency |
|---|---|---|---|---|
| 1 | 100-104 | 101-104 | 111.8 | **90-93%** |
| 12 | 906-922 | 724-872 | 1319 | 69-70% |
| 24 | 880-1037 | 664-974 | 1333 | 66-78% |

Conclusion: single-threaded, the register-level work is finished: the kernel is within about 7% of what the
hardware can do, and the small remainder is packing (around 3% at n = 2048) and C updates. The headroom
that is left is in the threaded version (66-78% of the chip's peak): packing, the barrier between packing
and compute for every K block, and tile balance.

### 13. Hand-vectorized, register-aware packing

The disassembly (section 12) showed both packing loops were plain scalar code: one `mov` load and one `mov`
store per float, an `imul` per element for the strided A writes, and a `memset` call for padding. They are
now written with AVX2 intrinsics in `src/backends/cpu/ops/blas/gemm_pack.hpp`.

| Packer | Technique | Registers |
|---|---|---|
| `pack_b` (copy 16 / 24 / 8 contiguous floats per depth step) | 256-bit loads and stores, 4 depth steps per iteration, all loads before all stores | 4 x (NR / 8) = 8 (NR = 16) to 12 (NR = 24) of 16 |
| `pack_a` (6-row panels; a transpose, A rows become columns of 6) | 8 x 8 in-register transpose (`unpacklo/hi`, `shuffle`, `permute2f128`), 8 depth steps per iteration. Rows 6 and 7 do not exist, so they are constant zeros that fold away | up to the full 16 ymm, no spills |

Details that matter:

- Column q of the transposed block (depth step p + q) holds 6 valid floats. A 256-bit store writes 8, so
  the two extra floats land in the next slot and are overwritten by the next store; the last column of each
  block is written as 4 + 2 floats so that nothing beyond the panel is touched. This is required because
  the threaded driver packs different panels on different threads.
- Partial panels (fewer than 6 rows or 16 columns) and depth tails use the scalar code with zero padding.
- Other micro-kernel shapes: `pack_b` is vectorized for NR = 8, 16, 24; `pack_a` only for MR = 6 (4 and
  12 use the scalar packer, which is fine for variants that are not the default).
- `scalar_pack = 1` in `PackedGemmBlocking` (and `pack=1` in `gemm_tune_benchmark`) selects the scalar
  packers, so the two can be timed in the same process.

`gemm_pack_test` compares both packers against the scalar reference for every panel shape and for sizes
around the vector widths (strided sources too), and checks that 64 guard words after each destination are
never written. `gemm_pack_benchmark` times them alone:

| Block | scalar | AVX2 | Speedup |
|---|---|---|---|
| A 240 x 1024 | 25.4 GB/s | 76.9 GB/s | 3.0x |
| A 240 x 256 | 22.7 GB/s | 72.8 GB/s | 3.2x |
| A 24 x 1024 | 26.6 GB/s | 94.6 GB/s | 3.6x |
| B 1024 x 4080 | 6.4 GB/s | 14.4 GB/s | 2.3x |
| B 256 x 4080 | 22.6 GB/s | 31.6 GB/s | 1.4x |
| B 1024 x 256 | 18.9 GB/s | 30.4 GB/s | 1.6x |

`pack_a` reads 6 sequential streams, so it runs close to cache bandwidth. `pack_b` is much slower even
vectorized because of its access pattern: it takes one cache line from each of the kc matrix rows, a
stride of `ldb` floats (a large power of two for n = 2048, 4096), and uses each line once; the hardware
prefetcher does not follow strides that large and the page and cache-set mapping is poor. Software
prefetching 8, 16, 32 or 64 rows ahead was tried and gave no consistent gain (+12% on one block, -15% on
another), so it was removed. Reading several panels per row (so each source row is read contiguously) is
the remaining idea and was not tried.

End to end, scalar packing against AVX2 packing (same process, GFLOP/s relative to the AVX2 default):

| Threads | n = 256 | n = 1024 | n = 2048 |
|---|---|---|---|
| 1 | 0.94x | 0.99x | 0.95x |
| 4 | 0.94x | 0.99x | 0.95x |
| 12 | 0.97x | 0.96x | 0.98x |
| 24 | 0.97x | 0.98x | 0.97x |

So the vectorized packing is worth 1-6% overall (more where the product is small or the thread count is
low). Absolute results with it: 106 GFLOP/s on 1 thread at n = 1024 (95% of the 112 GFLOP/s core peak),
and 1083 GFLOP/s on 24 threads at n = 2048 (81% of the chip's 1333).

### 14. Multithreading: thread placement, pinning and how the loops are split

The threaded kernel was at 66-78% of the chip's measured peak. To see why, `PackedGemmBlocking::profile`
records, per call, how long each thread spent packing and in the micro-kernel, and `gemm_thread_benchmark`
prints it next to the GFLOP/s. For a warm 12-thread call at n = 2048 the average thread spent about 85% of
the wall time in the micro-kernel, 7% packing and 7% waiting at barriers: the structure was not the main
loss. The big loss turned out to be where the operating system put the threads.

**Thread placement.** On a CPU with SMT the OS sometimes runs two of our threads on the two halves of one
core and leaves another core idle. Compute-bound code on the shared core runs at half speed and every
barrier waits for it. It comes and goes between runs (it depends on what else is running), which made
every earlier threaded measurement noisy:

| Measurement | Not pinned | Each thread pinned to its own core |
|---|---|---|
| matmul n = 2048, 12 threads, GFLOP/s | 597 (a bad run) to 1050 | 973-1060, stable |
| matmul n = 4096, 12 threads | 638 (a bad run) | 1075 |
| matmul n = 1024, 2 threads | 107 (both threads on one core) | 192 |
| register-only FMA loop, 12 threads, time of a ~1 ms slice | 0.50 ms, up to 0.93 ms after 100 ms idle | 0.48 ms always |

MSVC's OpenMP (version 2.0) cannot pin threads, so `src/backends/cpu/affinity.hpp` does it: the CPU
topology comes from `GetLogicalProcessorInformationEx` (Linux: sysfs and `pthread_setaffinity_np`; compiled and tested
on the GitHub Actions Ubuntu runner, correctness only, no Linux timings), each thread restricts itself to the two hardware threads of its own physical core, and a team
fills the cores in order, performance cores first on hybrid CPUs. Details that matter:

- The calling thread is thread 0 of the team. It is pinned for the call and gets its old restriction back
  at the end, so the application's own thread is never left pinned. Leaving it unpinned instead made
  things worse (237 vs 297 GFLOP/s at n = 512, 4 threads), because it lands on a worker's core.
- Only one call pins at a time, never from inside another parallel region, and only for products of at
  least 8 million multiply-adds, so concurrent callers do not all pin to core 0 and tiny calls pay nothing.
- Settings: `PackedGemmBlocking::pin` is 0 (off), 1 (core), 2 (one hardware thread) or 3 (only a
  preference, the Windows "ideal processor"). Mode 3 did not help: it still produced the 2-thread
  collision (107 GFLOP/s) in the same test.
- `affinity_test` checks that pinning restricts the thread, thread numbers wrap around the cores, the
  logical mode leaves one hardware thread, mode 3 restricts nothing, and unpinning restores the original
  mask exactly.

**Which cores.** Threads are packed onto adjacent cores (one chiplet, one L3) before using the next.
Spreading them over both chiplets looked natural and was clearly worse, because the packed A and B panels
are shared and a cache line that has to cross between chiplets is slow (GFLOP/s, pinned, 2 / 4 / 6 threads):

| n | spread over both chiplets | adjacent cores |
|---|---|---|
| 256 | 126 / 192 / 200 | 161 / 242 / 290 |
| 512 | 169 / 300 / 410 | 190 / 343 / 497 |
| 1024 | 193 / 364 / 515 | 198 / 379 / 551 |
| 2048 | 197 / 380 / 555 | 200 / 389 / 568 |

**Ways of splitting the loops.** All four are in `matmul_f32_packed_parallel`, chosen with
`PackedGemmBlocking::parallel`, all tested for correctness:

| # | Strategy | Result (pinned, 12 threads, GFLOP/s at n = 512 / 1024 / 2048 / 4096) |
|---|---|---|
| 0 | Two parallel loops per K step: pack everything, then dynamically scheduled tiles of C (an MC-row block times a group of B panels) | 612 / 868 / 967 / 1083 |
| 1 | The same work in one parallel region for the whole call (the team is woken once) | 634 / 881 / 917 / 1085 |
| 2 | One region, B packed together, each thread packs its own rows of A (static split into rows x columns of threads, like BLIS) | 767 / 888 / 639 / 1087 |
| 3 | Strategy 0's dynamic tiles, but A packed privately per tile and kept while the thread stays on the same rows | 722 / 951 / 626 / 1091 |

- Waking the team once instead of twice per step (0 vs 1) makes no measurable difference.
- Packing A privately wins at n = 512 and 1024 (+18-25% and +2-10%) and is level at n = 4096, but at
  n = 1536-2560 it loses 40% whenever a call does not directly follow another one (the numbers above are
  the median of calls separated by a checksum; back to back the same strategies reach 1075-1110, the best
  of all). The penalty is about 10 ms per call, shows up in the micro-kernel time (24-28 ms instead of
  13.5), and I did not find the cause. Allocating the private buffers from one block instead of per thread
  changed nothing, a pure FMA loop does not show it, and pinning does not remove it. So strategies 2 and 3
  are kept as options but are not the default.
- **Default: strategy 1.** Same speed as 0, robust after idle time, and the threads pin themselves when the
  team starts, without an extra thread wake-up.

**Loop order and tile size.** Inside a block the micro-kernel can sweep A panels for each B panel (order 0)
or the other way round (order 1). A packed B panel is KC x 16 floats = 64 KB at KC = 1024, more than the
32 KB L1; an A panel is KC x 6 floats = 24 KB and fits, so the A panel should stay in L1 while B streams
past (order 1; the default is derived from the L1 size). Strategy 0, pinned, 12 threads, GFLOP/s:

| Tiles per thread | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| n = 2048, order 0 | 956 | 975 | 959 | 1014 |
| n = 2048, order 1 | 618 | 707 | 1056 | 1052 |
| n = 4096, order 0 | 974 | 1070 | 1092 | 1105 |
| n = 4096, order 1 | 992 | 1065 | 1092 | 1090 |

Order 1 is 4-7% faster at n = 2048 and 25% faster for a call after idle time, but only with 16 or more
tiles per thread (with 4 or 8 it collapses, probably because the last tiles are too coarse). The default is
16 tiles per thread.

**One thread per physical core for small products.** The second hardware thread of a core adds no FMA
throughput; for small products it only adds barrier and packing overhead (n = 256: 266 GFLOP/s on 12
threads, 185 on 24). Products below 2^30 multiply-adds (about n = 1024) now use at most one thread per
core; larger ones use all hardware threads, which gains a few percent (n = 4096: 1144 on 24 threads against
1058 on 12).

**Defaults after this work (Ryzen 9 5900X):** strategy 1, loop order 1, 16 tiles per thread, pinning on
when the topology is known, MC = 240, KC = 1024, NC = 8160.

**Caveats.** All numbers are single runs on one busy desktop (antivirus and other background programs were
running), so differences below about 5% are not meaningful. The Linux pinning code passes its tests on the GitHub Ubuntu runner but has no timings, and pinning on the
i7 laptop has not been run. The cause of the private-A penalty after idle time is open.

### 15. Strides: no copies for transposed, sliced or padded operands

`matmul` used to turn every non-contiguous input into a contiguous copy before calling the kernel
(`a.contiguous()` in `src/ops/blas.cpp`), and that copy is a single-threaded strided read. The backward
pass is made of such inputs: `grad_a = grad x b^T` and `grad_b = a^T x grad`, where `grad` is often a
broadcast view with stride 0; `gemm` with transposes goes through the same path. At n = 2048 the copy of
one transposed matrix takes about as long as the multiplication itself.

The packing step already reads every element of A and B once, so it can read them in place. Everything
below takes strides: element (i, p) of A is `a[i * rsa + p * csa]`, element (p, j) of B is
`b[p * rsb + j * csb]`, and C has a row stride `ldc`.

| Operand layout | Packing path |
|---|---|
| A row-major, any leading dimension (`csa = 1`) | the 8 x 8 register transpose of section 13 |
| A transposed (`rsa = 1`) | direct 6-float copy per depth step, no transpose; depth steps outermost so each source column is read as one run |
| B row-major, any leading dimension (`csb = 1`) | 256-bit copies of section 13 |
| B transposed (`rsb = 1`) | 8 x 8 register transposes, 8 depth steps at a time |
| anything else (odd strides, stride 0) | generic scalar loops |

- The new entry points are `matmul_f32_packed_strided` and `matmul_f32_packed_parallel_strided`; the old
  contiguous ones call them with `(k, 1)`, `(n, 1)` and `ldc = n`.
- `src/ops/blas.cpp` no longer makes inputs contiguous when the tensors are on the CPU; the CPU backend
  reads `Tensor::strides()` and passes them on. Other devices still get contiguous copies.
- On a CPU without AVX2 the strided entry points copy the operands into contiguous temporaries and use the
  plain loops, so every CPU stays correct.

**Result** (public API, GFLOP/s, all threads; "copy" is the old behaviour, `contiguous()` then `matmul`):

| n | operands | copy first | in place | speedup |
|---|---|---|---|---|
| 512 | A B | 384 | 410 | 1.07x |
| 512 | A^T B | 102 | 386 | 3.80x |
| 512 | A B^T | 111 | 409 | 3.68x |
| 512 | A^T B^T | 63 | 416 | 6.61x |
| 1024 | A^T B | 217 | 766 | 3.52x |
| 1024 | A B^T | 222 | 765 | 3.44x |
| 1024 | A^T B^T | 118 | 762 | 6.49x |
| 2048 | A B | 921 | 919 | 1.00x |
| 2048 | A^T B | 364 | 920 | 2.53x |
| 2048 | A B^T | 360 | 905 | 2.51x |
| 2048 | A^T B^T | 226 | 904 | 4.00x |

With one operand transposed the product is 2.5-3.8x faster, with both 4.0-6.6x (the backward pass). In
place, a transposed product runs at the speed of a contiguous one (n = 2048: 919, 905 and 904 against
919): the transposed packers cost nothing measurable. Contiguous products are unchanged.

**Does the row stride matter?** A leading dimension that is a large power of two (n = 2048 floats =
8 KB) makes the rows of a block map to the same cache sets, which is a known problem for strided code.
Kernel-level sweep, row strides n, n + 16 and n + 1 for A, B and C (GFLOP/s):

| n | ld = n | ld = n + 16 | ld = n + 1 |
|---|---|---|---|
| 512 | 622 | 626 | 628 |
| 1024 | 945 | 942 | 940 |
| 2048 | 1093 | 1090 | 1070 |

Within 1-2%. The packing step absorbs the stride (every source row is read as a contiguous run and
written to a panel), so there is nothing to gain from padding the matrices.

**Tile layout.** The layouts were already contiguous where it matters: an A panel is `kc x 6` floats
(24 KB at KC = 1024) and a B panel `kc x 16` (64 KB), each read front to back by the micro-kernel, and
the panels sit one after the other. The sweep above shows no sensitivity to strides, the earlier sweeps
(sections 7, 11, 14) cover the tile shape, the block sizes and the loop order inside a block, and the
register-level analysis (section 12) shows the kernel at 90-95% of the FMA peak. I did not try padding
the stride between packed panels; nothing measured points to a gain from it.

**Tests.** `gemm_pack_test` compares every packer with the scalar reference for the row-major, padded,
transposed and odd-stride layouts (guard words after each panel). `gemm_packed_test` runs 5 layouts for A
times 5 for B (row-major, padded, transposed, transposed and padded, interleaved) on 8 shapes with both
kernels and default and tiny block sizes, writes C with a larger row stride, and checks the unused
columns and the memory after C are untouched; four large products also cover the pinned path.
`matmul_test` multiplies large transposed, sliced and transposed-and-sliced views through the public API
and checks the gradient of `sum(a x b)` exactly (the backward pass with the stride-0 gradient).

**Not covered.** The GPU backends still receive contiguous copies. `gemm` applies alpha and beta as
separate elementwise passes after the product (fusing them into the kernel is on the open list, probably to be done another time). `contiguous()` itself is still single-threaded for the
operations that need it.

### 16. Packing overlapped with compute (strategy 4) and KC = 768

The profile of strategy 1 (`gemm_thread_benchmark`, `profile` on) at n = 4096 showed where the last
14% goes: each thread spent about 5% of the call packing and 4-7% waiting at barriers. Packing is slow
under threads: 12 threads together packed at about 1.7 GB/s each, against 77 (A) and 14-30 (B) GB/s for
one thread alone, because every thread packs at the same moment and they share the DRAM bandwidth.
The micro-kernel, meanwhile, hardly touches memory. So the two should run at the same time.

**Strategy 4** (`PackedGemmBlocking::parallel = 4`, the new default) keeps two copies of the packed A and B.
Per (jc, pc) step there is one work list: the tiles of this step, with the pack items of the next step
spread evenly among them (item t is a pack item when `floor((t+1) P / N) > floor(t P / N)`), handed out
dynamically, the packs writing into the other buffer. Only the first step is packed on its own. One
barrier per step remains (the buffer of step s is repacked only during step s + 2). Cost: twice the
packing memory (at n = 4096 and KC = 768, 2 x 24 MB).

Each strategy measured in its own process (strategies in one process share the scratch buffers, and the
buffer size turned out to matter, see below), GFLOP/s, median / back to back / after 5 ms idle:

| n | threads | strategy 1, KC 1024 (old default) | strategy 4, KC 1024 | strategy 4, KC 768 (new default) |
|---|---|---|---|---|
| 1536 | 12 | 1013 / 1013 / 673 | 1032 / 1054 / 993 | 1067 / 1069 / 1055 |
| 2048 | 12 | 1056 / 1059 / 1053 | 695 / 1056 / 679 | 1084 / 1077 / 1057 |
| 2560 | 12 | 1066 / 1067 / 1011 | 963 / 1106 / 927 | 960 / 1100 / 820 |
| 3072 | 12 | 950 / 1080 / 864 | 1015 / 1135 / 939 | 986 / 1108 / 928 |
| 4096 | 12 | 1092 / 1111 / 1046 | 1161 / 1168 / 1090 | 1141 / 1133 / 1135 |
| 1536 | 24 | 1031 / 1048 / 999 | 935 / 1101 / 830 | 1089 / 1100 / 1084 |
| 2048 | 24 | 1080 / 1104 / 816 | 849 / 1105 / 664 | 1134 / 1119 / 1130 |
| 2560 | 24 | 1077 / 1096 / 860 | 891 / 1128 / 885 | 1101 / 1147 / 834 |
| 3072 | 24 | 983 / 1152 / 913 | 1052 / 1191 / 971 | 1042 / 1181 / 964 |
| 4096 | 24 | 1151 / 1165 / 1113 | 1213 / 1228 / 1152 | 1204 / 1213 / 1200 |

(One of two repeats shown; the second agreed within a few percent except where noted below.) On 6
threads, one chiplet, strategy 4 is 2-5% ahead of strategy 1 at every size from 512 to 4096 in all three
modes. Barrier wait dropped from 4-7% to 1.4-3% of the call.

**Why KC = 768.** With KC = 1024, strategy 4 has a large slowdown at exactly n = 2048 in the median and
after-idle modes (about 680 instead of 1060 GFLOP/s), while back to back it is fine. The same slowdown
appears for strategy 1 when its buffers are made as large (KC = 1536 or 2048, or after strategy 4 grew the
shared buffers in the same process), so it depends on how much packed memory is in use, not on the
overlap. A KC sweep at n = 2048 (12 threads): strategy 4 is clean at KC 512 and 768 and slow at 256,
384 and 1024; strategy 1 is clean at 512-1024 and slow at 256 and 1536. KC = 768 is clean at 2048 and
within 0-2% of KC 1024 elsewhere back to back. Single-thread speed does not change (105.6-105.9 GFLOP/s).

**The slowdown after idle time, what is known.** It is the same effect as the "10 ms penalty" of strategies
2 and 3 (section 14). Facts gathered this time: it only appears in calls that follow other work or idle
time, never back to back; it shows as extra micro-kernel time (about 8-12 ms per thread at n = 2048-4096),
not packing time; it **never appears on 6 threads** (one chiplet) at any size; it depends on the size of the
packing buffers; at n = 2560 and 3072 it still hits every strategy now and then (strategy 4, KC 768, 12
threads, n = 2560: median 960 and 822 in two repeats, against 1100 back to back). A first guess, that the
cross-chiplet traffic on shared packed panels causes it, was tested with strategy 5 (no panel is shared
across chiplets) and is wrong: strategy 5 shows it as well. Clock ramp-up was already excluded (section
14). Still untested: power states of the Infinity Fabric and memory controller after idle (BIOS settings
"DF C-states" / "APBDIS"), which would fit "only with both chiplets and only after idle".

### 17. Splitting the team by chiplet (strategy 5)

Section 14 found that putting threads on adjacent cores (one chiplet) before spreading them is up to 45%
faster for small products, because the packed panels are shared and a cache line that crosses chiplets is
slow. With 12 or 24 threads both chiplets are in use, so the same loss returns. Strategy 5 splits the team
by L3 cache: `core_cache_domain()` (new in `affinity.hpp`, from `GetLogicalProcessorInformationEx(
RelationCache)` on Windows and `cache/index3/id` on Linux; `affinity_test` checks it) gives each pinned
thread its chiplet. Each group computes the columns of C in proportion to its thread count, packs the B
panels of those columns and its **own copy of A**, and takes work only from its own lists (if the runtime
ever gives fewer threads than asked, every thread works through every list, so the result stays correct).
It runs on the double-buffered scheme of strategy 4.

Strategy 4 against 5, 12 threads, two separate processes each, median / back to back / after idle:

| n | strategy 4 | strategy 5 | |
|---|---|---|---|
| 256 | 277 / 274 / 212 | 309 / 307 / 209 | +11% |
| 384 | 499 / 493 / 430 | 618 / 613 / 452 | +24% |
| 512 | 678 / 677 / 574 | 812 / 810 / 641 | +20% |
| 768 | 869 / 866 / 728 | 1025 / 1024 / 560 | +18% (worse after idle) |
| 1024 | 918 / 916 / 904 | 801 / 1054 / 574, 628 / 885 / 529 | erratic |
| 4096 | 1161 / 1169 / 1083 | 1137 / 1168 / 1102 | equal; A is packed twice |

So the default strategy 4 splits by chiplet below 2^29 multiply-adds (about n = 810, `max_macs_for_groups`)
and shares above. Strategy 5 forces the split at every size.

### 18. Packing several B panels per pass

Section 13 left one idea for `pack_b`: one panel at a time reads a single 64-byte line from each of the kc
rows, with a stride the hardware prefetcher does not follow. `pack_b_panels_avx2` copies `run` panels per
pass over the rows (two rows per iteration), so each row is read as a run of `run` lines; the threaded
driver's B work items are runs of 4 panels. `gemm_pack_benchmark`, one thread, GB/s:

| Block (ld) | 1 panel per pass | 2 | 4 (default) | 8 |
|---|---|---|---|---|
| B 1024 x 4096 (4096) | 17.8 | 23.4 | 34.0 | 37.0 |
| B 1024 x 4080 (8192) | 17.1 | 24.2 | 26.4 | 32.4 |
| B 1024 x 2048 (2048) | 28.5 | 40.7 | 45.5 | 45.7 |
| B 256 x 4080 (8192) | 32.9 | 48.6 | 52.5 | 48.2 |
| B 1024 x 256 (2048) | 27.1 | 43.7 | 46.2 | 45.9 |

1.5-1.9x faster on its own. In the threaded matmul it changes nothing measurable (pack time at n = 4096,
12 threads, stayed at 5.4-5.8 ms per thread): with all cores packing at once the limit is DRAM bandwidth,
and strategy 4 already hides packing behind the arithmetic. Kept, since it costs nothing and helps one
thread. `gemm_pack_test` checks runs of 1, 2, 3, 4 and 8 panels against the scalar reference.

### 19. Measured and not adopted

| Approach | Result |
|---|---|
| Tiles per thread 8 / 16 / 24 / 32 with strategy 4 (n = 1536-4096, 12 and 24 threads) | 16 best or tied everywhere; 8 collapses in the median mode at 1536-2048; unchanged |
| Masked edge tiles for sizes that are not multiples of 6 x 16 | Not written: the current scratch-tile edges cost at most 2% (12 threads: n = 1000 931, 1024 918, 1025 899, 2047 1076, 2048 1076, 2050 1071; 1 thread: all within 2%) |
| Strategy 5 at every size | Erratic at n = 1024, equal at 4096 (A is packed twice); used only for small products |

Also fixed: strategy 3 wrote C with row stride n instead of `ldc` (wrong results for a padded C with that
non-default strategy). `gemm_packed_test` now runs the large strided case with every strategy.

### 20. Manual unrolling, register use and where the broadcasts go

Section 12 found the old unrolled kernel (shape 1) spilling and 25% slower, without knowing why. This round
wrote the variants again, read the generated code of each (`dumpbin /disasm` of `gemm_packed.obj`) and
measured the micro-kernel on its own with a new tool, `gemm_kernel_benchmark` (one pinned thread, packed
panels already in cache, no packing, the variants alternating over 9 rounds).

**The baseline loop** (shape 0), per depth step: `vmovups ymm1/ymm2` (the B row), then six times
`vbroadcastss ymm0, [rdx+4r]` followed by its two `vfmadd231ps`, then 2 `lea`, `sub`, `jne`: 23
instructions, 15 of the 16 ymm registers (accumulators ymm3-ymm14), no stack traffic. A step needs 6 cycles
on the two FMA pipes and issues in about 4, so it is FMA bound.

**Variants** (`PackedGemmBlocking::shape`, all tested by `gemm_packed_test`):

| shape | Variant | What the assembly shows |
|---|---|---|
| 4, 5 | Unrolled x2 / x4, the three temporaries (`b0`, `b1`, `a`) declared once and reused | MSVC hoists the B loads and broadcasts of *all* unrolled steps to the top of the body (e.g. 4 B vectors and 6 broadcasts live at once), runs out of registers and keeps 5-7 accumulators on the stack: 28-31 ymm loads/stores to `[rbp+...]` inside the loop |
| 6 | Broadcasts pipelined one row ahead: the 16th register holds the next row's A value while the current row's FMAs run (`x0`/`x1` alternate; the last step of a panel is peeled so nothing past the panel is read) | Exactly as written, 16 registers, no spills |
| 7 | Pipelined, unrolled x2, no fence | Hoisting again: 22 spill moves |
| 8, 9 | Unrolled x2 / x4 with a **compiler fence** between steps (`_ReadWriteBarrier()` on MSVC, an empty `asm volatile` memory clobber elsewhere: no instruction, but no load moves across it) | No spills; constant displacements (`[r8+40h]`, `[rdx+1Ch]`...), one pointer update and one branch per 2 / 4 steps |
| 10, 11 | Pipelined, unrolled x2 / x4, fenced | No spills; ymm0/ymm1 alternate as broadcast registers, ymm2/3 hold B, ymm4-15 accumulate: all 16 in use |

**Kernel alone** (GFLOP/s, median of 9; % of the measured 112 GFLOP/s core peak):

| Kernel | L1: kc 256, 1 B panel | L2: kc 768, 8 B panels (the real loop order) |
|---|---|---|
| 6x16 baseline (shape 0) | 109.1 (97.4%) | 109.4 (97.7%) |
| unrolled x2, fenced (8) | 109.0 (97.3%) | 109.7 (98.0%) |
| unrolled x4, fenced (9) | 108.5 (96.9%) | 109.7 (97.9%) |
| pipelined (6) | 107.7 (96.1%) | 108.0 (96.4%) |
| pipelined x2, fenced (10) | 108.4 (96.7%) | 108.7 (97.0%) |
| pipelined x4, fenced (11) | 107.2 (95.7%) | 108.6 (97.0%) |
| 4x24 (2) | 108.7 (97.1%) | 106.4 (95.0%) |
| 12x8 (3) | 99.3 (88.6%) | 97.2 (86.8%) |
| old unrolled x4 (1) | 82.5 (73.7%) | 83.2 (74.3%) |
| unrolled x2 / x4, no fence (4, 5) | 71.5 / 82.9 | 72.1 / 83.5 |
| pipelined x2, no fence (7) | 70.9 (63.3%) | 70.6 (63.1%) |

End to end (`gemm_tune_benchmark shape=0,8,9,10,11`, interleaved re-timing), 1 thread: shape 0 best or tied
at n = 512, 1024, 2048 and 4096; fenced unrolling 0-2% behind, pipelined x4 always last (1-5% behind). 12
and 24 threads: all within noise of shape 0 at every size. (The threaded part of that first run was affected
by a bug in `gemm_tune_benchmark`, see section 21; rerun with the fix, prefetch off for all, GFLOP/s:
n = 1024 / 2048 / 4096 on 12 threads, shape 0: 918 / 1084 / 1147, shapes 8-10: 921-923 / 1065-1071 /
1137-1141; on 24 threads shape 0: 958 / 1123 / 1204, shapes 8-10: 932-953 / 1119-1133 / 1185-1195. The
conclusion is the same.)

Findings:

- **The kernel itself is finished.** At 97.7% of the FMA peak there are about 2.5 points left, and they
  are the per-call costs (zeroing 12 accumulators, saving xmm6-xmm15 for the Windows ABI, the C update),
  which unrolling does not touch. Single-thread matmul sits at 94-95%; the difference to the kernel's 97.7%
  is packing and the C traffic, not the loop body.
- **Unrolling helps only if the compiler can be kept from spilling, and then it changes nothing.** The
  loop overhead it removes (2 `lea`, `sub`, `jne` per step) already issues in the spare front-end slots of an
  FMA-bound loop. Without the fence MSVC's scheduler turns every unrolled version into a spilling one, which
  explains the 25% loss of section 11.
- **Where the broadcast goes does not matter on Zen 3.** Broadcasting right before the two FMAs that use it
  (baseline) is as fast as broadcasting a row ahead into a second register; the out-of-order core already
  hides the broadcast's load latency, and the pipelined form costs about 1% (one more instruction
  dependency chain through the alternating registers and the peeled last step).
- The default stays shape 0. The variants remain selectable for other CPUs (a core with fewer rename
  registers or a smaller scheduler might prefer the pipelined form), and `gemm_kernel_benchmark` measures
  them on any machine in a few seconds.

### 21. Prefetching, second round

Section 7 tried two software prefetches (the C tile, and A/B 8 steps ahead) and found nothing. Since then the
picture has changed: the kernel alone runs at 97.7% of peak (section 20) but end to end it is 94-95% on one
thread and lower with 12-24 threads, so the losses are in memory traffic around the kernel. This round tried
prefetching at each place that touches memory.

**A benchmark bug, fixed first.** `gemm_tune_benchmark` built each swept variant from a blank
`PackedGemmBlocking{}` instead of the current defaults, so the variants ran with strategy 0, no pinning and
loop order 0, while the "default" row it compares them with used the real defaults. Single-thread sweeps
were barely affected; threaded variants looked 6-34% slower than the default (seen in section 20's first
run). Variants now start from the defaults and change only the swept settings. Earlier sections' block-size
sweeps (section 7) ran before strategy 1 and pinning existed, when the blank configuration *was* the
default, so they are not affected.

**The modes** (`PackedGemmBlocking::prefetch` inside the 6x16 kernel, `prefetch_block` between tiles in
`compute_block`; `gemm_packed_test` runs all 35 combinations in both loop orders):

| Setting | What is prefetched |
|---|---|
| prefetch 1 / 2 | The C tile (T0) before the loop / also A and B 8 steps ahead (section 7) |
| prefetch 3 | B 32 steps (2 KB) ahead into L1 |
| prefetch 4 | B 128 steps (8 KB) ahead into L2 (for B panels that stream from L3 under threads) |
| prefetch 5 | The C tile with **write intent** (`PREFETCHW`) before the depth loop |
| prefetch 6 | 5 and 3 together |
| prefetch_block 1 / 2 | The *next* tile's C, before computing this one (T0 / write intent) |
| prefetch_block 3 | The next A panel (18 KB at KC = 768), spread over the tiles of the current row into L2, as BLIS does |
| prefetch_block 4 | 2 and 3 together |
| `pack_b(..., ahead)` | In the B packer, the source rows `ahead` rows down (runs of 4 lines per row) |

**Results.** Kernel modes, one sweep (GFLOP/s relative to no prefetch): on 1 thread every mode is within
0.4% at n = 1024-4096. On 12 and 24 threads, modes 5 and 6 are first or second in all six size/thread
cells (+1-2%); B prefetch alone (3, 4) and the old mode 2 are level. Between-tile modes: the next-A-panel
prefetch (3, 4) gains nothing on 1 thread and twice fell into the after-idle slowdown (0.58x and 0.80x at
n = 2048), plausibly by pushing 18 KB per row of extra lines into L2; dropped.

Deciding between write-intent prefetch of the current tile (prefetch 5), of the next tile (prefetch_block 2)
and both, two repeats at n = 256-4096 on 1, 6, 12 and 24 threads; prefetch 5 against no prefetch:

| Threads | n = 256-512 | n = 1024 | n = 2048-4096 |
|---|---|---|---|
| 1 | +0.1 to +0.6% | -0.0 / +0.2% | +0.1 to +0.5% |
| 6 | -0.6 to +0.9% | +1.2 / +1.5% | +0.1 to +0.5% |
| 12 | -1.0 to +0.0% | +3.5 / +3.4% | +0.2 to +1.2% |
| 24 | -0.6 to -0.1% | +0.9 / +2.3% | -0.7 to +0.3% |

prefetch_block 2 costs 1-4% at n = 256-512 (the extra loop and prefetches per tile) and is mixed above; both
together are below prefetch 5 alone. **prefetch 5 is the new default.** Why it helps: the C update after the
depth loop (load, add, store 6 rows) cannot start until the loop is retired, so a C line that is not in L1
stalls the end of every tile; with threads the line was often last written by another core (the previous K
step's tile), and a plain load then a store needs two transactions to get it exclusive. One `PREFETCHW` per
line at the start of the tile, about 1.3 us before the update, brings it in owned. Why plain T0 prefetch of
C (mode 1) did not do the same in section 7: it brings the line shared, so the store still has to upgrade
it, and at that time pinning was off and the noise was larger than the effect.

The B packer with software prefetch (`gemm_pack_benchmark`, run 4, GB/s): `ahead` = 2 rows is +3-7% on
every B block (B 1024 x 4096: 34.8 -> 37.2; B 1024 x 2048: 44.9 -> 47.2), 4 mixed, 8-32 slower (down to
-10%, the prefetches crowd the miss buffers). Packing B is about 3% of a single-thread call and DRAM-bound
with threads, so +5% of it is under 0.2% end to end: left as an option, off.

**End to end with the new default** (`matmul_openblas_benchmark`, packed GFLOP/s, before -> after): n = 1024
on 12 / 24 threads 914 -> 942 / 951 -> 974; n = 2048 on 24 threads 1136 -> 1141; n = 4096 on 12 / 24
threads 1139 -> 1161 / 1202 -> 1213; one thread unchanged (106).

## Where we stand against OpenBLAS (Ryzen 9 5900X, latest run, default settings)

`matmul_openblas_benchmark 128 256 512 1024 2048` and `matmul_openblas_benchmark 4096`, OpenBLAS 0.3.34,
after section 21 (prefetch 5 on), packed vs OpenBLAS:

| n | 1 | 2 | 4 | 8 | 12 | 16 | 24 threads |
|---|---|---|---|---|---|---|---|
| 128 | 1.31x | 0.97x | 1.12x | 1.71x | 1.78x | 1.72x | 1.72x |
| 256 | 1.16x | 1.03x | 1.09x | 1.25x | 1.61x | 1.93x | 2.50x |
| 512 | 1.08x | 1.10x | 1.02x | 1.26x | 1.35x | 1.57x | 1.53x |
| 1024 | 1.04x | 1.03x | 1.04x | 1.06x | 1.08x | 2.29x | 1.45x |
| 2048 | 1.03x | 1.01x | 1.01x | 0.99x | 1.35x | 1.41x | 1.39x |
| 4096 | 1.01x | 1.02x | 1.02x | 1.00x | 1.22x | 1.26x | 1.05x |

Packed GFLOP/s in that run:

| n | 1 thread | 8 | 12 | 24 |
|---|---|---|---|---|
| 512 | 104 | 629 | 806 | 806 |
| 1024 | 105 | 706 | 942 | 974 |
| 2048 | 106 | 760 | 1050 | 1141 |
| 4096 | 106 | 800 | 1161 | 1213 |

At or above OpenBLAS in 40 of 42 combinations; the two below are 0.97x (n = 128, 2 threads) and 0.99x
(n = 2048, 8 threads), within noise. Best: 1213 GFLOP/s at n = 4096 on 24 threads (91% of the chip's measured
1333) and 1141 at n = 2048 (86%).

The earlier run after sections 16-18, packed vs OpenBLAS:

| n | 1 | 2 | 4 | 8 | 12 | 16 | 24 threads |
|---|---|---|---|---|---|---|---|
| 128 | 1.32x | 0.96x | 1.15x | 1.97x | 1.74x | 1.78x | 1.73x |
| 256 | 1.16x | 1.04x | 1.11x | 1.39x | 1.73x | 1.90x | 2.67x |
| 512 | 1.07x | 1.10x | 1.07x | 1.26x | 1.37x | 1.59x | 1.85x |
| 1024 | 1.04x | 1.04x | 1.04x | 1.05x | 1.03x | 1.74x | 1.02x |
| 2048 | 1.03x | 1.01x | 1.01x | 0.98x | 1.44x | 1.40x | 1.81x |
| 4096 | 1.01x | 1.01x | 1.01x | 1.00x | 1.20x | 1.28x | 1.05x |

At or above OpenBLAS in 40 of 42 combinations; the two below, 0.96x (n = 128, 2 threads) and 0.98x
(n = 2048, 8 threads), are within noise. Before sections 16-18 it was 26 of 30 up to n = 2048, with n = 512
at 4 threads (0.96x) and n = 2048 at 4 and 8 threads (0.97x, 0.95x) behind. As before, OpenBLAS's 12-24
thread medians vary a lot between runs (n = 2048, 12 threads: 15.7 to 34.4 ms per call within this run),
so the large ratios there say more about OpenBLAS's instability than about our speed.

A repeat after section 20 (same defaults, new build) gave 41 of 42 at or above OpenBLAS (only n = 128 on
2 threads, 0.96x); n = 2048 on 8 threads came out at 1.00x. Packed GFLOP/s in that run: 1066 / 1136 at
n = 2048 on 12 / 24 threads, 1139 / 1202 at n = 4096; one thread 105.8 (n = 2048) and 106.3 (n = 4096).

Absolute GFLOP/s of the packed kernel in that run:

| n | 1 thread | 8 | 12 | 24 |
|---|---|---|---|---|
| 512 | 104 | 639 | 807 | 812 |
| 1024 | 105 | 697 | 925 | 949 |
| 2048 | 106 | 750 | 1035 | 1131 |
| 4096 | 106 | 792 | 1145 | 1184 |

Best: 1131 GFLOP/s at n = 2048 on 24 threads (85% of the chip's measured 1333, before: 1094) and 1184 at
n = 4096 (89%, before: about 1142; up to 1228 back to back in the thread benchmark, 92%). One thread:
106 GFLOP/s (95% of the core's 112).

## Open items

| Item | Why |
|---|---|
| Threaded efficiency (85-89% of the 1333 GFLOP/s chip peak at n = 2048-4096; level with OpenBLAS at n = 2048 on 4-8 threads) | Overlapped packing and multi-panel `pack_b` are done (sections 16, 18). Left: the micro-kernel runs at about 94% of peak under 12 threads against 97% under 6, and 1.4-3% barrier wait remains |
| The slowdown of calls after idle time (sections 14, 16) | Now the largest loss: n = 2560-3072 at 12-24 threads lose up to 20% in the median and after-idle modes against back to back. Only with both chiplets, grows with the packing buffers, not cross-chiplet panel sharing, not clock ramp-up. Next test: BIOS "DF C-states" off (the user's decision, it changes idle power) |
| Strategy 5 at n = 1024 | Much faster back to back (1054-1088) but erratic in the other modes; if the slowdown above is fixed, the chiplet split may pay off at larger sizes too |
| Linux timings and the i7 laptop | The Linux pinning code (sysfs topology, `pthread_setaffinity_np`) compiles and passes `affinity_test` and `gemm_packed_test` on the GitHub Ubuntu runner, but nothing has been timed on Linux; the i7 (4 cores, 8 threads, 256 KB L2) has no threaded measurements, and its small-L2 defaults are untested |
| Quiet-machine repeat runs | All numbers here are single runs; OpenBLAS at 24 threads varies by up to 40% |
| Fuse alpha and beta into the kernel (probably another time) | `gemm` applies alpha and beta as separate elementwise passes over C (`mul`, `scal`, `add`, each a full read and write plus a temporary). Fusing them into the micro-kernel's C update (scale the accumulators by alpha, add beta times the old C when storing) would remove those passes and temporaries. Estimated cost of the passes at n = 2048 (not measured): about 4 ms on top of a 16 ms product with both alpha and beta, about 1 ms with alpha only, and several times the product when k is small. It only matters for `gemm` calls with alpha other than 1 or beta other than 0; `matmul` and the backward pass are unaffected. Decided to postpone until such calls are on a hot path; measure `gemm` against `matmul` first |

## Files

| File | Role |
|---|---|
| `src/backends/cpu/ops/blas/gemm_packed.hpp`, `.cpp` | The kernel, packing, threading, CPU detection and block sizes |
| `src/backends/cpu/backend.cpp` | Selects the packed kernel for `matmul` |
| `src/backends/cpu/ops/blas/gemm_pack.hpp` | Scalar and AVX2 packing, including the strided layouts |
| `src/ops/blas.cpp`, `src/backends/cpu/backend.cpp` | `matmul` passes strided CPU views on without a copy |
| `tests/ops/gemm_packed_test.cpp` | Correctness tests |
| `tests/ops/gemm_pack_test.cpp` | Packers against the scalar reference, with guard words |
| `benchmarks/blas/gemm_pack_benchmark.cpp` | Packing speed, scalar vs AVX2 |
| `benchmarks/blas/gemm_kernel_benchmark.cpp` | The micro-kernel variants alone (one thread, panels in cache), % of the core peak |
| `benchmarks/blas/matmul_openblas_benchmark.cpp` | Comparison against OpenBLAS |
| `benchmarks/blas/gemm_tune_benchmark.cpp` | Block-size sweep |
| `benchmarks/blas/gemm_stride_benchmark.cpp` | Transposed operands copied first vs read in place; row-stride sweep |
| `benchmarks/blas/fma_peak_benchmark.cpp` | Register-only FMA throughput: the real per-core and per-chip peak; `ramp` shows the effect of thread placement |
| `benchmarks/blas/gemm_thread_benchmark.cpp` | Compares parallelization strategies, pinning, loop order and tile size, with the time split per thread |
| `src/backends/cpu/affinity.hpp`, `.cpp` | CPU topology, L3 cache domains (`core_cache_domain`) and thread pinning |
| `src/backends/cpu/threads.hpp` | Adds parallel-region, worksharing-loop and barrier macros, thread index, wall clock |
| `tests/core/affinity_test.cpp` | Pinning restricts, wraps, and restores the original affinity |
| `CMakeLists.txt`, `docs/BENCHMARKING.md` | Build options and how to run the comparison |

## Design references

BLIS's Haswell sgemm configuration (6x16 tile, MC 168, KC 256, NC 4080), the `salykova/sgemm.c` write-up
(16x6 tile, zero-padded packing, masked edge stores, threading over the two loops around the
micro-kernel) and the Hadjis AVX2 tutorial.
