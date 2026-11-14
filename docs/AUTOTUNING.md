# Auto-tuning

Two things are tuned **for a particular algorithm**: the settings of kernels that have some (tile
sizes, which kernel variant to use), and the **device** each op runs on when it is called with
`Execution::Auto` (the default of every op). The user writes the algorithm as a function, the tuner measures the calls that function
makes on this machine, and the best settings are kept in a text file so later runs reuse them.

```cpp
#include "mensurflow/mensurflow.hpp"
namespace mf = mensurflow;

mf::Algorithm network([](const std::vector<mf::Tensor>& in) {
    mf::Tensor h = mf::matmul(in[0], in[1], mf::Execution::GPU);
    return std::vector<mf::Tensor>{mf::matmul(h, in[2], mf::Execution::GPU)};
});

mf::Tuner tuner("mensurflow_tuning.txt");      // reads earlier results, if any
mf::TuneReport report = tuner.tune(network, {x, w1, w2});
auto out = network.run({x, w1, w2}, tuner);    // runs with the stored settings
```

`examples/tune_mlp.cpp` is a complete program (`build/tune_mlp [batch] [cpu]`). The observability
dashboard has an **Autotune** section on every op page that tunes the op for a device and size and
shows the input parameters, the tuned settings against the defaults, and every candidate's time
([observability/README.md](../observability/README.md)).

## Modes

| mode | status | what it tunes |
|---|---|---|
| offline, tested: `Tuner::tune(algorithm, inputs)` | **done** | every tunable call the algorithm makes on exactly these inputs, and the device of every `Execution::Auto` call; then the whole algorithm is timed with the default and the tuned results, and the new results are dropped if tuning made it slower |
| offline, untested: shape ranges | planned | the same calls at a representative size of every power-of-two size class in a range, stored as size-class results that cover the shapes between |
| online | planned | during real runs, calls without a stored result try the candidates one call at a time and store the fastest |

## What `tune()` does

1. Runs the algorithm once on the given inputs and records every call of a tunable kernel (its
   arguments, device and shapes). The run is real, so ops that read values partway through and
   branches that depend on data behave as they will later.
2. Keeps each GPU busy for one second, because the GPU raises its clock only under sustained load.
3. For each distinct call, and only if no result is stored for it yet (unless `retune`):
   - runs the call with the kernel's defaults and keeps that result as the reference;
   - runs each candidate configuration once and rejects any that disagrees with the reference
     (more than 1e-3 of the largest magnitude);
   - times the default and every accepted candidate: the median of samples of at least 1 ms each,
     taking at least `min_time_s` and at least `min_samples` samples;
   - stores the fastest, which can be the default.
4. For every op called with `Execution::Auto`, chooses the device: each device that has a kernel for
   the op is timed through the normal dispatch, **including copying the inputs there and the result
   back** (the data stays where it lives). A device's tunable kernel settings are tuned first, so
   every device is compared at its best, and its result must agree with the CPU's. The fastest is
   stored. Without a stored choice, `Execution::Auto` runs on the CPU, as before.
5. Times the whole algorithm with and without the results, and saves the file. If the tuned run was
   slower, everything this tune added is dropped.

Each op chooses its device on its own, and its result always comes back to where the data lives.
A chain of ops therefore pays the copies once per op; keeping the data on the GPU across several ops
would need a plan for the whole algorithm (not done yet). Device choice for one op is keyed on the op,
the dtype, the device the data is on, and the input shapes.

Calls are keyed on the kernel's own key (for matmul: m, n, k) plus the thread count on the CPU, and
on the machine. A result is applied only to exactly that call, on that machine, and for that kernel
version.

## The results file

Plain text, one result per line, grouped under the machine it was measured on. It can be edited or
trimmed by hand; lines starting with `#` are comments.

```
machine Apple M3 Pro | Apple M3 Pro
matmul cpu v1 m=128 n=4096 k=1024 threads=11 -> rows=16 depth=64 cols=1024 | 8.4586 ms (default 10.4356 ms) tested
matmul metal v1 m=256 n=4096 k=1024 -> kernel=2 bm=0 bn=0 bk=0 tm=0 tn=0 sg_m=0 sg_n=0 | 0.7838 ms (default 0.8820 ms) tested
```

`<op> <device> v<version> <key> -> <settings> | <best> ms (default <ms> ms) <mode>`. Settings a
kernel does not use are 0. A malformed line is reported with its line number when the file is read.

Device choices have their own lines (times include the copies):

```
place matmul f32 cpu 512x512 512x512 -> gpu | 0.7584 ms (cpu 1.5547 ms, gpu 0.7584 ms) tested
place add f32 cpu 65536 65536 -> cpu | 0.0307 ms (cpu 0.0307 ms, gpu 0.2800 ms) tested
```

## Tunable kernels so far

| kernel | settings | candidates |
|---|---|---|
| matmul, Metal | `kernel` (0 tiled, 1 simdgroup, 2 MPS), tile `bm` x `bn` x `bk`, `tm` x `tn` per thread (tiled), `sg_m` x `sg_n` simdgroups (simdgroup) | MPS, simdgroup tiles with 1 to 8 accumulator blocks per simdgroup and 64 to 512 threads, three tiled shapes; tiles more than twice the matrix are skipped (up to 84 per call) |
| matmul, CPU without AVX2 | `rows`, `depth`, `cols` of `MatmulBlocking` | rows 0 (whole band) / 4 / 16 / 64, depth 64 to 512, cols 128 to 1024, cut off at the matrix's own size (up to 64) |

The AVX2 packed CPU kernel is not tunable yet (its `PackedGemmBlocking` would be next).

On an M3 Pro, `tune_mlp` (1024 -> 4096 -> 1024 -> 512) ran 1.12x faster on Metal (MPS won the two
large products, a simdgroup layout the small one) and 1.30x faster on the CPU after tuning.

## Making a kernel tunable

In the backend's `register_kernels`, next to `register_kernel`, register a `TunableKernel`
(`src/core/dispatcher.hpp`):

- `name` and `version`: the name in the file; raise the version whenever the kernel changes, so
  results measured on the old kernel are ignored;
- `key_names` and `key(args)`: what a call is identified by;
- `params` (at most 8), `defaults()` and `candidates(args)`: the settings, what the kernel does
  without tuning, and the valid configurations worth trying for this call (empty: nothing to tune);
- in the kernel itself, read `args.tuning` (`size == 0` means use the defaults; otherwise the values
  are in `params` order) and reject invalid values with `std::invalid_argument`, since the file can
  be edited by hand.

dispatch() only looks for tunable kernels while a tuner is active (`Algorithm::run(inputs, tuner)`
or `Tuner::tune`), so calls outside an algorithm cost nothing extra.
