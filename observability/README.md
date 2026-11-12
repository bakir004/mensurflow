# Observability

A dashboard for mensurflow's kernels: run any public op on any device, sweep sizes, see a trace of every kernel
call, and keep a history of the results. Everything lives in this folder; the library itself only has the
tracing macro.

## What is in the library

- `include/mensurflow/trace.hpp`: the `MENSURFLOW_TRACE(...)` macro and a hook pointer. With
  `MENSURFLOW_ENABLE_TRACING` defined, the macro opens a span that ends with the enclosing scope and passes
  one event (op name, device, dtype, input and output shapes, start and end time) to the hook. Without it the
  macro expands to nothing and its arguments are not evaluated.
- One-line `MENSURFLOW_TRACE(...)` spans at the points where a call spends its time:
  - `detail::dispatch()` (`src/core/dispatcher.cpp`): one span for the dispatch, named after the op, and one
    named `kernel` around the backend kernel call. Every public op in `src/ops` goes through `dispatch()`.
  - `src/core/tensor.cpp`: `Tensor::empty` (allocation), `Tensor::to` (copy to another device),
    `Tensor::copy_to_host`, `Tensor::from_host`, `Tensor::copy_from` and `Tensor::contiguous`.

  Ops built on other kernels (for example `gemm` using the matmul kernel, or autograd's backward ops) show up
  under the kernels they dispatch. `split` returns views and dispatches nothing, so it has no span.

The library only creates events. It never stores, prints or sends them. To trace an op-level function by
name, add `MENSURFLOW_TRACE(__func__, device, inputs, outputs);` at its start.

## What is in this folder

| file | what it does |
|---|---|
| `CMakeLists.txt` | the `MENSURFLOW_ENABLE_TRACING` option and the `observe` target |
| `trace_collector.hpp` | header-only collector: sets the hook, stores events, writes Chrome trace JSON |
| `observe.cpp` | the runner: a table of every public op with input builders, timing and tracing; prints JSON |
| `serve.py` | standard-library HTTP server: serves the page and runs `observe` for it |
| `index.html` | the dashboard (no external scripts) |
| `results/history.jsonl` | every run made from the dashboard (created on first run, not committed) |

## Run it

```sh
cmake -S . -B build -DMENSURFLOW_ENABLE_TRACING=ON      # add -DOpenMP_ROOT=/opt/homebrew/opt/libomp on macOS
cmake --build build --target observe
python3 observability/serve.py                          # or --bin <path to observe> --port 8765
# open http://localhost:8765
```

Tracing is off by default. Without it the dashboard still times everything, but the trace columns and the
Trace tab are empty. Turning it on adds about one atomic load per op call while no collector is attached.

## Tabs

- **Kernels**: every op × every device at one size (small, medium or large), with median time and rate (GB/s
  of inputs + output, or GFLOP/s for matrix products). The fastest device per op is marked, and cells without a
  registered kernel are greyed out. Support is found by calling each op once at a tiny size on each device
  when the page loads. *Run all* goes through the whole matrix (or the filtered rows).
- **Sweep**: one op, chosen devices and sizes. It shows rate and time charts, p10/median/p90, and the kernels
  the call dispatched (from the trace). *Execution* can differ from where the data is: data on `cpu:0` with
  `gpu` execution measures the copy-to-GPU-and-back path.
- **Call**: where the time of one call goes. The runner calls the op N times and wraps each call in its own
  spans: `call` (from entering the op until the result can be read), `<op>()` (until the function returns)
  and `read result` (waiting for the device). The library's spans sit inside them. The tab shows one call as
  a flame chart (the median call by default; any call can be picked), and a table of each step's *self time*
  (its time minus the spans inside it) averaged over the calls. That separates argument checks and setup (the
  op function's self time), dispatch, allocation, host-device copies, the kernel, and waiting for the result.
  It also times the same calls with tracing detached, which shows the tracing overhead.
- **Trace**: a timeline of kernel calls, one lane per thread, with totals per op. *Trace demo program* runs an
  MLP training step (forward, backward, concat, pad, dropout) on the chosen device. *Open trace file* loads any
  file written by `TraceCollector::write()`. The same files also open in chrome://tracing and in
  https://ui.perfetto.dev.
- **History**: every run, filterable by op and device, with the median of repeated runs over time.

Threads (`OMP_NUM_THREADS`) and the minimum time per size are set in the Kernels tab and apply to every run.

## Timing rules

Each size gets one warm-up call. Timed calls then run until the minimum time has passed, with at least 5 and at
most 5000 calls. Each timed call reads one element of its result, which waits for a GPU to finish. After
timing, one more call runs with the collector attached, so the trace hook never runs during the timed calls.
GPU spans measure the time spent in the call, and the GPU may still be working when a span ends.

## Tracing your own program

```cpp
#include "trace_collector.hpp"            // add observability/ to the include path
observe::TraceCollector trace;            // starts collecting
// ... mensurflow calls ...
trace.write("run.trace.json");           // open it in the Trace tab
```
