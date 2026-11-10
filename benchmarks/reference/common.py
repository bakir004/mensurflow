"""Shared timing and printing for the reference scripts, matching benchmarks/bench.hpp.

Each script times a PyTorch (and, for one thread, NumPy) version of an op at the same sizes and
thread counts as our C++ benchmark and prints the same columns, so the two tables can be put side by
side. Rules from docs/weekly/2026-11-04-plan.md: same dtype (float32), output preallocated where the
library allows it (out=), results checked before timing, a pause before every timed library, median
and minimum of 9 samples.
"""

import os
import platform
import statistics
import time

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

REPEATS = 9
MIN_SAMPLE_S = 2e-3


def thread_counts(maximum):
    """1, powers of two below the maximum, maximum / 2 and the maximum, as bench::thread_counts."""
    counts = {1, maximum, max(1, maximum // 2)}
    p = 2
    while p < maximum:
        counts.add(p)
        p *= 2
    return sorted(counts)


def settle():
    """Let the previous library's worker threads go idle before timing the next one."""
    time.sleep(0.5)


def measure(run):
    """Median, min and max seconds per call; fast calls are repeated until a sample lasts 2 ms."""
    run()  # warm-up
    calls = 1
    while True:
        start = time.perf_counter()
        for _ in range(calls):
            run()
        if time.perf_counter() - start >= MIN_SAMPLE_S or calls >= 1 << 20:
            break
        calls *= 2
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(calls):
            run()
        samples.append((time.perf_counter() - start) / calls)
    return statistics.median(samples), min(samples), max(samples)


def header(title):
    require_backends()
    print(f"== {title} ==")
    print(f"machine: {platform.machine()} {platform.processor()} ({os.cpu_count()} threads)")
    print(f"python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}")
    print("torch parallel info:")
    for line in torch.__config__.parallel_info().splitlines():
        if line.strip():
            print("  " + line.strip())
    print()
    print(f"{'n':>22} | {'impl':<18} | {'threads':>7} | {'median ms':>10} | {'min ms':>9} | "
          f"{'max ms':>9} | {'rate':>9}")


def require_backends():
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if missing:
        raise RuntimeError("missing reference package(s): " + ", ".join(missing))


def row(n, impl, threads, timing, work, scale=1e9):
    median, low, high = timing
    print(f"{n:>22} | {impl:<18} | {threads:>7} | {median * 1e3:10.4f} | {low * 1e3:9.4f} | "
          f"{high * 1e3:9.4f} | {work / median / scale:9.2f}")
