"""Reference for argmin_var_std_benchmark: torch.argmin (last axis), torch.var and torch.std along
an axis with correction=1, forward and their autograd backward, at the same shapes and thread
counts as our benchmark, plus NumPy on one thread.

    python benchmarks/reference/argmin_var_std.py [--threads=1,6,12,24] [--only=name]

Rate is GB/s of the input read once (plus the gradient written once for backward), as in our table.
"""

import os
import sys

import numpy as np
import torch

from common import header, measure, row, settle, thread_counts

ARGMIN = [
    ("classes 32x1000", 32, 1000),
    ("classes 1024x1000", 1024, 1000),
    ("k-means 65536x64", 65536, 64),
    ("long rows 4x2^22", 4, 1 << 22),
    ("small 8x100", 8, 100),
]

MOMENTS = [
    ("layer-norm stats [6272,768] axis 1", (6272, 768), 1),
    ("batch-norm stats [100352,64] axis 0", (100352, 64), 0),
    ("middle axis [32,64,3136] axis 1", (32, 64, 3136), 1),
    ("vector 2^24", (1 << 24,), 0),
    ("small [8,100] axis 1", (8, 100), 1),
]


def fill_a(n):
    """bench::fill_a: (i % 13) / 8 - 0.75."""
    return ((np.arange(n) % 13) / 8.0 - 0.75).astype(np.float32)


def main():
    counts, only = None, ""
    for arg in sys.argv[1:]:
        key, _, value = arg.lstrip("-").partition("=")
        if key == "threads":
            counts = [int(v) for v in value.split(",")]
        elif key == "only":
            only = value
    counts = counts or thread_counts(os.cpu_count())
    header("torch.argmin / torch.var / torch.std, float32, correction=1")

    for name, rows, width in ARGMIN:
        if only and only not in name:
            continue
        x_np = fill_a(rows * width).reshape(rows, width)
        x_np[-1, -1] = -10.0
        x = torch.from_numpy(x_np)
        out = torch.empty(rows, dtype=torch.int64)
        print(f"== argmin {name} ==")
        for threads in counts:
            torch.set_num_threads(threads)
            torch.argmin(x, dim=1, out=out)
            assert np.array_equal(out.numpy(), np.argmin(x_np, axis=1))
            settle()
            row("argmin", "torch", threads, measure(lambda: torch.argmin(x, dim=1, out=out)),
                4.0 * x.numel())
        settle()
        row("argmin", "numpy", 1, measure(lambda: np.argmin(x_np, axis=1)), 4.0 * x.numel())
        print()

    for name, shape, axis in MOMENTS:
        if only and only not in name:
            continue
        n = int(np.prod(shape))
        x_np = (fill_a(n) + 1000.0).astype(np.float32).reshape(shape)
        x = torch.from_numpy(x_np)
        print(f"== var/std {name} ==")
        for threads in counts:
            torch.set_num_threads(threads)
            with torch.no_grad():
                settle()
                row("var", "torch", threads,
                    measure(lambda: torch.var(x, dim=axis, correction=1)), 4.0 * n)
                settle()
                row("std", "torch", threads,
                    measure(lambda: torch.std(x, dim=axis, correction=1)), 4.0 * n)
            xg = x.clone().requires_grad_()
            v = torch.var(xg, dim=axis, correction=1)
            s = torch.std(xg, dim=axis, correction=1)
            g = torch.randn_like(v)
            settle()
            row("var backward", "torch autograd", threads,
                measure(lambda: torch.autograd.grad(v, xg, g, retain_graph=True)), 8.0 * n)
            settle()
            row("std backward", "torch autograd", threads,
                measure(lambda: torch.autograd.grad(s, xg, g, retain_graph=True)), 8.0 * n)
        settle()
        row("var", "numpy", 1, measure(lambda: np.var(x_np, axis=axis, ddof=1)), 4.0 * n)
        print()


if __name__ == "__main__":
    main()
