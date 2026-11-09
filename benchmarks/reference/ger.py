"""Reference for ger_benchmark: torch.addr (alpha * outer(x, y) + a, into a preallocated out) at every
thread count, and NumPy's outer product plus a on one thread.

    python benchmarks/reference/ger.py

Rate is GB/s of a read once and out written once, as in our benchmark.
"""

import os

import numpy as np
import torch

from common import header, measure, row, settle, thread_counts


def fill(n, period, offset):
    return ((np.arange(n) % period) / 8.0 - offset).astype(np.float32)


def main():
    header("torch.addr and numpy outer, rank-1 update, float32 (GB/s)")
    for n in [32, 181, 1024, 4096]:
        x_np, y_np = fill(n, 13, 0.75), fill(n, 17, 1.0)
        a_np = fill(n * n, 13, 0.75).reshape(n, n)
        x, y, a = torch.from_numpy(x_np), torch.from_numpy(y_np), torch.from_numpy(a_np)
        out = torch.empty(n, n)
        want = 0.5 * np.outer(x_np, y_np) + a_np
        moved = 8.0 * n * n
        for threads in thread_counts(os.cpu_count()):
            torch.set_num_threads(threads)
            torch.addr(a, x, y, alpha=0.5, out=out)
            np.testing.assert_allclose(out.numpy(), want, rtol=1e-6)
            settle()
            row(f"{n}x{n}", "torch.addr", threads,
                measure(lambda: torch.addr(a, x, y, alpha=0.5, out=out)), moved)
        out_np = np.empty((n, n), dtype=np.float32)
        settle()
        row(f"{n}x{n}", "numpy outer + a", 1,
            measure(lambda: np.add(np.multiply.outer(0.5 * x_np, y_np), a_np, out=out_np)), moved)
        print()


if __name__ == "__main__":
    main()
