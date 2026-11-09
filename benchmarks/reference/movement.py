"""Reference for movement_benchmark: torch.cat, torch.nn.functional.pad and torch.split at every
thread count, NumPy concatenate and pad as the single-thread reference.

    python benchmarks/reference/movement.py

Rate is GB/s of bytes moved (input read once, output written once), as in our benchmark.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from common import header, measure, row, settle, thread_counts


def fill(n):
    return ((np.arange(n) % 13) / 8.0 - 0.75).astype(np.float32)


def sweep(label, bytes_moved, torch_run, numpy_run, check):
    maximum = os.cpu_count()
    for threads in thread_counts(maximum):
        torch.set_num_threads(threads)
        torch_run()
        check()
        settle()
        row(label, "torch", threads, measure(torch_run), bytes_moved)
    torch.set_num_threads(maximum)
    settle()
    row(label, "numpy", 1, measure(numpy_run), bytes_moved)
    print()


def main():
    header("torch / numpy data movement, float32 (GB/s)")

    print("-- concat, 2 inputs along axis 0")
    for n in [1000, 32768, 1 << 20, 1 << 24]:
        half = n // 2
        a_np, b_np = fill(half), fill(n - half)
        a, b = torch.from_numpy(a_np), torch.from_numpy(b_np)
        out, out_np = torch.empty(n), np.empty(n, dtype=np.float32)
        want = np.concatenate([a_np, b_np])
        sweep(str(n), 8.0 * n, lambda: torch.cat([a, b], out=out),
              lambda: np.concatenate([a_np, b_np], out=out_np),
              lambda: np.testing.assert_array_equal(out.numpy(), want))

    print("-- concat, 8 inputs of [rows, 128] along the last axis")
    for rows in [1, 64, 4096, 16384]:
        pieces_np = [fill(rows * 128).reshape(rows, 128) for _ in range(8)]
        pieces = [torch.from_numpy(p) for p in pieces_np]
        out, out_np = torch.empty(rows, 1024), np.empty((rows, 1024), dtype=np.float32)
        want = np.concatenate(pieces_np, axis=1)
        sweep(f"{rows}x1024", 8.0 * rows * 1024, lambda: torch.cat(pieces, dim=1, out=out),
              lambda: np.concatenate(pieces_np, axis=1, out=out_np),
              lambda: np.testing.assert_array_equal(out.numpy(), want))

    print("-- pad [n, n] by 1 on every side (no out= in torch: the output is allocated per call)")
    for n in [32, 181, 1024, 4096]:
        x_np = fill(n * n).reshape(n, n)
        x = torch.from_numpy(x_np)
        want = np.pad(x_np, 1)
        sweep(f"{n}x{n}", 4.0 * (n * n + (n + 2) * (n + 2)), lambda: F.pad(x, (1, 1, 1, 1)),
              lambda: np.pad(x_np, 1),
              lambda: np.testing.assert_array_equal(F.pad(x, (1, 1, 1, 1)).numpy(), want))

    print("-- pad images [batch, 64, 56, 56] by 1 pixel")
    for batch in [1, 8, 32]:
        x_np = fill(batch * 64 * 56 * 56).reshape(batch, 64, 56, 56)
        x = torch.from_numpy(x_np)
        width = ((0, 0), (0, 0), (1, 1), (1, 1))
        want = np.pad(x_np, width)
        moved = 4.0 * batch * 64 * (56 * 56 + 58 * 58)
        sweep(f"{batch}x64x56x56", moved, lambda: F.pad(x, (1, 1, 1, 1)),
              lambda: np.pad(x_np, width),
              lambda: np.testing.assert_array_equal(F.pad(x, (1, 1, 1, 1)).numpy(), want))

    print("-- split [4096, 1024] into 8 column pieces, then contiguous() each")
    x = torch.from_numpy(fill(4096 * 1024).reshape(4096, 1024))
    settle()
    row("4096x1024", "split (views)", 1, measure(lambda: torch.split(x, 128, dim=1)), 0.0)
    for threads in [1, os.cpu_count()]:
        torch.set_num_threads(threads)
        settle()
        row("4096x1024", "split+contiguous", threads,
            measure(lambda: [p.contiguous() for p in torch.split(x, 128, dim=1)]),
            8.0 * 4096 * 1024)


if __name__ == "__main__":
    main()
