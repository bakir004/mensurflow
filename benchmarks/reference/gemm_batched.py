"""Reference for gemm_batched_benchmark: torch.bmm at every thread count, and NumPy's stacked matmul
(one BLAS sgemm call per matrix, the "loop of cblas_sgemm" reference).

    python benchmarks/reference/gemm_batched.py

Rate is GFLOP/s for 2 * batch * m * n * k, as in our benchmark. NumPy's row runs with its BLAS's own
default threading (printed as "blas"); which BLAS that is is printed at the end. Do not set
MKL_NUM_THREADS or OMP_NUM_THREADS for this script: PyTorch reads them as its own thread count.
On Apple silicon both libraries call Apple Accelerate, whose sgemm runs on the AMX matrix unit
rather than the NEON cores our kernels use, so it is a reference beyond the CPU cores' FMA peak.
"""

import os
import re

import numpy as np
import torch

from common import header, measure, row, settle, thread_counts

SIZES = [(1024, 8, 8, 8), (256, 32, 32, 32), (64, 128, 128, 64), (16, 256, 256, 256),
         (4, 512, 512, 512)]


def fill(shape, period, offset):
    """bench::fill_a / fill_b: (i % period) / 8 - offset."""
    n = int(np.prod(shape))
    return ((np.arange(n) % period) / 8.0 - offset).astype(np.float32).reshape(shape)


def main():
    header("torch.bmm and numpy.matmul, batched GEMM, float32 (GFLOP/s)")
    maximum = os.cpu_count()
    for batch, m, n, k in SIZES:
        label = f"{batch}x{m}x{n}x{k}"
        a_np = fill((batch, m, k), 13, 0.75)
        b_np = fill((batch, k, n), 17, 1.0)
        a, b = torch.from_numpy(a_np), torch.from_numpy(b_np)
        out = torch.empty(batch, m, n)
        flops = 2.0 * batch * m * n * k
        expected = np.matmul(a_np.astype(np.float64), b_np.astype(np.float64))
        for threads in thread_counts(maximum):
            torch.set_num_threads(threads)
            torch.bmm(a, b, out=out)
            assert np.allclose(out.numpy(), expected, rtol=1e-4, atol=1e-3), label
            settle()
            row(label, "torch.bmm", threads, measure(lambda: torch.bmm(a, b, out=out)), flops)
        torch.set_num_threads(maximum)
        out_np = np.empty((batch, m, n), dtype=np.float32)
        np.matmul(a_np, b_np, out=out_np)
        assert np.allclose(out_np, expected, rtol=1e-4, atol=1e-3), label
        settle()
        row(label, "numpy.matmul", "blas", measure(lambda: np.matmul(a_np, b_np, out=out_np)),
            flops)
        print()
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    print(f"numpy BLAS: {blas['name']} {blas.get('version', '')}")
    found = re.search(r"BLAS_INFO=(\w+)", torch.__config__.show())
    print(f"torch BLAS: {found.group(1) if found else 'unknown'}")


if __name__ == "__main__":
    main()
