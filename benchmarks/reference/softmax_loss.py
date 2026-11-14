"""Reference for softmax_loss_benchmark: PyTorch's own kernels at the same shapes and thread counts.

    python benchmarks/reference/softmax_loss.py

What is timed, op by op (outputs preallocated with out= where ATen allows it):
  softmax / log_softmax           aten._softmax.out / aten._log_softmax.out
  their backward                  aten._softmax_backward_data.out / _log_softmax_backward_data.out
  cross_entropy (mean)            F.cross_entropy
  cross_entropy_backward (mean)   what autograd runs: aten.nll_loss_backward, then
                                  aten._log_softmax_backward_data (log_softmax saved from forward)
  mse_loss and backward           F.mse_loss, aten.mse_loss_backward
  bce_with_logits and backward    F.binary_cross_entropy_with_logits; backward as the formula
                                  (sigmoid(x) - t) * (1 / n), since ATen has no single kernel for it
  attention (causal)              F.scaled_dot_product_attention (fused on the CPU) and the naive
                                  q k^T, masked softmax, p v written with torch ops (what ours does)

Rates use the same byte and flop counts as the C++ benchmark.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from common import header, measure, row, settle, thread_counts

aten = torch.ops.aten


def logits(rows, cols):
    i = np.arange(rows * cols)
    return torch.from_numpy((((i * 37) % 97) - 48).astype(np.float32) / 12.0).reshape(rows, cols)


def labels(rows, classes):
    return torch.from_numpy(((np.arange(rows) * 13 + 5) % classes).astype(np.int64))


def sweep(label, impl, work, run, check=None):
    for threads in thread_counts(os.cpu_count()):
        torch.set_num_threads(threads)
        run()
        if check is not None:
            check()
        settle()
        row(label, impl, threads, measure(run), work)
    torch.set_num_threads(os.cpu_count())


def main():
    header("torch softmax, losses and attention, float32")
    shapes = [(64, 1000), (1024, 1000), (4096, 128), (16384, 512), (32, 32000)]

    for log in (False, True):
        name = "log_softmax" if log else "softmax"
        print(f"\n-- {name} over the last axis [rows, cols]")
        for rows, cols in shapes:
            x = logits(rows, cols)
            out = torch.empty_like(x)
            op = aten._log_softmax.out if log else aten._softmax.out
            ref = torch.log_softmax(x, -1) if log else torch.softmax(x, -1)
            sweep(f"{rows}x{cols}", "torch", 8.0 * x.numel(), lambda: op(x, -1, False, out=out),
                  lambda: torch.testing.assert_close(out, ref))
            print()

    for log in (False, True):
        name = "log_softmax" if log else "softmax"
        print(f"\n-- {name}_backward [rows, cols]")
        for rows, cols in [(1024, 1000), (16384, 512)]:
            x = logits(rows, cols)
            y = torch.log_softmax(x, -1) if log else torch.softmax(x, -1)
            g = torch.from_numpy(((np.arange(x.numel()) % 17) / 8.0 - 1.0).astype(np.float32)).reshape(rows, cols)
            out = torch.empty_like(x)
            op = aten._log_softmax_backward_data.out if log else aten._softmax_backward_data.out
            # ATen names the output `out` for one and `grad_input` for the other.
            target = {"out": out} if log else {"grad_input": out}
            sweep(f"{rows}x{cols}", "torch", 12.0 * x.numel(),
                  lambda: op(g, y, -1, torch.float32, **target))
            print()

    ce_shapes = [(64, 1000), (1024, 1000), (8192, 1000), (32, 32000)]
    print("\n-- cross_entropy, mean [rows, classes]")
    for rows, classes in ce_shapes:
        x, y = logits(rows, classes), labels(rows, classes)
        sweep(f"{rows}x{classes}", "torch", 4.0 * x.numel() + 8.0 * rows,
              lambda: F.cross_entropy(x, y))
        print()

    print("\n-- cross_entropy_backward, mean [rows, classes]")
    for rows, classes in ce_shapes:
        x, y = logits(rows, classes), labels(rows, classes)
        logp = torch.log_softmax(x, -1)
        one = torch.ones(())
        total_weight = torch.tensor(float(rows))
        out = torch.empty_like(x)

        def run():
            d_logp = aten.nll_loss_backward(one, logp, y, None, 1, -100, total_weight)
            aten._log_softmax_backward_data.out(d_logp, logp, -1, torch.float32, out=out)

        expected = torch.softmax(x, -1)
        expected[torch.arange(rows), y] -= 1.0
        expected /= rows
        sweep(f"{rows}x{classes}", "torch", 8.0 * x.numel() + 8.0 * rows, run,
              lambda: torch.testing.assert_close(out, expected))
        print()

    sizes = [1000, 1 << 16, 1 << 20, 1 << 24]
    for name in ("mse_loss", "binary_cross_entropy_with_logits"):
        print(f"\n-- {name}, mean [n]")
        for n in sizes:
            x = logits(1, n).reshape(n)
            t = torch.from_numpy((np.arange(n) % 2).astype(np.float32))
            fn = F.mse_loss if name == "mse_loss" else F.binary_cross_entropy_with_logits
            sweep(str(n), "torch", 8.0 * n, lambda: fn(x, t))
            print()
        print(f"\n-- {name}_backward, mean [n]")
        for n in sizes:
            x = logits(1, n).reshape(n)
            t = torch.from_numpy((np.arange(n) % 2).astype(np.float32))
            out = torch.empty_like(x)
            if name == "mse_loss":
                one = torch.ones(())
                run = lambda: aten.mse_loss_backward.grad_input(one, x, t, 1, grad_input=out)
            else:
                scale = 1.0 / n
                run = lambda: torch.sub(torch.sigmoid(x), t, out=out).mul_(scale)
            sweep(str(n), "torch", 12.0 * n, run)
            print()

    print("\n-- attention [batch, heads, L, 64], causal, GFLOP/s of q k^T and p v")
    for batch, heads, length in [(1, 8, 128), (4, 12, 256), (1, 16, 1024)]:
        q = logits(batch * heads * length, 64).reshape(batch, heads, length, 64)
        scale = 1.0 / math.sqrt(64)
        hidden = torch.ones(length, length, dtype=torch.bool).triu(1)

        def naive():
            s = torch.matmul(q, q.transpose(-2, -1)).mul_(scale).masked_fill_(hidden, float("-inf"))
            return torch.matmul(torch.softmax(s, -1), q)

        fused = lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True)
        flops = 4.0 * batch * heads * length * length * 64
        label = f"{batch}x{heads}x{length}"
        sweep(label, "torch sdpa", flops, fused,
              lambda: torch.testing.assert_close(fused(), naive(), rtol=1e-4, atol=1e-5))
        sweep(label, "torch naive", flops, naive)
        print()


if __name__ == "__main__":
    main()
