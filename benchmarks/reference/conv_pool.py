"""Reference for conv2d_benchmark and pooling_benchmark: PyTorch's conv2d, max_pool2d (with
return_indices) and avg_pool2d, forward and their autograd backward, at the same shapes and thread
counts as our benchmarks.

    python benchmarks/reference/conv_pool.py [--threads=1,6,12,24] [--batches=1,32] [--only=name]

Rates: conv2d in GFLOP/s at 2 flops per multiply-add (backward counts data and weight, 4 flops per
multiply-add), pooling in GB/s of input read once and output (and indices) written once, as in our
tables. The backward timing is one torch.autograd.grad call on a graph built once, so it includes
PyTorch's autograd bookkeeping, as our op-level autograd would.
"""

import os
import sys

import torch
import torch.nn.functional as F

from common import header, measure, row, settle, thread_counts

# The same layers as benchmarks/conv/conv2d_benchmark.cpp: name, c_in, h, c_out, k, stride, pad.
CONV_LAYERS = [
    ("r18.conv1 7x7/2", 3, 224, 64, 7, 2, 3),
    ("r18.layer1 3x3", 64, 56, 64, 3, 1, 1),
    ("r18.layer2.0 3x3/2", 64, 56, 128, 3, 2, 1),
    ("r18.layer3 3x3", 256, 14, 256, 3, 1, 1),
    ("r18.layer4 3x3", 512, 7, 512, 3, 1, 1),
    ("r50.layer1 1x1 256>64", 256, 56, 64, 1, 1, 0),
    ("r50.layer4 1x1 512>2048", 512, 7, 2048, 1, 1, 0),
    ("small 3x3 16ch 16px", 16, 16, 16, 3, 1, 1),
]

# The same pools as benchmarks/conv/pooling_benchmark.cpp: name, channels, h, kernel, stride, pad.
POOLS = [
    ("r18.maxpool 3x3/2", 64, 112, 3, 2, 1),
    ("2x2/2 128ch 56px", 128, 56, 2, 2, 0),
    ("global 7x7 512ch", 512, 7, 7, 7, 0),
    ("small 3x3/1 8ch 16px", 8, 16, 3, 1, 1),
]


def options():
    opts = {"threads": None, "batches": [1, 32], "only": ""}
    for arg in sys.argv[1:]:
        key, _, value = arg.lstrip("-").partition("=")
        if key in ("threads", "batches"):
            opts[key] = [int(v) for v in value.split(",")]
        elif key == "only":
            opts["only"] = value
    return opts


def conv(opts, counts):
    for name, c_in, h, c_out, k, stride, pad in CONV_LAYERS:
        if opts["only"] and opts["only"] not in name:
            continue
        for batch in opts["batches"]:
            torch.manual_seed(0)
            x = torch.randn(batch, c_in, h, h)
            w = torch.randn(c_out, c_in, k, k) * 0.05
            b = torch.randn(c_out)
            h_out = (h + 2 * pad - k) // stride + 1
            macs = batch * c_out * c_in * k * k * h_out * h_out
            label = f"{name} N={batch}"
            print(f"== {label}  ({macs // 1_000_000} M multiply-adds) ==")
            for threads in counts:
                torch.set_num_threads(threads)
                with torch.no_grad():
                    settle()
                    row("forward", "torch conv2d", threads,
                        measure(lambda: F.conv2d(x, w, b, stride, pad)), 2 * macs)
                xg = x.clone().requires_grad_()
                wg = w.clone().requires_grad_()
                bg = b.clone().requires_grad_()
                y = F.conv2d(xg, wg, bg, stride, pad)
                gy = torch.randn_like(y)
                settle()
                row("backward", "torch autograd", threads,
                    measure(lambda: torch.autograd.grad(y, (xg, wg, bg), gy, retain_graph=True)),
                    4 * macs)
            print()


def pooling(opts, counts):
    for name, channels, h, k, stride, pad in POOLS:
        if opts["only"] and opts["only"] not in name:
            continue
        for batch in opts["batches"]:
            torch.manual_seed(0)
            x = torch.randn(batch, channels, h, h)
            h_out = (h + 2 * pad - k) // stride + 1
            n_in, n_out = x.numel(), batch * channels * h_out * h_out
            label = f"{name} N={batch}"
            print(f"== {label} ==")
            for threads in counts:
                torch.set_num_threads(threads)
                with torch.no_grad():
                    settle()
                    row("max forward", "torch +indices", threads,
                        measure(lambda: F.max_pool2d(x, k, stride, pad, return_indices=True)),
                        4 * n_in + 12 * n_out)
                    settle()
                    row("avg forward", "torch", threads,
                        measure(lambda: F.avg_pool2d(x, k, stride, pad)), 4 * n_in + 4 * n_out)
                xg = x.clone().requires_grad_()
                y_max = F.max_pool2d(xg, k, stride, pad)
                y_avg = F.avg_pool2d(xg, k, stride, pad)
                gy = torch.randn_like(y_max)
                settle()
                row("max backward", "torch autograd", threads,
                    measure(lambda: torch.autograd.grad(y_max, xg, gy, retain_graph=True)),
                    12 * n_out + 4 * n_in)
                settle()
                row("avg backward", "torch autograd", threads,
                    measure(lambda: torch.autograd.grad(y_avg, xg, gy, retain_graph=True)),
                    4 * n_out + 4 * n_in)
            print()


def main():
    opts = options()
    counts = opts["threads"] or thread_counts(os.cpu_count())
    header("torch conv2d / max_pool2d / avg_pool2d, float32 NCHW")
    print(torch.__config__.show())
    conv(opts, counts)
    pooling(opts, counts)


if __name__ == "__main__":
    main()
