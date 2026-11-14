"""Reference for optimizer_normalization_benchmark: PyTorch's own kernels at the same shapes and thread counts.

    python benchmarks/reference/optimizer_normalization.py

Inputs are the same numbers as the C++ benchmark's (fill_values / fill_positive there), so results can
be compared value for value. What is timed, op by op (outputs preallocated with out= where ATen
allows it, like our kernels write into preallocated buffers):

  layer_norm              aten.native_layer_norm.out (also writes mean and rstd, [rows] floats)
  layer_norm_backward     aten.native_layer_norm_backward.out (uses the saved mean/rstd; ours
                          recomputes them from x, one extra read of the input)
  rms_norm                F.rms_norm (allocates its result; ATen has no out= variant)
  rms_norm_backward       autograd.grad over F.rms_norm with a retained graph (allocates)
  batch_norm              aten.native_batch_norm.out, training and evaluation (also writes the
                          saved statistics)
  batch_norm_backward     aten.native_batch_norm_backward.out, training
  sgd (momentum)          torch._fused_sgd_ (the fused CPU kernel, called directly), and
                          torch.optim.SGD with its default implementation
  adam / adamw            torch._fused_adam_ / torch._fused_adamw_ (the fused CPU kernels, called
                          directly), and torch.optim.Adam / AdamW with fused=True (adds Python overhead)

Rates use the same byte counts as the C++ benchmark (inputs read once, outputs written once).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from common import header, measure, row, settle, thread_counts

aten = torch.ops.aten


def fill_values(n, seed):
    i = np.arange(n, dtype=np.int64)
    return torch.from_numpy((((i * 37 + seed * 101) % 97) - 48).astype(np.float32) / 12.0)


def fill_positive(n, seed):
    return 0.5 + fill_values(n, seed).abs() / 4.0


def sweep(label, impl, work, run, check=None):
    """One row per thread count; the result is checked (once, before timing) when `check` is given."""
    for threads in thread_counts(os.cpu_count()):
        torch.set_num_threads(threads)
        run()
        if check is not None:
            check()
        settle()
        row(label, impl, threads, measure(run), work)
    torch.set_num_threads(os.cpu_count())


def close(actual, expected, what, rtol=1e-4, atol=1e-4):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda m: f"{what}: {m}")


def optimizers():
    for name in ("sgd_momentum", "adam", "adamw"):
        print(f"\n-- {name} [n elements], in place")
        for n in (1000, 1 << 16, 1 << 20, 1 << 24):
            p = fill_values(n, 1).clone()
            g = fill_values(n, 2) / 8.0
            m = torch.zeros(n)
            v = torch.zeros(n)
            arrays = 5.0 if name == "sgd_momentum" else 7.0
            work = arrays * 4.0 * n
            step = torch.tensor(1.0)  # a fixed step number, as the C++ benchmark uses
            if name == "sgd_momentum":
                def fused():
                    torch._fused_sgd_([p], [g], [m], weight_decay=0.01, momentum=0.9, lr=0.01, dampening=0.0,
                                      nesterov=False, maximize=False, is_first_step=False)
                sweep(str(n), "torch fused op", work, fused)
                param = torch.nn.Parameter(p.clone())
                param.grad = g.clone()
                opt = torch.optim.SGD([param], lr=0.01, momentum=0.9, weight_decay=0.01)
                sweep(str(n), "torch optim.SGD", work, opt.step)
            else:
                op = torch._fused_adam_ if name == "adam" else torch._fused_adamw_
                decay = 0.01

                def fused():
                    op([p], [g], [m], [v], [], [step], lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=decay, eps=1e-8,
                       amsgrad=False, maximize=False)
                sweep(str(n), "torch fused op", work, fused)
                param = torch.nn.Parameter(p.clone())
                param.grad = g.clone()
                cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
                opt = cls([param], lr=1e-3, weight_decay=decay, fused=True)
                sweep(str(n), "torch optim fused", work, opt.step)
            print()


def line_norms():
    shapes = [(32, 768), (1024, 768), (8192, 1024), (32768, 1024)]
    eps = 1e-5
    print("\n-- layer_norm [rows x width]")
    for rows, width in shapes:
        x = fill_values(rows * width, 1).reshape(rows, width)
        w, b = fill_positive(width, 3), fill_values(width, 4)
        y = torch.empty_like(x)
        mean = torch.empty(rows, 1)
        rstd = torch.empty(rows, 1)
        run = lambda: aten.native_layer_norm.out(x, [width], w, b, eps, out0=y, out1=mean, out2=rstd)
        sweep(f"{rows}x{width}", "torch", 8.0 * x.numel(), run,
              lambda: close(y[:4], F.layer_norm(x[:4].double(), (width,), w.double(), b.double(), eps).float(), "layer_norm"))
        print()
    print("\n-- layer_norm_backward [rows x width]")
    for rows, width in shapes:
        x = fill_values(rows * width, 1).reshape(rows, width)
        dy = fill_values(rows * width, 2).reshape(rows, width)
        w, b = fill_positive(width, 3), fill_values(width, 4)
        y = torch.empty_like(x)
        mean = torch.empty(rows, 1)
        rstd = torch.empty(rows, 1)
        aten.native_layer_norm.out(x, [width], w, b, eps, out0=y, out1=mean, out2=rstd)
        dx, dw, db = torch.empty_like(x), torch.empty(width), torch.empty(width)
        run = lambda: aten.native_layer_norm_backward.out(dy, x, [width], mean, rstd, w, b, [True, True, True],
                                                          out0=dx, out1=dw, out2=db)
        sweep(f"{rows}x{width}", "torch", 12.0 * x.numel(), run)
        print()
    print("\n-- rms_norm [rows x width]")
    for rows, width in shapes:
        x = fill_values(rows * width, 1).reshape(rows, width)
        w = fill_positive(width, 3)
        sweep(f"{rows}x{width}", "torch (allocates)", 8.0 * x.numel(), lambda: F.rms_norm(x, (width,), w, eps))
        print()
    print("\n-- rms_norm_backward [rows x width]")
    for rows, width in shapes:
        x = fill_values(rows * width, 1).reshape(rows, width).requires_grad_()
        dy = fill_values(rows * width, 2).reshape(rows, width)
        w = fill_positive(width, 3).requires_grad_()
        y = F.rms_norm(x, (width,), w, eps)
        run = lambda: torch.autograd.grad(y, [x, w], dy, retain_graph=True)
        sweep(f"{rows}x{width}", "torch (allocates)", 12.0 * x.numel(), run)
        print()


def batch_norms():
    shapes = [(32, 64, 3136), (8, 256, 196), (1024, 1024, 1), (256, 512, 1)]
    eps, momentum = 1e-5, 0.1
    for title, mode in (("batch_norm, training", 0), ("batch_norm, evaluation", 1), ("batch_norm_backward, training", 2)):
        print(f"\n-- {title} [n x c x spatial]")
        for n, c, sp in shapes:
            total = n * c * sp
            shape = (n, c, sp) if sp > 1 else (n, c)
            x = fill_values(total, 1).reshape(shape)
            dy = fill_values(total, 2).reshape(shape)
            w, b = fill_positive(c, 3), fill_values(c, 4)
            rm, rv = fill_values(c, 5).clone(), fill_positive(c, 6).clone()
            y, dx = torch.empty_like(x), torch.empty_like(x)
            save_mean, save_invstd = torch.empty(c), torch.empty(c)
            dw, db = torch.empty(c), torch.empty(c)
            label = f"{n}x{c}x{sp}"
            if mode == 2:
                aten.native_batch_norm.out(x, w, b, rm, rv, True, momentum, eps, out=y, save_mean=save_mean,
                                           save_invstd=save_invstd)
                run = lambda: aten.native_batch_norm_backward.out(dy, x, w, rm, rv, save_mean, save_invstd, True, eps,
                                                                  [True, True, True], out0=dx, out1=dw, out2=db)
                sweep(label, "torch", 12.0 * total, run)
            else:
                training = mode == 0
                run = lambda: aten.native_batch_norm.out(x, w, b, rm, rv, training, momentum, eps, out=y,
                                                         save_mean=save_mean, save_invstd=save_invstd)
                expected = F.batch_norm(x.double(), rm.double().clone(), rv.double().clone(), w.double(), b.double(),
                                        training, momentum, eps)[: min(n, 4)].float()
                sweep(label, "torch", 8.0 * total, run, lambda: close(y[: min(n, 4)], expected, "batch_norm"))
            print()


def main():
    header("torch optimizers and normalization, float32")
    optimizers()
    line_norms()
    batch_norms()


if __name__ == "__main__":
    main()
