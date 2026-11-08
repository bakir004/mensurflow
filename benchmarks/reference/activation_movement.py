"""Compare activation backward and tensor movement operations with PyTorch and NumPy.

Run from the repository root:

    python benchmarks/reference/activation_movement.py

PyTorch is timed at the requested thread counts. NumPy is timed as the one-thread
array reference. Backward operations use torch.autograd.grad, which is the
reference explicitly permitted by the weekly plan.
"""

import os
import sys
import math

from common import header, measure, np, row, require_backends, settle, torch, thread_counts


def fill(shape):
    size = int(np.prod(shape))
    return ((np.arange(size) % 17) / 8.0 - 1.0).astype(np.float32).reshape(shape)


def check(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5, err_msg=label)


def exact_gelu_derivative(values):
    erf_values = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in values.ravel()),
        dtype=np.float64,
        count=values.size,
    ).reshape(values.shape)
    return 0.5 * (1.0 + erf_values) + values * np.exp(-0.5 * values * values) / math.sqrt(
        2.0 * math.pi
    )


def time_pair(label, torch_run, numpy_run, check_torch, work, maximum):
    numpy_run()
    for threads in thread_counts(maximum):
        torch.set_num_threads(threads)
        torch_run()
        check_torch()
        settle()
        row(label, "torch", threads, measure(torch_run), work)
    settle()
    row(label, "numpy", 1, measure(numpy_run), work)
    print()


def benchmark_backward(maximum):
    print("-- activation backward, float32")
    x_np = fill((1 << 20,))
    gradient_np = fill(x_np.shape)
    x = torch.from_numpy(x_np).requires_grad_()
    gradient = torch.from_numpy(gradient_np)

    operations = {
        "relu_backward": lambda: torch.autograd.grad(
            torch.relu(x), x, gradient, retain_graph=True
        )[0],
        "sigmoid_backward": lambda: torch.autograd.grad(
            torch.sigmoid(x), x, gradient, retain_graph=True
        )[0],
        "tanh_backward": lambda: torch.autograd.grad(
            torch.tanh(x), x, gradient, retain_graph=True
        )[0],
        "gelu_backward": lambda: torch.autograd.grad(
            torch.nn.functional.gelu(x), x, gradient, retain_graph=True
        )[0],
        "silu_backward": lambda: torch.autograd.grad(
            torch.nn.functional.silu(x), x, gradient, retain_graph=True
        )[0],
    }

    for name, torch_run in operations.items():
        if name == "relu_backward":
            numpy_run = lambda: np.multiply(gradient_np, x_np > 0)
        elif name == "sigmoid_backward":
            numpy_run = lambda: np.multiply(
                gradient_np, (1.0 / (1.0 + np.exp(-x_np))) *
                (1.0 - 1.0 / (1.0 + np.exp(-x_np)))
            )
        elif name == "tanh_backward":
            numpy_run = lambda: np.multiply(gradient_np, 1.0 - np.tanh(x_np) ** 2)
        elif name == "gelu_backward":
            numpy_run = lambda: np.multiply(gradient_np, exact_gelu_derivative(x_np))
        else:
            numpy_run = lambda: np.multiply(
                gradient_np,
                (1.0 / (1.0 + np.exp(-x_np))) +
                x_np * (1.0 / (1.0 + np.exp(-x_np))) *
                (1.0 - 1.0 / (1.0 + np.exp(-x_np)))
            )
        torch_result = [None]
        numpy_result = [None]

        def run_torch():
            torch_result[0] = torch_run()

        def run_numpy():
            numpy_result[0] = numpy_run()

        time_pair(
            f"{name}, n=1048576",
            run_torch,
            run_numpy,
            lambda: check(torch_result[0].detach().numpy(), numpy_result[0], name),
            8.0 * x_np.size,
            maximum,
        )


def benchmark_views(maximum):
    print("-- contiguous materialization and permute")
    matrix_np = fill((2048, 2048))
    matrix = torch.from_numpy(matrix_np)
    torch_result = [None]
    numpy_result = [None]
    time_pair(
        "transpose contiguous, 2048x2048",
        lambda: torch_result.__setitem__(0, matrix.t().contiguous()),
        lambda: numpy_result.__setitem__(0, np.ascontiguousarray(matrix_np.T)),
        lambda: check(torch_result[0].numpy(), numpy_result[0], "transpose contiguous"),
        2.0 * matrix_np.nbytes,
        maximum,
    )

    cube_np = fill((64, 128, 256))
    cube = torch.from_numpy(cube_np)
    time_pair(
        "permute contiguous, 64x128x256",
        lambda: torch_result.__setitem__(0, cube.permute(2, 0, 1).contiguous()),
        lambda: numpy_result.__setitem__(0, np.ascontiguousarray(cube_np.transpose(2, 0, 1))),
        lambda: check(torch_result[0].numpy(), numpy_result[0], "permute contiguous"),
        2.0 * cube_np.nbytes,
        maximum,
    )


def benchmark_selection(maximum):
    print("-- gather and index_select")
    values_np = fill((1024, 1024))
    values = torch.from_numpy(values_np)
    columns_np = np.arange(1024, dtype=np.int64)[::-1].copy()
    columns = torch.from_numpy(columns_np)
    torch_result = [None]
    numpy_result = [None]
    time_pair(
        "index_select, 1024x1024 axis=1",
        lambda: torch_result.__setitem__(0, torch.index_select(values, 1, columns)),
        lambda: numpy_result.__setitem__(0, np.take(values_np, columns_np, axis=1)),
        lambda: check(torch_result[0].numpy(), numpy_result[0], "index_select"),
        2.0 * values_np.nbytes,
        maximum,
    )

    indices_np = np.tile(columns_np, (1024, 1))
    indices = torch.from_numpy(indices_np)
    time_pair(
        "gather, 1024x1024 axis=1",
        lambda: torch_result.__setitem__(0, torch.gather(values, 1, indices)),
        lambda: numpy_result.__setitem__(0, np.take_along_axis(values_np, indices_np, axis=1)),
        lambda: check(torch_result[0].numpy(), numpy_result[0], "gather"),
        2.0 * values_np.nbytes,
        maximum,
    )


def main():
    try:
        require_backends()
    except RuntimeError as error:
        print(f"SKIPPED: {error}", file=sys.stderr)
        return 2
    header("Activation backward and tensor movement: PyTorch and NumPy")
    maximum = os.cpu_count() or 1
    benchmark_backward(maximum)
    benchmark_views(maximum)
    benchmark_selection(maximum)


if __name__ == "__main__":
    raise SystemExit(main())
