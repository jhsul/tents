import numpy as np
import torch as th
import time

device = th.device("cuda:0")


def matmul_forloop_cpu(n: int) -> float:
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)

    out = np.zeros((n, n))

    start = time.time_ns()

    for r in range(n):
        for c in range(n):
            buf = 0
            for k in range(n):
                buf += a[r, k]*b[k, c]

            out[r, c] = buf
    # print(out)
    # act = np.matmul(a, b)
    # print(act)

    return (time.time_ns() - start) / 1e6


def matmul_numpy_cpu(n: int) -> float:
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)

    start = time.time_ns()

    c = np.matmul(a, b)

    return (time.time_ns() - start) / 1e6


def matmul_torch_cpu(n: int) -> float:
    a = th.rand((n, n))
    b = th.rand((n, n))

    start = time.time_ns()

    c = th.matmul(a, b)

    return (time.time_ns() - start) / 1e6


def matmul_torch_gpu(n: int) -> float:
    a = th.rand((n, n))
    b = th.rand((n, n))

    start = time.time_ns()

    c = th.matmul(a.to(device), b.to(device))

    return (time.time_ns() - start) / 1e6


def _batch_matmul_torch_cpu(n: int, size: int) -> float:
    a = th.rand((n, size, size))
    b = th.rand((n, size, size))

    start = time.time_ns()

    c = th.matmul(a, b)

    return (time.time_ns() - start) / 1e6


def _batch_matmul_torch_gpu(n: int, size: int) -> float:
    a = th.rand((n, size, size))
    b = th.rand((n, size, size))

    start = time.time_ns()

    c = th.matmul(a.to(device), b.to(device))

    return (time.time_ns() - start) / 1e6


def batch_matmul_4_torch_gpu(n: int) -> float:
    return _batch_matmul_torch_gpu(n, 4)


def batch_matmul_64_torch_gpu(n: int) -> float:
    return _batch_matmul_torch_gpu(n, 64)


def batch_matmul_256_torch_gpu(n: int) -> float:
    return _batch_matmul_torch_gpu(n, 256)


def batch_matmul_4_torch_cpu(n: int) -> float:
    return _batch_matmul_torch_cpu(n, 4)


def batch_matmul_64_torch_cpu(n: int) -> float:
    return _batch_matmul_torch_cpu(n, 64)


def batch_matmul_256_torch_cpu(n: int) -> float:
    return _batch_matmul_torch_cpu(n, 256)


if __name__ == "__main__":
    pass
    # matmul_for_cpu(3)
