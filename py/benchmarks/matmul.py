import numpy as np
import torch as th
import time

device = th.device("mps")


def matmul_forloop_cpu(n: int) -> float:
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)

    out = np.zeros((n, n))

    start = time.time()

    for r in range(n):
        for c in range(n):
            buf = 0
            for k in range(n):
                buf += a[r, k]*b[k, c]

            out[r, c] = buf
    # print(out)
    # act = np.matmul(a, b)
    # print(act)

    return time.time() - start


def matmul_numpy_cpu(n: int) -> float:
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)

    start = time.time()

    c = np.matmul(a, b)

    return time.time() - start


def matmul_torch_cpu(n: int) -> float:
    a = th.rand(n)
    b = th.rand(n)

    start = time.time()

    c = th.matmul(a, b)

    return time.time() - start


def matmul_torch_gpu(n: int) -> float:
    a = th.rand(n)
    b = th.rand(n)

    start = time.time()

    c = th.matmul(a.to(device), b.to(device))

    return time.time() - start


if __name__ == "__main__":
    matmul_forloop_colwise_cpu(3)
