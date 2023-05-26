import numpy as np
import torch as th
import time

device = th.device("mps")


def cpu_for_loop_addition1d(n: int) -> float:
    a = np.random.rand(n)
    b = np.random.rand(n)

    c = np.zeros(n)

    start = time.time()

    for i in range(n):
        c[i] = a[i] + b[i]

    return time.time() - start


def cpu_numpy_addition1d(n: int) -> float:
    a = np.random.rand(n)
    b = np.random.rand(n)

    start = time.time()

    c = a + b

    return time.time() - start


def cpu_torch_addition1d(n: int) -> float:
    a = th.rand(n)
    b = th.rand(n)

    start = time.time()

    c = a + b

    return time.time() - start


def gpu_torch_addition1d(n: int) -> float:
    # start = time.time()
    a = th.rand(n)
    b = th.rand(n)

    start = time.time()

    c = a.to(device) + b.to(device)

    return time.time() - start
