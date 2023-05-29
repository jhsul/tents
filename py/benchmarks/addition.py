import numpy as np
import torch as th
import time

device = th.device("mps")


def vecadd_forloop_cpu(n: int) -> float:
    a = np.random.rand(n)
    b = np.random.rand(n)

    c = np.zeros(n)

    start = time.time()

    for i in range(n):
        c[i] = a[i] + b[i]

    return time.time() - start


def vecadd_numpy_cpu(n: int) -> float:
    a = np.random.rand(n)
    b = np.random.rand(n)

    start = time.time()

    c = a + b

    return time.time() - start


def vecadd_torch_cpu(n: int) -> float:
    a = th.rand(n)
    b = th.rand(n)

    start = time.time()

    c = a + b

    return time.time() - start


def vecadd_torch_gpu(n: int) -> float:
    # start = time.time()
    a = th.rand(n)
    b = th.rand(n)

    start = time.time()

    c = a.to(device) + b.to(device)

    return time.time() - start
