import numpy as np
import torch as th

n = 1
r = 2

limit = 100000  # The largest size array

while n < limit:
    np_array_a = np.random.normal(size=n)
    np_array_b = np.random.normal(size=n)
    n = int(n * r)
