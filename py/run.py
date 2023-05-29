import time
import numpy as np
import pandas as pd
import fire

from tqdm import tqdm

import sys

from benchmarks.addition import vecadd_forloop_cpu, vecadd_numpy_cpu, vecadd_torch_cpu, vecadd_torch_gpu
from benchmarks.matmul import matmul_forloop_cpu,  matmul_numpy_cpu, matmul_torch_cpu, matmul_torch_gpu


def main(name: str, k: int = 1, c: int = 1, save: bool = False):
    # print(sys.modules)
    df = pd.DataFrame(columns=['n', 'time', 'stddev'])

    for scale in tqdm(range(k)):
        n = 1 << scale
        times = []
        for i in range(c):
            f = globals()[name]
            t = f(n)
            times.append(t)

        df.loc[len(df)] = [n, np.mean(times), np.std(times)]

    if save:
        filename = f"./data/{name}-py.csv"

        df.to_csv(filename, index=False)

    else:
        print(df)


if __name__ == "__main__":
    fire.Fire(main)
