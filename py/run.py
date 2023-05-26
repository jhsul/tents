import time
import numpy as np
import pandas as pd
import fire

from tqdm import tqdm

import sys

from benchmarks.addition import cpu_for_loop_addition1d, cpu_numpy_addition1d, cpu_torch_addition1d, gpu_torch_addition1d


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
