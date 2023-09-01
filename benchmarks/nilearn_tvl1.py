import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nilearn.decoding.proximal_operators import _prox_tvl1

from proxtorch.operators import TVL1_3DProx

sns.set_context("paper")
sns.set_style("whitegrid")


def time_function_call(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    return time.time() - start


dims = [10, 50, 100]
n_repeats = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating a dataframe to hold results
df = pd.DataFrame(columns=["Dimension", "Time", "Method"])

for dim in dims:
    for _ in range(n_repeats):
        x = np.random.rand(dim, dim, dim)
        x_torch_cpu = torch.tensor(x, device="cpu", dtype=torch.float32)
        x_torch_gpu = torch.tensor(x, device=device, dtype=torch.float32)

        tvl1_proxtorch = TVL1_3DProx(alpha=1.0, l1_ratio=0.5).prox

        # Appending results to the dataframe using pd.concat()
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Dimension": dim,
                            "Time": time_function_call(
                                tvl1_proxtorch, x_torch_cpu, 1.0
                            ),
                            "Method": "ProxTorch CPU",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "Dimension": dim,
                            "Time": time_function_call(
                                _prox_tvl1, x, l1_ratio=0.5, weight=1.0
                            ),
                            "Method": "Nilearn",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        if device.type == "cuda":
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "Dimension": dim,
                                "Time": time_function_call(
                                    tvl1_proxtorch, x_torch_gpu, 1.0
                                ),
                                "Method": "ProxTorch GPU",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Dimension", y="Time", hue="Method", marker="o", errorbar="sd")
plt.title("Performance comparison of TV-L1 Proximal Operator solvers with Uncertainty")
plt.xlabel("Dimension (size of the cubic array)")
plt.ylabel("Average Execution Time (seconds)")
plt.tight_layout()
plt.savefig("joss/TVL1_Benchmark.svg")
plt.show()
