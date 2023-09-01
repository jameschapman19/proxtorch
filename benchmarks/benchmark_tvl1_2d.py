import time

import numpy as np
import torch
from proxtorch.proxtorch.operators.tvl1_2d import TVL1_2D
from skprox.operators import TVL1


def benchmark_tvl1_vs_proxtorch_tvl2d1():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sigma_l1 = 1e-3
    sigma_tv = 1e-1
    p = int(np.sqrt(100 * 100 * 100))
    x = torch.cat((torch.ones(p // 2, p), torch.zeros(p // 2, p)), dim=0)
    x = x + torch.randn(p, p) * 0.1
    x = x.to(device)

    # Measure tensor movement time
    start_move = time.time()
    x.detach().cpu().numpy()
    torch.tensor(x, device=device)
    x.to(device)
    end_move = time.time()

    # Benchmark TVL1_2DProx
    tv = TVL1_2D(sigma_l1=sigma_l1, sigma_tv=sigma_tv, device=device, max_iter=10)
    start_proxtorch = time.time()
    _ = tv.prox(x, 1)
    end_proxtorch = time.time()

    # Benchmark TVL1
    sigma = sigma_l1 + sigma_tv
    l1_ratio = sigma_l1 / sigma

    x_numpy = x.cpu().numpy().ravel()
    tvl1 = TVL1(sigma, l1_ratio=l1_ratio, max_iter=500, shape=(p, p))
    start_tvl1 = time.time()
    _ = tvl1.prox(x_numpy, 1)
    end_tvl1 = time.time()

    # Results
    print("Time to move tensor to device: ", end_move - start_move)
    print("TVL1_2DProx time: ", end_proxtorch - start_proxtorch)
    print("TVL1 time: ", end_tvl1 - start_tvl1)


if __name__ == "__main__":
    benchmark_tvl1_vs_proxtorch_tvl2d1()
