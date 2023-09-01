# load cameraman.png as a numpy array

import numpy as np
import torch
from proxtorch.proxtorch.operators.tv_2d import TV_2D
from proxtorch.proxtorch.operators.tvl1_2d import TVL1_2D
from skimage import data

x = data.camera()

# add noise to cameraman
x = x / 255
x_noisy = x + np.random.randn(*x.shape) * 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

x_noisy = torch.tensor(x_noisy, device=device)

tvl1 = TVL1_2D(sigma_l1=1e-2, sigma_tv=1e-2, device=device, max_iter=50)
prox_result_tvl1 = tvl1.prox(x_noisy, 1)
assert prox_result_tvl1.shape == x.shape

tv = TV_2D(sigma=1e-1, device=device, max_iter=50)
prox_result_tv = tv.prox(x_noisy, 1)
assert prox_result_tv.shape == x.shape

# Make a plot of prox_result to convince yourself that it is a smoothed version of x
import matplotlib.pyplot as plt

plt.imshow(x)
plt.title("Original Image")
plt.show()

plt.imshow(x_noisy.cpu().numpy())
plt.title("Noisy Image")
plt.show()

plt.imshow(prox_result_tv.cpu().numpy())
plt.title("Prox Result TV")
plt.show()

plt.imshow(prox_result_tvl1.cpu().numpy())
plt.title("Prox Result TVL1")
plt.show()

print("Plotting prox_result and original image...")
