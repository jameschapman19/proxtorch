"""
Image Restoration: ProxTorch Logo via TV and TV-L1 Regularization
==================================================================

Using ProxTorch's TV_2D and TVL1_2D operators, this example demonstrates image restoration
of a noisy ProxTorch logo through TV and TV-L1 regularization.

Dependencies:
- torch, torch.nn, torch.optim
- numpy
- matplotlib
- pytorch_lightning
- proxtorch

"""

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset

from proxtorch.operators import TVL1_2D

# Set seed
seed_everything(42)

# Load ProxTorch Logo as jpg then convert to grayscale numpy array
proxtorch_logo = plt.imread("../proxtorch-logo.jpg")
# Downsample to 64x64
proxtorch_logo = proxtorch_logo[::4, ::4]
proxtorch_logo = 1 - np.mean(proxtorch_logo, axis=2)
# Normalize to [0, 1]
proxtorch_logo = (proxtorch_logo - np.min(proxtorch_logo)) / (
    np.max(proxtorch_logo) - np.min(proxtorch_logo)
)


class TVL1Restoration(pl.LightningModule):
    def __init__(self, alpha, l1_ratio):
        super().__init__()
        self.restored = torch.nn.Parameter(torch.zeros(proxtorch_logo.shape))
        self.tvl1_prox = TVL1_2D(alpha=alpha, l1_ratio=l1_ratio)
        self.automatic_optimization = False

    def forward(self, x):
        return self.restored

    def training_step(self, batch, _):
        opt = self.optimizers()
        opt.zero_grad()
        noisy, original = batch
        y_hat = self.restored
        loss = torch.sum((y_hat - noisy) ** 2)
        tv_loss = self.tvl1_prox(self.restored)
        self.manual_backward(loss)
        opt.step()
        with torch.no_grad():
            optimizer = self.trainer.optimizers[0]
            self.restored.data = self.tvl1_prox.prox(
                self.restored.data, optimizer.param_groups[0]["lr"]
            )

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)


# Data Preparation
noisy_logo = proxtorch_logo + np.random.normal(
    loc=0, scale=0.1, size=proxtorch_logo.shape
)
dataset = TensorDataset(
    torch.tensor(noisy_logo).unsqueeze(0), torch.tensor(proxtorch_logo).unsqueeze(0)
)
loader = DataLoader(dataset, batch_size=1)

# Model Initialization
tv_l1_model = TVL1Restoration(alpha=0.2, l1_ratio=0.05)
tv_model = TVL1Restoration(alpha=0.2, l1_ratio=0.0)

# Training
trainer = pl.Trainer(max_epochs=50)
trainer.fit(tv_model, loader)
trainer = pl.Trainer(max_epochs=50)
trainer.fit(tv_l1_model, loader)


# Evaluation
def evaluate(model, label):
    model.eval()
    loss = torch.mean((model.restored - torch.tensor(proxtorch_logo).unsqueeze(0)) ** 2)
    print(f"{label} loss: {loss.item()}")


evaluate(tv_model, "TV")
evaluate(tv_l1_model, "TV-L1")

# Determine the global min and max across all images to set a consistent colorscale
global_min = 0
global_max = 1

# Visualization
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

# Original Image
ax[0].imshow(proxtorch_logo, cmap="gray", vmin=global_min, vmax=global_max)
ax[0].set_title("Original")

# Noisy Image
ax[1].imshow(noisy_logo, cmap="gray", vmin=global_min, vmax=global_max)
ax[1].set_title("Noisy")

# TV Restored Image
tv_restored_image = tv_model.restored.detach().numpy()
ax[2].imshow(tv_restored_image, cmap="gray", vmin=global_min, vmax=global_max)
ax[2].set_title("TV Restored")

# TV-L1 Restored Image
tv_l1_restored_image = tv_l1_model.restored.detach().numpy()
ax[3].imshow(tv_l1_restored_image, cmap="gray", vmin=global_min, vmax=global_max)
ax[3].set_title("TV-L1 Restored")

plt.tight_layout()
plt.show()
