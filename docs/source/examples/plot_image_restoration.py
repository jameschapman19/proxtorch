"""
Image Restoration: ProxTorch Logo via TV and TV-L1 Regularization
==================================================================

Using ProxTorch's TV_2DProx and TVL1_2DProx operators, this example demonstrates image restoration
of a noisy ProxTorch logo through TV and TV-L1 regularization.

Dependencies:
- torch, torch.nn, torch.optim
- numpy
- matplotlib
- pytorch_lightning
- proxtorch

"""

import numpy as np
import torch
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset
from proxtorch.operators import TV_2DProx, TVL1_2DProx

# Set seed
seed_everything(42)

# Load ProxTorch Logo as jpg then convert to grayscale numpy array
proxtorch_logo = plt.imread("../proxtorch-logo.jpg")
proxtorch_logo = 1 - np.mean(proxtorch_logo, axis=2)
# Normalize to [0, 1]
proxtorch_logo = (proxtorch_logo - np.min(proxtorch_logo)) / (
    np.max(proxtorch_logo) - np.min(proxtorch_logo)
)


class TVL1Restoration(pl.LightningModule):
    def __init__(self, lasso_param, tv_param):
        super().__init__()
        self.restored = torch.nn.Parameter(torch.zeros(proxtorch_logo.shape))
        self.tvl1_prox = TVL1_2DProx(alpha_l1=lasso_param, alpha_tv=tv_param)

    def forward(self, x):
        return self.restored

    def training_step(self, batch, _):
        noisy, original = batch
        y_hat = self.restored
        loss = torch.mean((y_hat - noisy) ** 2)
        self.log("fidelity_loss", loss)
        self.log("tvl1_loss", self.tvl1_prox(y_hat))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

    def on_train_batch_end(self, _, __, batch_idx: int):
        with torch.no_grad():
            optimizer = self.trainer.optimizers[0]
            self.restored.data = self.tvl1_prox.prox(
                self.restored.data, optimizer.param_groups[0]["lr"]
            )


class TVRestoration(pl.LightningModule):
    def __init__(self, tv_param):
        super().__init__()
        self.restored = torch.nn.Parameter(torch.zeros(proxtorch_logo.shape))
        self.tv_prox = TV_2DProx(alpha=tv_param)

    def forward(self, x):
        return self.restored

    def training_step(self, batch, _):
        noisy, original = batch
        y_hat = self.restored
        loss = torch.mean((y_hat - noisy) ** 2)
        self.log("fidelity_loss", loss)
        self.log("tv_loss", self.tv_prox(y_hat))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

    def on_train_batch_end(self, _, __, batch_idx: int):
        with torch.no_grad():
            optimizer = self.trainer.optimizers[0]
            self.restored.data = self.tv_prox.prox(
                self.restored.data, optimizer.param_groups[0]["lr"]
            )


# Data Preparation
noisy_logo = proxtorch_logo + np.random.normal(
    loc=0, scale=0.5, size=proxtorch_logo.shape
)
dataset = TensorDataset(
    torch.tensor(noisy_logo).unsqueeze(0), torch.tensor(proxtorch_logo).unsqueeze(0)
)
loader = DataLoader(dataset, batch_size=1)

# Model Initialization
tv_l1_model = TVL1Restoration(lasso_param=0.1, tv_param=0.8)
tv_model = TVRestoration(tv_param=0.8)

# Training
trainer = pl.Trainer(max_epochs=100)
trainer.fit(tv_model, loader)
trainer = pl.Trainer(max_epochs=100)
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
