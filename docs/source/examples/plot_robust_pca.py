"""
Robust Principal Component Analysis with PyTorch Lightning
===========================================================

This script demonstrates how to perform Robust Principal Component Analysis (RPCA) using
PyTorch Lightning. RPCA decomposes a matrix into two components:

1. A low-rank matrix that captures the global structure.
2. A sparse matrix that identifies the sparse errors.

The goal of RPCA is to find the best low-rank and sparse matrices that, when combined, closely
approximate the original matrix.

In this example:

- A custom `RobustPCA` class is defined using PyTorch Lightning, which learns the low-rank and sparse matrices.
- A `RandomMatrixDataset` class is designed to generate synthetic matrices composed of a true low-rank matrix and a true sparse matrix.
- The model is trained to approximate these matrices.
- The true and learned matrices are visualized for comparison.

By the end of this script, you will have a clear idea of how to implement and visualize RPCA using PyTorch Lightning.

Dependencies:
- `pytorch_lightning`
- `torch`
- `matplotlib`
- `proxtorch`
"""
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset

from proxtorch.operators import L1, TraceNorm

seed_everything(42)


class RobustPCA(pl.LightningModule):
    def __init__(self, input_shape, sigma_tn, sigma_l1):
        super(RobustPCA, self).__init__()
        self.input_shape = input_shape
        self.L = torch.nn.Parameter(torch.zeros(input_shape))
        self.S = torch.nn.Parameter(torch.zeros(input_shape))
        # Proximal operators
        self.trace_norm_prox = TraceNorm(alpha=sigma_tn)
        self.l1_prox = L1(alpha=sigma_l1)

    def forward(self, x):
        return self.L + self.S

    def training_step(self, batch, batch_idx):
        M = batch[0]
        # change lr to 1/(maximum absolute value of M)
        self.trainer.optimizers[0].param_groups[0]["lr"] = 1 / torch.abs(M).max()
        loss = torch.norm(M - self.forward(M), "fro")
        self.log("train_loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Proximal updates
        with torch.no_grad():
            self.L.data = self.trace_norm_prox.prox(
                self.L.data, tau=self.trainer.optimizers[0].param_groups[0]["lr"]
            )
            self.S.data = self.l1_prox.prox(
                self.S.data, tau=self.trainer.optimizers[0].param_groups[0]["lr"]
            )
        self.log("trace_norm", self.trace_norm_prox(self.L.data))
        self.log("l1_norm", self.l1_prox(self.S.data))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        return optimizer


# Define the Dataset
class RandomMatrixDataset(Dataset):
    def __init__(self, samples, features, rank):
        self.samples = samples
        self.features = features
        self.data = []

        # Low rank component
        low_rank = torch.randn((features, rank)) @ torch.randn((rank, features))
        # Sparse component
        sparse = torch.Tensor(features, features).random_(0, 2) - 1  # -1 or 1 values
        mask = torch.rand((features, features)) > 0.95  # Only 5% entries have noise
        sparse = sparse * mask
        self.L = low_rank
        self.S = sparse
        self.M = self.L + self.S

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return self.M[idx], 0  # The second item (label) is just a placeholder


n_samples = 50
features = 50
rank = 5
# DataLoader
train_dataset = RandomMatrixDataset(n_samples, features, rank)
train_loader = DataLoader(train_dataset, batch_size=n_samples, shuffle=False)

# Initialize the model and trainer
model = RobustPCA(input_shape=(n_samples, features), sigma_tn=0.05, sigma_l1=0.02)
trainer = pl.Trainer(max_epochs=1000)
trainer.fit(model, train_loader)

# Visualization of the true and learned low-rank and sparse components
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# True Components
axes[0, 0].imshow(train_dataset.L, cmap="gray", aspect="auto")
axes[0, 0].set_title("True Low Rank Component")
axes[0, 0].axis("off")

axes[0, 1].imshow(train_dataset.S, cmap="gray", aspect="auto")
axes[0, 1].set_title("True Sparse Component")
axes[0, 1].axis("off")

axes[0, 2].imshow(train_dataset.M, cmap="gray", aspect="auto")
axes[0, 2].set_title("Original Matrix (M)")
axes[0, 2].axis("off")

# Learned Components
axes[1, 0].imshow(
    model.L.detach().numpy() @ model.L.T.detach().numpy(),
    cmap="gray",
    aspect="auto",
)
axes[1, 0].set_title("Learned Low Rank Component")
axes[1, 0].axis("off")

axes[1, 1].imshow(model.S.detach().numpy(), cmap="gray", aspect="auto")
axes[1, 1].set_title("Learned Sparse Component")
axes[1, 1].axis("off")

model_M = model.L.detach().numpy() + model.S.detach().numpy()
axes[1, 2].imshow(
    model_M,
    cmap="gray",
    aspect="auto",
)
axes[1, 2].set_title("Reconstructed Matrix")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()
