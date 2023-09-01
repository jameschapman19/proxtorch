"""
Matrix Completion for Movie Recommendations using PyTorch Lightning
=====================================================================

Have you ever wondered how Netflix can accurately predict which movies you might like? One of the core challenges in recommendation systems is the task of 'Matrix Completion'. This problem arises when we have a massive user-item interaction matrix (like user-movie ratings) but only a few of these interactions are observed. Given this sparsity, the question becomes: how do we predict the unknown interactions?

In a perfect world, every user would rate every movie. But in reality, a user might only rate a handful of movies, leading to a sparse matrix. Enter the matrix completion problem.

Matrix completion thrives on an assumption that though the matrix is large, it can be approximated as low-rank. In our Netflix context, this means that while there are millions of users and thousands of movies, only a few factors (or 'latent features') influence a user's decision to like a movie. This could include aspects like genre preferences, favorite actors, or movie eras. When users rate movies, they inadvertently give hints about their preferences in this latent feature space. The task, then, is to uncover these latent features.

In this tutorial:

- We simulate a user-movie interaction matrix, where only a fraction of the entries are observed.
- We deploy a matrix completion algorithm to recover the missing entries, assuming the underlying matrix is of low-rank.
- By leveraging the power of PyTorch Lightning and the ProxTorch package, we illustrate how to learn and predict user preferences over movies.

By the end of this, you'll be equipped with the knowledge to build your recommendation system for any user-item interaction dataset.

Dependencies:
- `pytorch_lightning`
- `torch`
- `proxtorch`

Let's dive in!

"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from proxtorch.operators import TraceNormProx


# Define the Robust Matrix Completion model
class MatrixCompletion(pl.LightningModule):
    def __init__(self, input_shape, lambda_trace):
        super(MatrixCompletion, self).__init__()
        self.input_shape = input_shape
        self.B = torch.nn.Parameter(torch.randn(input_shape))
        # Proximal operator for trace norm
        self.trace_norm_prox = TraceNormProx(alpha=lambda_trace)

    def forward(self, x):
        return self.B

    def training_step(self, batch, batch_idx):
        Y_observed, mask = batch
        loss = 0.5 * torch.norm(mask * (Y_observed - self.forward(Y_observed)), "fro")
        self.log("train_loss", loss)
        tn_loss = self.trace_norm_prox(self.B.data)
        self.log("trace_norm", tn_loss)
        total_loss = loss + tn_loss
        self.log("total_loss", total_loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Proximal update for trace norm
        with torch.no_grad():
            tau = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.B.data = self.trace_norm_prox.prox(self.B.data, tau=tau)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.5)
        return optimizer


class MatrixCompletionDataset(Dataset):
    def __init__(self, num_users, num_movies, rank, mask_ratio=0.7):
        self.num_users = num_users
        self.num_movies = num_movies
        self.rank = rank

        # Generate a low-rank user-movie matrix
        user_pref = torch.rand((num_users, rank))
        movie_features = torch.rand((rank, num_movies))
        self.Y = user_pref @ movie_features

        # Generate a consistent mask for missing data
        self.mask = (
                torch.rand((num_users, num_movies)) > mask_ratio
        )  # If mask_ratio is 0.7, 30% of entries will be known
        self.known_entries = self.Y * self.mask

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.known_entries, self.mask  # Returns known entries and mask


# DataLoader
num_users = 100
num_movies = 10
latent_features = 2  # Number of latent features that influence a user's decision
observed_ratio = 0.1  # Only 20% of the user-movie interactions are observed

dataset = MatrixCompletionDataset(
    num_users=num_users,
    num_movies=num_movies,
    rank=latent_features,
    mask_ratio=1 - observed_ratio,
)
loader = DataLoader(dataset, batch_size=1)

# Initialize the model and trainer
model = MatrixCompletion(input_shape=(num_users, num_movies), lambda_trace=0.1)
trainer = pl.Trainer(max_epochs=500)
trainer.fit(model, loader)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))

# Prepare a version of the dataset matrix where missing entries are set to a value outside the colormap's range (e.g., -1)
matrix_display = np.where(dataset.known_entries == 0, np.nan, dataset.known_entries)

# 1. Original matrix with missing entries
axes[0].imshow(
    matrix_display, cmap="coolwarm", aspect="auto", vmin=0, vmax=dataset.Y.max()
)
axes[0].set_title("Original Matrix with Missing Entries")
axes[0].set_xlabel("Movies")
axes[0].set_ylabel("Users")

# 2. Completed matrix B but only at the locations where Y had values
# Prepare a version where only the filled values from B are shown and rest are set to -1
B_display = np.where(~np.isnan(matrix_display), model.B.detach().numpy(), np.nan)
axes[1].imshow(B_display, cmap="coolwarm", aspect="auto", vmin=0, vmax=dataset.Y.max())
axes[1].set_title("Filled Values in Matrix B")
axes[1].set_xlabel("Movies")
axes[1].set_ylabel("Users")

# 3. Completed matrix B but only at the locations where Y had missing values
# Prepare a version where only the filled values from B are shown and rest are set to -1
B_display = np.where(np.isnan(matrix_display), model.B.detach().numpy(), np.nan)
axes[2].imshow(B_display, cmap="coolwarm", aspect="auto", vmin=0, vmax=dataset.Y.max())
axes[2].set_title("Filled Values in Matrix B")
axes[2].set_xlabel("Movies")
axes[2].set_ylabel("Users")

# 4. The fully completed matrix B
axes[3].imshow(
    model.B.detach().numpy(),
    cmap="coolwarm",
    aspect="auto",
    vmin=0,
    vmax=dataset.Y.max(),
)
axes[3].set_title("Fully Completed Matrix B")
axes[3].set_xlabel("Movies")
axes[3].set_ylabel("Users")

# 5. The true matrix Y
axes[4].imshow(dataset.Y, cmap="coolwarm", aspect="auto", vmin=0, vmax=dataset.Y.max())
axes[4].set_title("True Matrix Y")
axes[4].set_xlabel("Movies")
axes[4].set_ylabel("Users")

plt.tight_layout()
plt.show()
