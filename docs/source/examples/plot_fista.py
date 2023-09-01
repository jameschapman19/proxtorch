"""
Lasso Regression with FISTA Optimization using PyTorch's Optimizer
===================================================================

...

Dependencies:
- `numpy`
- `torch`
- `matplotlib`
- `proxtorch`
- `sklearn.datasets`
"""
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from pytorch_lightning import seed_everything
from sklearn.datasets import make_regression
from torch.nn import Parameter

from proxtorch.operators import L1Prox

seed_everything(42)

# Create synthetic data
X, y, coef = make_regression(
    n_samples=100, n_features=20, noise=0.1, coef=True, random_state=42
)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Parameters
alpha = 0.1  # Regularization parameter for Lasso
lr = 0.01  # Learning rate
n_iter = 100  # Number of iterations
l1_prox = L1Prox(alpha=alpha)


def fista(X, y, l1_prox, lr, n_iter):
    theta = Parameter(torch.zeros(X.shape[1]))  # Initialize weights
    optimizer = optim.SGD([theta], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()  # Reset gradients
        # Forward pass: Compute predicted y
        y_pred = X @ theta

        # Compute loss
        loss = ((y_pred - y) ** 2).mean()

        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Optimizer step (gradient descent)
        optimizer.step()

        # Proximal operation
        with torch.no_grad():
            theta.data = l1_prox.prox(theta, lr)

    return theta


# Run FISTA
weights = fista(X, y, l1_prox, lr, n_iter)

# Plot non-zero coefficients
plt.stem(weights.detach().numpy(), label="FISTA")
plt.stem(coef, linefmt="r-", markerfmt="ro", use_line_collection=True, label="True")
plt.title("Lasso Coefficients with FISTA using PyTorch's Optimizer")
plt.legend()
plt.show()
