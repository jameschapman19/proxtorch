"""
Lasso Regression with FISTA
===========================

In this example, we'll implement Lasso regression using the FISTA algorithm,
leveraging `proxtorch` for the proximal operations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from proxtorch.operators import L1Prox
from sklearn.datasets import make_regression

# Create synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Parameters
alpha = 0.1  # Regularization parameter for Lasso
lr = 0.01  # Learning rate
n_iter = 100  # Number of iterations
l1_prox = L1Prox(sigma=alpha)


def fista(X, y, l1_prox, lr, n_iter):
    theta = torch.zeros(X.shape[1], requires_grad=True)  # Initialize weights
    t = 1
    z = theta.clone()
    y_old = theta.clone()

    for _ in range(n_iter):
        # Gradient descent step
        y_pred = X @ z
        loss = ((y_pred - y) ** 2).mean() + l1_prox(z)
        loss.backward()

        with torch.no_grad():
            theta_new = z - lr * z.grad
            # Proximal operation
            theta_new = l1_prox.prox(theta_new, lr)

            # Update t and y for next iteration
            t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2
            y_new = theta_new + (t - 1) / t_new * (theta_new - theta)

            z, y_old, t = theta_new, y_new, t_new
            z.grad.zero_()  # Reset gradients

    return z


# Run FISTA
weights = fista(X, y, l1_prox, lr, n_iter)

# Plot non-zero coefficients
plt.stem(weights.detach().numpy())
plt.title("Lasso Coefficients with FISTA")
plt.show()
