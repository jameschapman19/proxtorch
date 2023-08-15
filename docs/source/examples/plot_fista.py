"""
Lasso Regression with FISTA Optimization
========================================

This script demonstrates how to implement Lasso regression using the
Fast Iterative Shrinkage-Thresholding Algorithm (FISTA). Lasso regression
is a method in linear regression that incorporates L1 regularization,
leading to a sparser solution where many coefficients are set to zero.

By using FISTA, an accelerated gradient-based optimization technique, we
can achieve faster convergence in solving the Lasso problem compared to
standard gradient descent methods.

Key Highlights:
- Leverages the `proxtorch` library for efficient proximal operations.
- Creates synthetic data using `sklearn.datasets.make_regression`.
- Defines and runs the FISTA algorithm for Lasso regression.
- Visualizes the non-zero coefficients of the learned Lasso model.

Dependencies:
- `numpy`
- `torch`
- `matplotlib`
- `proxtorch`
- `sklearn.datasets`
"""
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_regression

from proxtorch.operators import L1Prox

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
