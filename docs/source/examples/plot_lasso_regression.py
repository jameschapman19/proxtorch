"""
Comparing Custom Lasso Regression with scikit-learn's Implementation
====================================================================

This script demonstrates how to implement and train a Lasso regression model using
PyTorch Lightning and compares it with scikit-learn's built-in Lasso regression.

Lasso regression is a linear regression variant that incorporates L1 regularization,
leading to sparse weight vectors. In other words, many weights become exactly zero,
allowing for simpler and more interpretable models.

In this example:

- A custom `LassoRegression` class is defined using PyTorch Lightning, which
  incorporates the L1 regularization via a proximal gradient method.

- Synthetic data is generated where the ground truth weights are partly set to zero
  to mimic sparse structures.

- Both the custom Lasso model and scikit-learn's Lasso model are trained on the
  synthetic data.

- The models' performances are compared based on their mean squared error (MSE) on
  a test set.

- The predicted values of both models are visualized against the true values for
  a comparative look.

- Finally, the learned weights from both models are compared with the true weights
  through bar plots.

By the end of this script, you should have insights into how Lasso regression can
be implemented in PyTorch Lightning and how its performance matches up against
traditional implementations in packages like scikit-learn.

Dependencies:
- `torch`
- `torch.nn`
- `torch.optim`
- `sklearn.model_selection`
- `sklearn.linear_model`
- `numpy`
- `matplotlib`
- `pytorch_lightning`
- `proxtorch`
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from proxtorch.operators import L1Prox


class LassoRegression(pl.LightningModule):
    def __init__(self, input_size, lasso_param):
        super(LassoRegression, self).__init__()
        self.input_size = input_size
        self.lasso_param = lasso_param
        self.linear = nn.Linear(input_size, 1)
        self.l1_prox = L1Prox(sigma=lasso_param)

    def forward(self, x):
        return self.linear(x)

    def proximal_step(self, w, parameter):
        return self.l1_prox.prox(w, parameter)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()
        loss = nn.MSELoss()(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad:
                    param.data = self.proximal_step(
                        param.data, self.trainer.optimizers[0].param_groups[0]["lr"]
                    )

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        return optimizer


# Generate synthetic data
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
w_true = np.random.randn(n_features)
# make 50% of the weights zero
w_true[np.random.choice(range(n_features), int(n_features * 0.5), replace=False)] = 0
y = X.dot(w_true) + np.random.normal(0, 0.1, n_samples)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Train the custom Lasso regression model
lasso_param = 0.1
lasso_model = LassoRegression(input_size=n_features, lasso_param=lasso_param)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(lasso_model, DataLoader(TensorDataset(x_train_tensor, y_train_tensor)))

# Train scikit-learn's Lasso regression model
sklearn_lasso = Lasso(alpha=lasso_param)
sklearn_lasso.fit(X_train, y_train)

# Compare the models' performance using MSE on the test set
with torch.no_grad():
    y_pred_custom = lasso_model(x_test_tensor).numpy().flatten()
    mse_custom = np.mean((y_pred_custom - y_test) ** 2)

    y_pred_sklearn = sklearn_lasso.predict(X_test)
    mse_sklearn = np.mean((y_pred_sklearn - y_test) ** 2)

print(f"Custom Lasso Test MSE: {mse_custom:.4f}")
print(f"scikit-learn Lasso Test MSE: {mse_sklearn:.4f}")

# Create a plot comparing predictions
plt.figure()
plt.scatter(y_test, y_pred_custom, label="Custom Lasso")
plt.scatter(y_test, y_pred_sklearn, label="scikit-learn Lasso")
plt.plot(y_test, y_test, color="black", linestyle="--", label="Ground Truth")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Lasso Regression Comparison on Test Data")
plt.legend()
plt.show()

# Compare weights
w_custom = lasso_model.linear.weight.data.numpy().flatten()
w_sklearn = sklearn_lasso.coef_

plt.figure(figsize=(15, 5))
x = np.arange(n_features)
plt.bar(x - 0.2, w_true, 0.2, label="True Weights", align="center", alpha=0.8)
plt.bar(x, w_custom, 0.2, label="Custom Lasso Weights", align="center", alpha=0.8)
plt.bar(
    x + 0.2, w_sklearn, 0.2, label="sklearn Lasso Weights", align="center", alpha=0.8
)
plt.xlabel("Features")
plt.ylabel("Weights")
plt.title("Weight Comparison")
plt.legend()
plt.show()
