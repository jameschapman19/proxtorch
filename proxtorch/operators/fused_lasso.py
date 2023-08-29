import torch
from torch import Tensor

from proxtorch.base import ProxOperator


class FusedLassoProx(ProxOperator):
    r"""Proximal operator for the 1D Fused Lasso."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def prox(self, x: Tensor, tau: float) -> Tensor:
        r"""
        Apply the proximal operation for the 1D fused lasso.
        Uses a simple soft-thresholding approach.

        Args:
            x (Tensor): Input tensor.
            tau (float): Proximal step size.

        Returns:
            Tensor: Result after applying the fused lasso operation.

        Note:
            More efficient algorithms exist for larger-scale problems.
        """
        diff = x[:-1] - x[1:]
        threshold = self.alpha * tau
        diff = torch.sign(diff) * torch.clamp(torch.abs(diff) - threshold, min=0)
        result = torch.zeros_like(x)
        result[0] = x[0] + diff[0]
        for i in range(1, len(x)):
            result[i] = result[i - 1] - diff[i - 1]
        return result

    def __call__(self, x: Tensor) -> float:
        r"""Compute the Fused Lasso objective for a given input tensor."""
        return self.alpha * torch.sum(torch.abs(x[:-1] - x[1:]))
