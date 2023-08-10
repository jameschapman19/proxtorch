from torch import Tensor

from proxtorch.base import ProxOperator
import torch


class NonNegativeProx(ProxOperator):
    r"""Proximal operator for the non-negative constraint."""

    def prox(self, x: Tensor, tau: float) -> Tensor:
        r"""
        Apply the proximal operation for non-negative constraint.

        Args:
            x (Tensor): Input tensor.
            tau (float): Proximal step size (not used here, but kept for consistency).

        Returns:
            Tensor: Result after applying the non-negative constraint.
        """
        return torch.clamp(x, min=0)

    def __call__(self, x: Tensor) -> float:
        r"""Compute the penalty. Here, no explicit penalty, only a constraint."""
        return 0.0
