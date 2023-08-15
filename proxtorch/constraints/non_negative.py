from torch import Tensor

from proxtorch.base import Constraint
import torch


class NonNegativeConstraint(Constraint):
    r"""Proximal operator for the non-negative constraint."""

    def prox(self, x: Tensor) -> Tensor:
        r"""
        Apply the proximal operation for non-negative constraint.

        Args:
            x (Tensor): Input tensor.
            tau (float): Proximal step size (not used here, but kept for consistency).

        Returns:
            Tensor: Result after applying the non-negative constraint.
        """
        return torch.clamp(x, min=0)

    def __call__(self, x: torch.Tensor) -> bool:
        r"""Check if the tensor satisfies the non-negative constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if all elements of tensor are non-negative, False otherwise.
        """
        return torch.all(x >= 0)
