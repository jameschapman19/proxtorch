from torch import Tensor

from proxtorch.base import ProxOperator
import torch


class GroupLassoProx(ProxOperator):
    r"""Proximal operator for the Group Lasso."""

    def __init__(self, alpha: float = 1.0, groups: list = None):
        super().__init__()
        self.alpha = alpha
        self.groups = groups

    def prox(self, x: Tensor, tau: float) -> Tensor:
        r"""
        Apply the proximal operation for Group Lasso.

        Args:
            x (Tensor): Input tensor.
            tau (float): Proximal step size.

        Returns:
            Tensor: Result after applying the group lasso operation.
        """
        if self.groups is None:
            return x
        for group in self.groups:
            norm_group = torch.norm(x[group])
            x[group] = (
                torch.clamp(norm_group - self.alpha * tau, min=0)
                * x[group]
                / (norm_group + 1e-10)
            )
        return x

    def __call__(self, x: Tensor) -> float:
        r"""Compute the Group Lasso penalty for a given input tensor."""
        if self.groups is None:
            return 0.0
        return self.alpha * sum(torch.norm(x[group]) for group in self.groups)

