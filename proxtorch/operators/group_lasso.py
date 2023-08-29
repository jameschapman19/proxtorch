import torch

from proxtorch.base import ProxOperator


class GroupLassoProx(ProxOperator):
    def __init__(self, alpha: float, group_sizes: list):
        r"""
        Initialize the GroupLassoProx operator.

        Args:
            alpha (float): Group Lasso regularization parameter.
            group_sizes (list): List containing the sizes of each group.
        """
        super(GroupLassoProx, self).__init__()
        self.alpha = alpha
        self.group_sizes = group_sizes

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""Proximal mapping of the Group Lasso operator.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal parameter.

        Returns:
            torch.Tensor: Result of the proximal mapping.
        """
        start = 0
        result = torch.zeros_like(x)

        for size in self.group_sizes:
            end = start + size
            group_norm = torch.norm(x[start:end], p=2)
            if group_norm > 0:
                multiplier = max(1 - self.alpha * tau / group_norm, 0)
                result[start:end] = multiplier * x[start:end]
            start = end

        return result

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r"""Function call to evaluate the Group Lasso penalty.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of the Group Lasso penalty.
        """
        penalty = 0.0
        start = 0

        for size in self.group_sizes:
            end = start + size
            penalty += torch.norm(x[start:end], p=2)
            start = end

        return self.alpha * penalty
