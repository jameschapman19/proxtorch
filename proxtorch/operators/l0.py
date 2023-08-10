from proxtorch.base import ProxOperator


import torch


class L0Prox(ProxOperator):
    r"""
    L0Prox norm proximal operator.

    The L0Prox "norm" counts the number of non-zero entries in a tensor.
    This proximal operator can be useful for promoting sparsity.

    Attributes:
        threshold (float): Threshold value for the proximal operator.
    """

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Apply the proximal operator for the L0Prox pseudo-norm.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal operator step size.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        return torch.where(torch.abs(x) > tau * self.threshold, x, torch.zeros_like(x))

    def __call__(self, x: torch.Tensor) -> float:
        r"""
        Count the number of non-zero entries in the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float: Number of non-zero entries.
        """
        return torch.sum(torch.abs(x) > self.threshold).float()
