import torch
from proxtorch.base import ProxOperator


class L1Prox(ProxOperator):
    r"""
    L1Prox norm proximal operator.

    The L1Prox norm promotes sparsity in the tensor.

    Attributes:
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Soft-thresholding for the L1Prox norm.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal operator step size.

        Returns:
            torch.Tensor: Resultant tensor after soft-thresholding.
        """
        return torch.sign(x) * torch.clamp(torch.abs(x) - tau * self.alpha, min=0)

    def __call__(self, x: torch.Tensor) -> float:
        r"""
        Compute the L1Prox norm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float: The L1Prox norm value.
        """
        return torch.sum(torch.abs(x)) * self.alpha
