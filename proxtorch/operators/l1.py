import torch

from proxtorch.base import ProxOperator


class L1(ProxOperator):
    r"""
    L1 norm proximal operator.

    The L1 norm promotes sparsity in the tensor.

    Attributes:
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Soft-thresholding for the L1 norm.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal operator step size.

        Returns:
            torch.Tensor: Resultant tensor after soft-thresholding.
        """
        return torch.sign(x) * torch.clamp(torch.abs(x) - tau * self.alpha, min=0)

    def _nonsmooth(self, x):
        return self.alpha * torch.linalg.norm(x.reshape(-1), ord=1)
