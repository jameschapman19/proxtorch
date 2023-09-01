import torch

from proxtorch.base import ProxOperator


class L2Prox(ProxOperator):
    r"""
    L2Prox norm proximal operator.

    This class provides methods for soft-thresholding and computation of the L2Prox norm.

    Attributes:
        alpha (float): Regularization parameter.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Apply the L2 proximal operator.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal operator step size.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        return x / (1.0 + self.alpha * tau)

    def _smooth(self, x):
        return 0.5 * self.alpha * torch.linalg.norm(x.reshape(-1), 2) ** 2

    def _nonsmooth(self, x):
        return 0
