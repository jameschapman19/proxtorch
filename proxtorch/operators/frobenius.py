from proxtorch.base import ProxOperator
import torch


class FrobeniusProx(ProxOperator):
    r"""
    Proximal operator for the Frobenius norm regularization.

    Attributes:
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Proximal operator for the Frobenius norm regularization.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal step size.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        return x / (1.0 + self.alpha * tau)

    def __call__(self, x: torch.Tensor) -> float:
        r"""
        Compute the Frobenius norm regularization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float: Frobenius norm regularization term.
        """
        return 0.5 * self.alpha * torch.norm(x, p="fro") ** 2
