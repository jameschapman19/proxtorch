from proxtorch.base import ProxOperator
import torch


class ElasticNetProx(ProxOperator):
    r"""
    Elastic Net proximal operator.

    Combines both L1 and L2 penalties for regularization.

    Attributes:
        alpha (float): Regularization strength.
        l1_ratio (float): Proportion of L1 regularization.
        l2_ratio (float): Proportion of L2 regularization.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.l2_ratio = 1.0 - l1_ratio

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Proximal operator for the Elastic Net regularization.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal step size.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        return (
            torch.sign(x)
            * torch.clamp(torch.abs(x) - self.alpha * self.l1_ratio * tau, min=0)
            / (1.0 + self.alpha * self.l2_ratio * tau)
        )

    def __call__(self, x: torch.Tensor) -> float:
        r"""
        Compute the combined L1 and L2 regularizations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float: Elastic Net regularization term.
        """
        l1_term = torch.norm(x, 1)
        l2_term = 0.5 * torch.norm(x) ** 2
        return self.alpha * (self.l1_ratio * l1_term + self.l2_ratio * l2_term)
