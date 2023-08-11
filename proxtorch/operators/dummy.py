import torch
from proxtorch.base import ProxOperator


class DummyProx(ProxOperator):
    r"""
    Dummy proximal operator that acts as an identity operation.

    This class provides methods for soft-thresholding consistent with L1Prox norm and computation of the L1Prox norm.
    """

    def __init__(self):
        super().__init__()

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Soft-thresholding for the L1Prox norm.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal operator step size.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        return x

    def __call__(self, x: torch.Tensor) -> float:
        r"""
        Compute the L1Prox norm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float: The L1Prox norm value.
        """
        return 0.0
