import torch

from proxtorch.base import ProxOperator


class Dummy(ProxOperator):
    r"""
    Dummy proximal operator that acts as an identity operation.

    This class provides methods for soft-thresholding consistent with L1 norm and computation of the L1 norm.
    """

    def __init__(self):
        super().__init__()

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Soft-thresholding for the L1 norm.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal operator step size.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        return x
