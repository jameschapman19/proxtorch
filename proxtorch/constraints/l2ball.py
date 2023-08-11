import torch

from proxtorch.base import Constraint


class L2Ball(Constraint):
    r"""
    Projection onto the L1Prox ball.

    Attributes:
        s (float): Radius of the L1Prox ball.
    """

    def __init__(self, s: float = 1.0):
        super().__init__()
        self.s = s

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Project x onto the L1Prox-ball of radius `s`.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal step size. Not used for L1Ball, but kept for API consistency.

        Returns:
            torch.Tensor: Resultant tensor after the projection.
        """
        # The logic provided is one of the ways to achieve this projection.
        if torch.norm(x, p=2) <= self.s:
            return x
        else:
            return x * self.s / torch.norm(x, p=2)

    def __call__(self, x: torch.Tensor) -> bool:
        r"""Check if the tensor satisfies the L1 constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if L1-norm of tensor is less than or equal to `s`, False otherwise.
        """
        return torch.norm(x, p=2) <= self.s
