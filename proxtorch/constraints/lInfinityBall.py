import torch

from proxtorch.base import Constraint


class LInfinityBall(Constraint):
    r"""
    Projection onto the LInfinity ball of radius `s`.

    Attributes:
        s (float): Radius of the LInfinity ball.
    """

    def __init__(self, s: float = 1.0):
        super().__init__()
        self.s = s

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Project x onto the LInfinity-ball of radius `s`.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal step size. Not used for LInfinityBall, but kept for API consistency.

        Returns:
            torch.Tensor: Resultant tensor after the projection.
        """
        return torch.clamp(x, min=-self.s, max=self.s)

    def __call__(self, x: torch.Tensor) -> bool:
        r"""Check if the tensor satisfies the LInfinity constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if the infinity-norm of tensor is less than or equal to `s`, False otherwise.
        """
        return torch.max(torch.abs(x)) <= self.s
