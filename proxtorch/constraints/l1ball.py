import torch
from proxtorch.base import Constraint
import torch
from proxtorch.base import ProxOperator


class L1Ball(Constraint):
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
        if torch.norm(x, p=1) <= self.s:
            return x
        else:
            u, _ = torch.sort(torch.abs(x), descending=True)
            cssv = torch.cumsum(u, dim=0) - self.s
            idx = torch.arange(1, x.numel() + 1, device=x.device)
            cond = u - cssv / idx > 0
            rho = idx[cond][-1]
            theta = cssv[cond][-1] / float(rho)
            return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0)

    def __call__(self, x: torch.Tensor) -> bool:
        r"""Check if the tensor satisfies the L1 constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if L1-norm of tensor is less than or equal to `s`, False otherwise.
        """
        return torch.norm(x, p=1) <= self.s
