from proxtorch.base import Constraint
import torch


class FrobeniusConstraint(Constraint):
    r"""
    Constraint for the Frobenius norm.

    Attributes:
        s (float): Regularization strength.
    """

    def __init__(self, s: float = 1.0):
        super().__init__()
        self.s = s

    def __call__(self, x: torch.Tensor) -> bool:
        r"""
        Check if the constraint is satisfied for the given tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if Frobenius norm of x is less than or equal to s, False otherwise.
        """
        frobenius_norm = torch.norm(x, p="fro")
        return frobenius_norm <= self.s

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Projects the tensor onto the feasible set defined by the Frobenius norm constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after projection.
        """
        frobenius_norm = torch.norm(x, p="fro")
        if frobenius_norm > self.s:
            return self.s * (x / frobenius_norm)
        return x
