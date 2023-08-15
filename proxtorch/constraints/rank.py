from proxtorch.base import Constraint
import torch


class RankConstraint(Constraint):
    r"""
    Constraint for rank regularization.

    Attributes:
        max_rank (int): Maximum allowable rank.
    """

    def __init__(self, max_rank: int):
        super().__init__()
        self.max_rank = max_rank

    def __call__(self, x: torch.Tensor) -> bool:
        r"""
        Check if the constraint is satisfied for the given tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if rank of x is less than or equal to max_rank, False otherwise.
        """
        singular_values = torch.svd(x).S
        rank = (
            (singular_values > 1e-5).sum().item()
        )  # Count non-negligible singular values to determine rank
        return rank <= self.max_rank

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Projects the tensor onto the feasible set defined by the rank constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after projection.
        """
        u, s, v = torch.svd(x)

        # Set singular values beyond the max_rank-th value to zero
        s[self.max_rank :] = 0

        return u @ torch.diag(s) @ v.T
