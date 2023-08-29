from proxtorch.base import Constraint
import torch


class TraceNormConstraint(Constraint):
    r"""
    Constraint for trace norm regularization.

    Attributes:
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: torch.Tensor) -> bool:
        r"""
        Check if the constraint is satisfied for the given tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if trace norm of x is less than or equal to s, False otherwise.
        """
        singular_values = torch.svd(x).S
        trace_norm = torch.sum(singular_values)
        return trace_norm <= self.alpha

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Projects the tensor onto the feasible set defined by the trace norm constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after projection.
        """
        u, s, v = torch.svd(x)
        # Clip singular values so that their sum doesn't exceed s
        cumulative_s = torch.cumsum(s, dim=0)
        k = (cumulative_s <= self.alpha).sum()
        if k > 0:
            s[k:] = 0  # Set singular values beyond the k-th value to zero
            scaling_factor = min(1, self.alpha / cumulative_s[k - 1])
            s[:k] *= scaling_factor
        else:
            s[:] = 0

        return u @ torch.diag(s) @ v.T


NuclearNormConstraint = TraceNormConstraint
