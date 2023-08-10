from torch import Tensor

from proxtorch.base import ProxOperator


class HuberProx(ProxOperator):
    r"""Proximal operator for the Huber penalty."""

    def __init__(self, alpha: float = 1.0, delta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta

    def prox(self, x: Tensor, tau: float) -> Tensor:
        r"""
        Apply the proximal operation for the Huber penalty.

        Args:
            x (Tensor): Input tensor.
            tau (float): Proximal step size.

        Returns:
            Tensor: Result after applying the Huber operation.
        """
        cond1 = x.abs() <= self.delta
        cond2 = x > self.delta
        cond3 = x < -self.delta
        x[cond1] = x[cond1] / (1.0 + tau * self.alpha)
        x[cond2] = x[cond2] - tau * self.alpha * self.delta
        x[cond3] = x[cond3] + tau * self.alpha * self.delta
        return x

    def __call__(self, x: Tensor) -> float:
        r"""Compute the Huber penalty for a given input tensor."""
        cond1 = x.abs() <= self.delta
        cond2 = ~cond1
        return self.alpha * (
            0.5 * (x[cond1] ** 2).sum()
            + self.delta * x[cond2].abs().sum()
            - 0.5 * self.delta**2 * cond2.float().sum()
        )
