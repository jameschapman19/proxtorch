import torch

from proxtorch.base import ProxOperator


class L0Ball(ProxOperator):
    r"""
    L0Prox ball proximal operator.

    Projects onto a vector with at most `s` non-zero elements.

    Attributes:
        s (int): Budget of non-zero elements.
    """

    def __init__(self, s: int):
        super().__init__()
        self.s = s

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Proximal operator for the L0Prox ball.

        Keeps the `s` largest elements in magnitude and sets the rest to zero.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal step size. Not used for L0Ball, but kept for API consistency.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        _, indices = torch.topk(torch.abs(x), self.s)
        result = torch.zeros_like(x)
        result[indices] = x[indices]
        return result

    def __call__(self, x: torch.Tensor) -> int:
        r"""
        Compute the distance to the L0Prox ball.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            int: Distance to the L0Prox ball.
        """
        non_zero_entries = torch.sum(x != 0).item()
        return max(0, non_zero_entries - self.s)