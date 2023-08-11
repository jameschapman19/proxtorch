from proxtorch.base import ProxOperator
import torch


class TraceNormProx(ProxOperator):
    r"""
    Proximal operator for the trace norm regularization.

    Attributes:
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Proximal operator for the trace norm regularization.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal step size.

        Returns:
            torch.Tensor: Resultant tensor after applying the proximal operator.
        """
        u, s, v = torch.svd(x)
        s = torch.clamp(s - self.alpha * tau, min=0)
        return u @ torch.diag(s) @ v.T

    def __call__(self, x: torch.Tensor) -> float:
        r"""
        Compute the trace norm regularization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float: Trace norm regularization term.
        """
        return self.alpha * torch.sum(torch.svd(x).S)


NuclearNormProx = TraceNormProx
