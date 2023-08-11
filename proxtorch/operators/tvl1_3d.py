import torch

from .l1 import L1Prox
from .tv_3d import TV_3DProx
from proxtorch.base import ProxOperator


class TVL1_3DProx(ProxOperator):
    def __init__(
        self, sigma_l1: float, sigma_tv: float, max_iter: int = 50, tol: float = 1e-4
    ) -> None:
        """
        Initialize the 3D Total Variation L1 proximal operator.

        Args:
            sigma_l1 (float): L1 regularization strength.
            sigma_tv (float): TV regularization strength.
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 50.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-2.
        """
        super(TVL1_3DProx, self).__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.sigma_l1 = sigma_l1
        self.sigma_tv = sigma_tv
        self.tv = TV_3DProx(sigma_tv, max_iter)
        self.l1 = L1Prox(sigma_l1)

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Iterative algorithm to compute the proximal mapping of the tensor using L1 and TV proximals.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Step size.

        Returns:
            torch.Tensor: Tensor after applying the L1 and TV proximal operations.
        """
        z = x.clone()
        for k in range(self.max_iter):
            y = self.l1.prox(z, tau)  # Prox of L1Prox
            x_next = self.tv.prox(2 * y - z, tau)  # Prox of TV
            z = z + tau * (x_next - y)  # Update z

            # Termination criterion: Check if the change in x is below a threshold
            diff = (torch.norm(x_next - x) / (torch.norm(x) + 1e-10)).item()
            if diff < self.tol:
                break

            x = x_next
        self.last_call_iter = k
        self.last_call_diff = diff
        return x_next

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the combined L1 and TV for a given tensor x.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Combined L1 and TV of the tensor x.
        """
        return self.l1(x) + self.tv(x)
