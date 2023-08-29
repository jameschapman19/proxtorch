import torch

from .l1 import L1Prox
from .tv_3d import TV_3DProx
from proxtorch.base import ProxOperator


class TVL1_3DProx(ProxOperator):
    def __init__(
        self,
        alpha_l1: float,
        alpha_tv: float,
        shape=None,
        max_iter: int = 50,
        tol: float = 1e-7,
    ) -> None:
        """
        Initialize the 3D Total Variation L1 proximal operator.

        Args:
            alpha_l1 (float): L1 regularization strength.
            alpha_tv (float): TV regularization strength.
            shape (tuple, optional): Desired shape for the input tensor. Defaults to None.
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 50.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-2.
        """
        super(TVL1_3DProx, self).__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_l1 = alpha_l1
        self.alpha_tv = alpha_tv
        self.tv = TV_3DProx(alpha_tv, shape, max_iter)
        self.l1 = L1Prox(alpha_l1)

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Iterative algorithm to compute the Douglas-Rachford splitting for the sum of L1 and TV functions.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Step size.

        Returns:
            torch.Tensor: Tensor after applying the L1 and TV proximal operations.
        """
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)
        x_next = x.clone()

        lambda_relax = 1.0  # Relaxation parameter (can be tuned)

        # Step 1: Proximal of TV
        y.copy_(self.tv.prox(x_next, tau))

        # Step 2: Proximal of L1
        z.copy_(self.l1.prox(2 * y - x_next, tau))

        # check that z is sparse
        sparse = torch.sum(z != 0) < torch.numel(z)
        sparsity = torch.sum(z != 0) / torch.numel(z)

        # Step 3: Update x using relaxation
        x_next += lambda_relax * (z - y)

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
