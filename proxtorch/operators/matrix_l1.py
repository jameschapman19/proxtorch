import torch
from torch.nn import Parameter
from proxtorch.base import ProxOperator


class MatrixL1(ProxOperator):
    r"""
    L1 norm proximal operator.

    The L1 norm promotes sparsity in the tensor.

    Attributes:
        alpha (float): Regularization strength.
    """

    def __init__(self, alpha: float = 1.0, matrix=None):
        super().__init__()
        self.alpha = alpha
        self.matrix = Parameter(torch.Tensor(matrix), requires_grad=False)
        self.inv_matrix = Parameter(torch.linalg.inv(torch.Tensor(matrix)), requires_grad=False)

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        Soft-thresholding for the L1 norm.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal operator step size.

        Returns:
            torch.Tensor: Resultant tensor after soft-thresholding.
        """
        mat_x = self.matrix@ x
        soft_thresholded = torch.sign(mat_x) * torch.clamp(torch.abs(mat_x) - tau * self.alpha, min=0)
        return self.inv_matrix@soft_thresholded

    def _nonsmooth(self, x):
        mat_x = torch.matmul(self.matrix, x)
        return self.alpha * torch.linalg.norm(mat_x.reshape(-1), ord=1)
