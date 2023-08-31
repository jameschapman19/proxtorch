from math import sqrt

import torch
import torch.nn.functional as F
from proxtorch.operators.tvl1_3d import TVL1_3DProx


class TVL1_2DProx(TVL1_3DProx):
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = F.pad(x[1:, :] - x[:-1, :], (0, 0, 0, 1))
        grad_y = F.pad(x[:, 1:] - x[:, :-1], (0, 1, 0, 0))
        grad_l1 = self.l1_ratio * x

        return torch.stack([grad_x, grad_y, grad_l1], dim=0)

    def divergence(self, p: torch.Tensor) -> torch.Tensor:
        div_x = torch.zeros_like(p[-1])
        div_y = torch.zeros_like(p[-1])

        div_x[:-1].add_(p[0, :-1, :])
        div_y[:, :-1].add_(p[1, :, :-1])

        div_x[1:-1].sub_(p[0, :-2, :])
        div_y[:, 1:-1].sub_(p[1, :, :-2])

        div_x[-1].sub_(p[0, -2, :])
        div_y[:, -1].sub_(p[1, :, -2])

        return (div_x + div_y) * (1 - self.l1_ratio) - self.l1_ratio * p[-1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Total Variation (TV) for a given tensor x.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The TV of the tensor x.
        """
        # Check if self.shape is not None and x has different shape, then try to reshape
        if self.shape and x.shape != self.shape:
            x = x.reshape(self.shape)
        gradients = self.gradient(x)
        return self.tv_from_grad(gradients) * self.alpha

    @staticmethod
    def tv_from_grad(gradients: torch.Tensor) -> float:
        r"""
        Calculate the TV from gradients.

        Args:
            gradients (torch.Tensor): Gradient tensor.

        Returns:
            float: The TV value computed from the gradients.
        """
        grad_x, grad_y = gradients[0], gradients[1]
        return torch.sum(torch.sqrt(grad_x**2 + grad_y**2))
