from math import sqrt

import torch
import torch.nn.functional as F
from proxtorch.operators.tvl1_3d import TVL1_3DProx


class TVL1_2DProx(TVL1_3DProx):
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
        gradients = self.gradient(x)
        return self.tvl1_from_grad(gradients) * self.alpha
