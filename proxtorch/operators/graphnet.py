import torch

from proxtorch.operators.l1 import L1
from proxtorch.operators.tvl1_2d import TVL1_2D
from proxtorch.operators.tvl1_3d import TVL1_3D


class GraphNet3D(TVL1_3D):
    def __init__(self, alpha, l1_ratio):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio)
        self.l1_prox = L1(alpha * l1_ratio)

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        return self.l1_prox.prox(x, tau)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        # The last channel is the for the l1 norm
        grad = self.gradient(x)[:-1] / (1 - self.l1_ratio)
        # sum of squares of the gradients
        norm = torch.sum(grad**2)
        return 0.5 * norm * self.alpha * (1 - self.l1_ratio)

    def _nonsmooth(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_prox(x)
        return l1


class GraphNet2D(GraphNet3D, TVL1_2D):
    pass
