import torch

from proxtorch.operators import L1Prox, TVL1_3DProx, TVL1_2DProx


class GraphNet3DProx(TVL1_3DProx):
    def __init__(self, alpha, l1_ratio):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio)
        self.l1_prox = L1Prox(alpha * l1_ratio)

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        return self.l1_prox.prox(x, tau)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        # The last channel is the for the l1 norm
        grad = self.gradient(x)[:-1]
        # norm of the gradient
        norm = torch.norm(grad) ** 2
        return 0.5 * norm * self.alpha * (1 - self.l1_ratio)

    def _nonsmooth(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_prox(x)
        return l1


class GraphNet2DProx(GraphNet3DProx, TVL1_2DProx):
    pass
