import torch

from proxtorch.operators import L1Prox, TVL1_3DProx, TVL1_2DProx


class GraphNet3DProx(TVL1_3DProx):
    def __init__(self, alpha, l1_ratio, verbose=False):
        super().__init__(alpha, verbose)
        self.l1_prox = L1Prox(alpha * l1_ratio)

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        return self.l1_prox.prox(x, tau)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        grad = self.gradient(x)
        # norm of the gradient
        norm = torch.norm(grad) ** 2
        return norm

    def _nonsmooth(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_prox(x)
        return l1


class GraphNet2DProx(GraphNet3DProx, TVL1_2DProx):
    pass
