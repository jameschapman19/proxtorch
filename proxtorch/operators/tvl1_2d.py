import torch

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
