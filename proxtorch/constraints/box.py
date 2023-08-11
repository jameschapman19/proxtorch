# Box Constraint
import torch

from proxtorch.base import Constraint


class BoxConstraint(Constraint):
    def __init__(self, a: float = 0.0, b: float = 1.0):
        super().__init__()
        self.a = a
        self.b = b

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.a, max=self.b)

    def __call__(self, x: torch.Tensor) -> bool:
        return torch.all(x >= self.a) and torch.all(x <= self.b)
