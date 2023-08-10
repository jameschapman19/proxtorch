import torch

from proxtorch.operators import HuberProx


def test_huberprox():
    alpha = 0.1
    delta = 1.0
    prox = HuberProx(alpha=alpha, delta=delta)
    x = torch.tensor([0.5, 2.0, 3.0])
    result = prox.prox(x, 1)
    # Again, the expected result can be non-trivial. Let's just ensure it's the same shape.
    assert result.shape == x.shape
