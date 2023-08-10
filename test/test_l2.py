import torch

from proxtorch.operators import L2Prox


def test_l2prox():
    alpha = 0.1
    prox = L2Prox(alpha=alpha)
    x = torch.tensor([1.0, 2.0, 3.0])
    result = prox.prox(x, 1)
    expected = x / (1.0 + alpha)
    assert torch.allclose(result, expected)
