import torch

from proxtorch.operators import L2Prox
from proxtorch.constraints import L2Ball


def test_l2prox():
    alpha = 0.1
    prox = L2Prox(alpha=alpha)
    x = torch.tensor([1.0, 2.0, 3.0])
    result = prox.prox(x, 1)
    expected = x / (1.0 + alpha)
    assert torch.allclose(result, expected)


def test_l2ball():
    alpha = 0.1
    constraint = L2Ball(s=alpha)
    x = torch.tensor([1.0, 2.0, 3.0])
    result = constraint.prox(x)
    # check if the norm of the result is less than or equal to alpha
    assert torch.norm(result) <= alpha