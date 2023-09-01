import torch

from proxtorch.operators import Dummy

torch.manual_seed(0)


def test_dummyprox():
    prox = Dummy()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = prox.prox(x, 1)
    assert torch.allclose(result, x)
