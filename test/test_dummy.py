from proxtorch.operators import DummyProx

import torch

torch.manual_seed(0)


def test_dummyprox():
    prox = DummyProx()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = prox.prox(x, 1)
    assert torch.allclose(result, x)
