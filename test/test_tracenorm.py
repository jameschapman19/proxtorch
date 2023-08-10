import torch

from proxtorch.operators import TraceNormProx


def test_tracenormprox():
    alpha = 0.1
    prox = TraceNormProx(alpha=alpha)
    x = torch.rand((3, 3))
    result = prox.prox(x, 1)
    # Ensuring the shape is maintained after applying prox
    assert result.shape == x.shape
