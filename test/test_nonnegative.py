import torch

from proxtorch.operators import NonNegativeProx


def test_nonnegativeprox():
    prox = NonNegativeProx()
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    result = prox.prox(x, 1)

    # Expected result is [0, 0, 1, 2] since we're projecting onto the non-negative orthant
    expected = torch.tensor([0.0, 0.0, 1.0, 2.0])

    assert torch.allclose(result, expected)

    # Ensuring all values are non-negative
    assert torch.all(result >= 0)
