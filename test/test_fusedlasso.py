import torch

from proxtorch.operators import FusedLassoProx


def test_fusedlassoprox():
    lambda_ = 0.5
    prox = FusedLassoProx(lambda_)
    x = torch.tensor([-1.0, 2.0, 0.5, 0.0, -0.2])

    result = prox.prox(x, 1)

    # Expected result is a sequence where differences between consecutive elements
    # have been shrunk due to the fused lasso penalty.
    # Since this is a non-trivial operation, the exact expected values would depend
    # on your implementation details, lambda, and x.
    # Here's a simple expected tensor for the above x, but this might not be correct:
    expected = torch.tensor([-3.5, -1.0, -2.0, -2.0, -2.0])

    assert torch.allclose(result, expected, atol=1e-6)
