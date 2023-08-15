import torch

from proxtorch.operators import FrobeniusProx
from proxtorch.constraints import FrobeniusConstraint


def test_matrixfrobeniusprox():
    alpha = 0.1
    prox = FrobeniusProx(alpha=alpha)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = prox.prox(x, 1)
    expected = x / (1.0 + alpha)
    assert torch.allclose(result, expected)

def test_matrixfrobeniusconstraint():
    alpha = 0.1
    constraint = FrobeniusConstraint(alpha=alpha)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = constraint.prox(x)
    # Ensuring the trace norm is less than or equal to alpha
    assert torch.norm(result) <= alpha

    # test call method
    val = constraint(x)
    expected_val = False
    assert val == expected_val

    val = constraint(result)
    expected_val = True
    assert val == expected_val

