from proxtorch.constraints import NonNegativeConstraint

import torch

def test_nonnegativeconstraint():
    x = torch.tensor([[-1.0, 2.0], [3.0, 4.0]])
    constraint = NonNegativeConstraint()
    result = constraint.prox(x)
    expected = torch.tensor([[0.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)
    print("Non-negative constraint test passed!")