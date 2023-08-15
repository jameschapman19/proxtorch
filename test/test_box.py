from proxtorch.constraints import BoxConstraint
import torch


def test_boxconstraint():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    constraint = BoxConstraint(1, 3)
    result = constraint.prox(x)
    expected = torch.tensor([[1.0, 2.0], [3.0, 3.0]])
    assert torch.allclose(result, expected)
    print("Box constraint test passed!")
