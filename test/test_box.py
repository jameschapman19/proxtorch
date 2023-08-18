from proxtorch.constraints import BoxConstraint
import torch
torch.manual_seed(0)

def test_boxconstraint():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    constraint = BoxConstraint(1, 3)
    result = constraint.prox(x)
    expected = torch.tensor([[1.0, 2.0], [3.0, 3.0]])
    assert torch.allclose(result, expected)
    print("Box constraint test passed!")

    # test call method
    val = constraint(x)
    expected_val = False

    assert val == expected_val

    val = constraint(result)
    expected_val = True

    assert val == expected_val
