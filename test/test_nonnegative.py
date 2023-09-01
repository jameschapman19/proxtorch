import torch

from proxtorch.constraints import NonNegative

torch.manual_seed(0)


def test_nonnegativeconstraint():
    x = torch.tensor([[-1.0, 2.0], [3.0, 4.0]])
    constraint = NonNegative()
    result = constraint.prox(x)
    expected = torch.tensor([[0.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)
    print("Non-negative constraint test passed!")

    # test call method
    val = constraint(x)
    expected_val = False

    assert val == expected_val

    val = constraint(result)
    expected_val = True

    assert val == expected_val
