import torch

from proxtorch.constraints import L1Ball
from proxtorch.operators import L1

torch.manual_seed(0)


def test_l1():
    l1_op = L1(alpha=2.0)
    x = torch.tensor([1.0, 3.0, -0.5, -2.5])

    # Test prox method
    y = l1_op.prox(x, 1.0)
    expected = torch.tensor([0.0, 1.0, 0.0, -0.5])
    assert torch.equal(y, expected)

    # Test call method
    val = l1_op(x)
    expected_val = 14
    assert val == expected_val


def test_l1_ball():
    l1_ball = L1Ball(s=2)
    x = torch.tensor([1.0, 3.0, -0.5, -2.5])

    # Test prox method
    y = l1_ball.prox(x)
    expected = torch.tensor([0.0, 1.25, 0.0, -0.75])
    assert torch.equal(y, expected)

    # Test call method
    val = l1_ball(x)
    expected_val = False
    assert val == expected_val
