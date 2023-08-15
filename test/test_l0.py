import torch

from proxtorch.operators import L0Prox
from proxtorch.constraints import L0Ball


def test_l0():
    l0_op = L0Prox(threshold=2.0)
    x = torch.tensor([1.0, 3.0, -0.5, -2.5])

    # Test prox method
    y = l0_op.prox(x, 1.0)
    expected = torch.tensor([0.0, 3.0, 0.0, -2.5])
    assert torch.equal(y, expected)

    # Test call method
    val = l0_op(x)
    expected_val = 2  # 3 values > threshold
    assert val == expected_val

def test_l0_ball():
    l0_ball = L0Ball(s=2)
    x = torch.tensor([1.0, 3.0, -0.5, -2.5])

    # Test prox method
    y = l0_ball.prox(x)
    expected = torch.tensor([0.0, 3.0, 0.0, -2.5])
    assert torch.equal(y, expected)

    # Test call method
    val = l0_ball(x)
    expected_val = False
    assert val == expected_val