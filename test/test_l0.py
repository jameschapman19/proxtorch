import torch

from proxtorch.constraints import L0Ball

torch.manual_seed(0)


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
