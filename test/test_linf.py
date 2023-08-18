from proxtorch.constraints import LInfinityBall
import torch

torch.manual_seed(0)


def test_linf_ball():
    linf_ball = LInfinityBall(s=2)
    x = torch.tensor([1.0, 3.0, -0.5, -2.5])

    # Test prox method
    y = linf_ball.prox(x)
    expected = torch.tensor([1.0, 2.0, -0.5, -2.0])
    assert torch.equal(y, expected)

    # Test call method
    val = linf_ball(x)
    expected_val = False
    assert val == expected_val
