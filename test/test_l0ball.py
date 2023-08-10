import torch

from proxtorch.operators import L0Ball


def test_basic_l0ball():
    l0ball_op = L0Ball(s=2)
    x = torch.tensor([1.0, 3.0, -0.5, -2.5, 0.0])

    # Test prox method
    y = l0ball_op.prox(x, 1.0)
    expected = torch.tensor(
        [0.0, 3.0, 0.0, -2.5, 0.0]
    )  # Only the two largest magnitudes are kept
    assert torch.equal(y, expected)

    # Test call method
    val = l0ball_op(x)
    expected_val = 2.0
    assert val == expected_val
