import torch

from proxtorch.constraints import L1Ball


def test_L1Ball():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    l1_ball = L1Ball(s=2.0)

    # Test data
    x = torch.tensor([-1.5, -0.5, 0, 0.5, 1.5]).to(device)

    # Apply the prox operator (projection)
    result = l1_ball.prox(x)
    # The expected result can vary based on the algorithm used. This is a simple test.
    assert torch.norm(result, p=1).item() <= 2.0 + 1e-6

    # Check the distance to the L1Prox ball
    assert l1_ball(x) == 2.0
    print("L1Ball tests passed!")
