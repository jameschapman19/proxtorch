import torch

from proxtorch.operators import L1Prox


def test_L1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    l1 = L1Prox(sigma=0.5)

    # Test data
    x = torch.tensor([-1.5, -0.5, 0, 0.5, 1.5]).to(device)

    # Apply the prox operator
    result = l1.prox(x, 1)
    expected_result = torch.tensor([-1.0, 0, 0, 0, 1.0]).to(device)
    assert torch.allclose(result, expected_result, atol=1e-6)

    # Check the L1Prox norm
    assert l1(x) == 4.0
    print("L1Prox tests passed!")
