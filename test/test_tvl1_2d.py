import torch

from proxtorch.operators import TVL1_2DProx

torch.manual_seed(0)


# Test the TV_2DProx class
def test_total_variation_2d_prox():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tv = TVL1_2DProx(alpha=1e-3, l1_ratio=0.1, max_iter=50)

    p = 10

    # Test Data
    # x = torch.ones(p, p).to(device)  # Example tensor
    # Top half of x is 1, bottom half is 0
    x = torch.cat((torch.ones(p // 2, p), torch.zeros(p // 2, p)), dim=0)
    # add guassian noise
    x = x + torch.randn(p, p) * 0.1
    x = x.to(device)

    # Prox Test
    prox_result = tv.prox(x, 1)
    assert prox_result.shape == x.shape
    print("Prox Test Passed!")

    print("All tests passed!")
