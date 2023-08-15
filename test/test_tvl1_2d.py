import torch

from proxtorch.operators import TVL1_2DProx


# Test the TV_2DProx class
def test_total_variation_2d_prox():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tv = TVL1_2DProx(sigma_l1=1e-3, sigma_tv=1e-3, max_iter=50)

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
    #
    # # Make a plot of prox_result to convince yourself that it is a smoothed version of x
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(x.cpu().numpy())
    # # label
    # plt.title("Original Image")
    # plt.show()
    # plt.imshow(prox_result.cpu().numpy())
    # # label
    # plt.title("Prox Result")
    # plt.show()
    # print("Plotting prox_result and original image...")
