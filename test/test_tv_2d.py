import torch

from proxtorch.operators import TV_2DProx


# Test the TV_2DProx class
def test_total_variation_2d_prox():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tv = TV_2DProx(0.1, max_iter=1000)

    p = 10

    # Test Data
    # x = torch.ones(p, p).to(device)  # Example tensor
    # Top half of x is 1, bottom half is 0
    x = torch.cat((torch.ones(p // 2, p), torch.zeros(p // 2, p)), dim=0)
    # add guassian noise
    x = x + torch.randn(p, p) * 0.1
    x = x.to(device)

    # Gradient Test
    gradient = tv.gradient(x)
    assert gradient.shape == (2, p, p)
    print("Gradient Test Passed!")

    # Divergence Test
    divergence = tv.divergence(gradient)
    assert divergence.shape == x.shape
    print("Divergence Test Passed!")

    # Prox Test
    prox_result = tv.prox(x, 1)
    assert prox_result.shape == x.shape
    print("Prox Test Passed!")

    print("All tests passed!")

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


# test that zero is returned when x is zero
def test_zero():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tv = TV_2DProx(0.5)
    x = torch.zeros(10, 10).to(device)
    assert torch.all(tv.prox(x, 1.0) == 0)
    print("Zero test passed!")
