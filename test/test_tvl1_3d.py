import torch
import matplotlib.pyplot as plt

from proxtorch.operators import TVL1_3DProx


def test_total_variation_3d_prox():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tv = TVL1_3DProx(sigma_l1=1e-2, sigma_tv=1e-2, max_iter=50)

    p = 10

    # Test Data for 3D
    # x = torch.ones(p, p, p).to(device)  # Example 3D tensor
    # Top half of x in one dimension is 1, rest is 0
    x = torch.cat((torch.ones(p // 2, p, p), torch.zeros(p // 2, p, p)), dim=0)
    # add Gaussian noise
    x_noised = x + torch.randn(p, p, p) * 0.1
    x_noised = x_noised.to(device)

    # Prox Test
    prox_result = tv.prox(x_noised, 1)
    assert prox_result.shape == x.shape
    print("Prox Test Passed!")

    print("All tests passed!")

    # # Make a plot of prox_result to convince yourself that it is a smoothed version of x
    # # For 3D data, you might want to just visualize a slice for simplicity
    # slice_idx = p // 2  # mid slice
    #
    # vmin = 0
    # vmax = 1
    #
    # plt.imshow(x[slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
    # plt.title("Original Image Slice")
    # plt.show()
    #
    # plt.imshow(x_noised[slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
    # plt.title("Noisy Image Slice")
    # plt.show()
    #
    # plt.imshow(prox_result[slice_idx - 1].cpu().numpy(), vmin=vmin, vmax=vmax)
    # plt.title("Prox Result Slice")
    # plt.show()
    #
    # plt.imshow(prox_result[slice_idx].cpu().numpy(), vmin=vmin, vmax=vmax)
    # plt.title("Prox Result Slice")
    # plt.show()
    # print("Plotting prox_result and original image slice...")
