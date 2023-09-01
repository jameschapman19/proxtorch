import torch

from proxtorch.operators import GraphNet3DProx, GraphNet2DProx

import numpy as np
import matplotlib.pyplot as plt

import torch
from proxtorch.operators import GraphNet2DProx


def test_converges_to_sparse_smooth():
    torch.manual_seed(0)

    # generate a spatially sparse signal
    x_true = torch.zeros(10, 10)
    x_true[3:7, 3:7] = torch.ones(4, 4)
    x_true = x_true.flatten()

    # generate a random matrix
    A = torch.rand(100, 100)

    # generate measurements
    y = A @ x_true

    # define the proximal operator
    alpha = 10
    l1_ratio = 0.0 # 0.5
    prox = GraphNet2DProx(alpha, l1_ratio)

    # define the objective function
    def objective(x):
        return 0.5 * torch.norm(A @ x.reshape(-1) - y) ** 2

    # define the step size
    tau = 1 / torch.norm(A.t() @ A)

    # initialize the solution
    x = torch.nn.Parameter(torch.rand(10, 10, requires_grad=True))

    # optimizer
    optimizer = torch.optim.SGD([x], lr=tau)

    # optimization loop
    for i in range(1000):
        optimizer.zero_grad()
        obj = objective(x) + prox(x)
        obj.backward()
        optimizer.step()
        x.data = prox.prox(x.data, tau)

    # check that the result is smooth
    plt.imshow(x.data.detach().numpy())
    plt.show()

    # compare with x_true
    difference = torch.norm(x.data.flatten() - x_true)
    assert difference < 1e-3

    # check that the result is sparse
    assert torch.sum(x.data == 0) > 50


def test_graph_net_3d_prox():
    alpha = 0.1
    l1_ratio = 0.5
    prox = GraphNet3DProx(alpha, l1_ratio)

    # Test _smooth method
    x = torch.rand(3, 5, 5)
    smooth_result = prox._smooth(x)
    assert isinstance(smooth_result, torch.Tensor)

    # Test _nonsmooth method
    nonsmooth_result = prox._nonsmooth(x)
    assert isinstance(nonsmooth_result, torch.Tensor)


def test_graph_net_2d_prox():
    alpha = 0.1
    l1_ratio = 0.5
    prox = GraphNet2DProx(alpha, l1_ratio)

    # Test _smooth method
    x = torch.rand(3, 5)
    smooth_result = prox._smooth(x)
    assert isinstance(smooth_result, torch.Tensor)

    # Test _nonsmooth method
    nonsmooth_result = prox._nonsmooth(x)
    assert isinstance(nonsmooth_result, torch.Tensor)
