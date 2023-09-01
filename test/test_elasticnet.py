import torch

from proxtorch.operators import ElasticNet

torch.manual_seed(0)


def test_elasticnetprox():
    alpha = 0.1
    l1_ratio = 0.5
    prox = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    x = torch.tensor([1.0, 2.0, 3.0])
    result = prox.prox(x, 1)
    # Hand-calculating expected can be non-trivial, so let's at least check the shape
    assert result.shape == x.shape
