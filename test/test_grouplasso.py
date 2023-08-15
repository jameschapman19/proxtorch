import torch
from proxtorch.operators import GroupLassoProx


def test_prox():
    lambda_ = 2.0
    group_sizes = [2, 3]
    x = torch.tensor([1.5, 2.0, -1.0, 0.5, 0.5], dtype=torch.float32)

    operator = GroupLassoProx(lambda_, group_sizes)
    result = operator.prox(x, 1.0)

    expected_result = torch.tensor([0.3, 0.4, 0.0, 0.0, 0.0], dtype=torch.float32)

    assert torch.allclose(result, expected_result, atol=1e-4)


def test_call():
    lambda_ = 0.1
    group_sizes = [2, 3]
    x = torch.tensor([1.5, 2.0, -1.0, 0.5, 0.5], dtype=torch.float32)

    operator = GroupLassoProx(lambda_, group_sizes)
    penalty = operator(x)

    expected_penalty = lambda_ * (torch.norm(x[:2], p=2) + torch.norm(x[2:], p=2))

    assert torch.allclose(penalty, expected_penalty, atol=1e-4)
