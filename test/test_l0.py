import torch

from proxtorch.operators.l0 import L0Prox


def test_basic_l0():
    l0_op = L0Prox(threshold=2.0)
    x = torch.tensor([1.0, 3.0, -0.5, -2.5])

    # Test prox method
    y = l0_op.prox(x, 1.0)
    expected = torch.tensor([0.0, 3.0, 0.0, -2.5])
    assert torch.equal(y, expected)

    # Test call method
    val = l0_op(x)
    expected_val = 2  # 3 values > threshold
    assert val == expected_val
