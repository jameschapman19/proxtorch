import torch


def test_tracenormprox():
    from proxtorch.operators import TraceNorm

    torch.manual_seed(0)
    alpha = 0.1
    prox = TraceNorm(alpha=alpha)
    x = torch.rand((3, 3))
    result = prox.prox(x, 0.1)
    # Ensuring the trace norm is less than trace norm of x
    assert torch.trace(result) <= torch.trace(x)


def test_tracenormconstraint():
    from proxtorch.constraints import TraceNorm

    torch.manual_seed(0)
    alpha = 2.0
    constraint = TraceNorm(alpha=alpha)
    x = torch.rand((3, 3))
    result = constraint.prox(x)
    # Ensuring the trace norm is less than or equal to s
    assert torch.trace(result) <= alpha

    # test call method
    val = constraint(x)
    expected_val = False
    assert val == expected_val

    val = constraint(result)
    expected_val = True
    assert val == expected_val
