import torch

from proxtorch.operators import TraceNormProx
from proxtorch.constraints import TraceNormConstraint


def test_tracenormprox():
    alpha = 0.1
    prox = TraceNormProx(alpha=alpha)
    x = torch.rand((3, 3))
    result = prox.prox(x, 0.1)
    # Ensuring the trace norm is less than trace norm of x
    assert torch.trace(result) <= torch.trace(x)



def test_tracenormconstraint():
    alpha = 2.0
    constraint = TraceNormConstraint(alpha=alpha)
    x = torch.rand((3, 3))
    result = constraint.prox(x)
    # Ensuring the trace norm is less than or equal to alpha
    assert torch.trace(result) <= alpha

    # test call method
    val = constraint(x)
    expected_val = False
    assert val == expected_val

    val = constraint(result)
    expected_val = True
    assert val == expected_val
