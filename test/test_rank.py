import torch

from proxtorch.constraints import RankConstraint

torch.manual_seed(0)


def test_rank_constraint():
    # Create a 10x10 rank 5 matrix
    u = torch.randn(10, 5)
    v = torch.randn(10, 5)
    matrix = u @ v.T
    assert matrix.size() == (10, 10)

    # RankConstraint for maximum rank of 3
    constraint = RankConstraint(3)

    # Test if constraint correctly identifies matrix rank
    assert not constraint(matrix)

    # Test prox operation
    projected_matrix = constraint.prox(matrix)
    projected_rank = (torch.svd(projected_matrix).S > 1e-5).sum().item()

    assert projected_rank == 3
    assert constraint(projected_matrix)  # The constraint should now be satisfied
