from proxtorch.operators.tvl1_2d import TVL1_2D


class TV_2D(TVL1_2D):
    def __init__(self, alpha: float, max_iter: int = 200, tol: float = 1e-4) -> None:
        """
        Initialize the 3D Total Variation proximal operator.

        Args:
            alpha (float): Regularization strength.
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 50.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-2.
        """
        super().__init__(alpha, l1_ratio=0.0, max_iter=max_iter, tol=tol)
