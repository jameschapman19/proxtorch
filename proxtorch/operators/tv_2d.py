from proxtorch.operators.tv_3d import TV_3DProx


class TV_2DProx(TV_3DProx):
    def __init__(
            self, alpha: float, shape=None, max_iter: int = 200, tol: float = 1e-4
    ) -> None:
        """
        Initialize the 3D Total Variation proximal operator.

        Args:
            alpha (float): Regularization strength.
            shape (tuple, optional): Desired shape for the input tensor. Defaults to None.
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 50.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-2.
        """
        super().__init__(alpha, shape, max_iter, tol, l1_ratio=0.0)