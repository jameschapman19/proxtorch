from .tv_2d import TV_2DProx
from .tvl1_3d import TVL1_3DProx


class TVL1_2DProx(TVL1_3DProx):
    def __init__(
        self, sigma_l1: float, sigma_tv: float, max_iter: int = 50, tol: float = 1e-4
    ) -> None:
        """
        Initialize the 2D Total Variation L1 proximal operator.

        Args:
            sigma_l1 (float): L1 regularization strength.
            sigma_tv (float): TV regularization strength.
            device (str, optional): Device to use for calculations ("cuda" or "cpu"). Defaults to "cuda".
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 50.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-2.
        """
        super(TVL1_2DProx, self).__init__(sigma_l1, sigma_tv, max_iter, tol)
        self.tv = TV_2DProx(sigma_tv, max_iter)
