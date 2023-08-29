from .tv_2d import TV_2DProx
from .tvl1_3d import TVL1_3DProx


class TVL1_2DProx(TVL1_3DProx):
    def __init__(
        self,
        alpha_l1: float,
        alpha_tv: float,
        shape=None,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> None:
        """
        Initialize the 2D Total Variation L1 proximal operator.

        Args:
            alpha_l1 (float): L1 regularization strength.
            alpha_tv (float): TV regularization strength.
            shape (tuple, optional): Desired shape for the input tensor. Defaults to None.
            device (str, optional): Device to use for calculations ("cuda" or "cpu"). Defaults to "cuda".
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 50.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-2.
        """
        super(TVL1_2DProx, self).__init__(
            alpha_l1=alpha_l1, alpha_tv=alpha_tv, max_iter=max_iter, tol=tol
        )
        self.tv = TV_2DProx(alpha_tv, shape, max_iter)
