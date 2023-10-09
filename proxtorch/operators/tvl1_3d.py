"""
Total Variation L1 (TV-L1) denoising algorithm.

Inspired by Nilearn's proxtvl1 implementation.

References:
- Wikipedia: Total Variation Denoising (http://en.wikipedia.org/wiki/Total_variation_denoising)
- Beck, A., & Teboulle, M. (2009). Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems.
- Nilearn (https://nilearn.github.io/)
"""

from math import sqrt

import torch
import torch.nn.functional as F

from proxtorch.base import ProxOperator


def get_padding_tuple(dim_index, ndim):
    """
    Return a padding tuple for a specified dimension index.

    Args:
        dim_index (int): Index of the dimension for which padding is needed.
        ndim (int): Total number of dimensions in the tensor.

    Returns:
        tuple: Padding tuple with a value of 1 at the specified dimension and 0 elsewhere.
    """
    padding_tuple = [0] * (ndim * 2)
    padding_tuple[-2 * dim_index - 1] = 1
    return tuple(padding_tuple)


class TVL1_3D(ProxOperator):
    """
    Class for the 3D Total Variation proximal operator.
    """

    def __init__(
        self,
        alpha: float,
        l1_ratio=0.05,
        max_iter: int = 200,
        tol: float = 5e-5,
    ) -> None:
        """
        Initialize the 3D Total Variation proximal operator.

        Args:
            alpha (float): Regularization strength.
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 200.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-4.
            l1_ratio (float, optional): The L1 ratio. Defaults to 0.0.
        """
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio

    def gradient(self, x):
        """
        Compute the gradient of the tensor x using finite differences.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Gradients of the tensor x.
        """
        gradients = torch.zeros((x.dim() + 1,) + x.shape, device=x.device)
        # For each dimension compute the gradient using torch.diff
        for d in range(x.dim()):
            gradients[d, ...] = F.pad(
                torch.diff(x, dim=d, n=1), pad=get_padding_tuple(d, x.dim())
            )
        gradients[:-1] *= 1.0 - self.l1_ratio
        gradients[-1] = self.l1_ratio * x
        return gradients

    def divergence(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute the divergence of the tensor p.

        Args:
            p (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Divergence of the tensor p.
        """
        div_x = torch.zeros_like(p[-1])
        div_y = torch.zeros_like(p[-1])
        div_z = torch.zeros_like(p[-1])

        div_x[:-1].add_(p[0, :-1, :, :])
        div_y[:, :-1].add_(p[1, :, :-1, :])
        div_z[:, :, :-1].add_(p[2, :, :, :-1])

        div_x[1:-1].sub_(p[0, :-2, :, :])
        div_y[:, 1:-1].sub_(p[1, :, :-2, :])
        div_z[:, :, 1:-1].sub_(p[2, :, :, :-2])

        div_x[-1].sub_(p[0, -2, :, :])
        div_y[:, -1].sub_(p[1, :, -2, :])
        div_z[:, :, -1].sub_(p[2, :, :, -2])

        return (div_x + div_y + div_z) * (1 - self.l1_ratio) - self.l1_ratio * p[-1]

    def _projector_on_tvl1_dual(self, grad):
        """
        Function to compute TV-l1 duality gap.

        Modifies IN PLACE the gradient + id to project it
        on the l21 unit ball in the gradient direction and the L1 ball in the
        identity direction.

        Args:
            grad (torch.Tensor): Gradient tensor.

        Returns:
            torch.Tensor: Projected gradient tensor.
        """
        # The l21 ball for the gradient direction
        if self.l1_ratio < 1.0:
            # infer number of axes and include an additional axis if l1_ratio > 0
            end = len(grad) - int(self.l1_ratio > 0.0)
            norm = torch.sqrt(torch.sum(grad[:end] * grad[:end], 0))
            norm = torch.clamp(norm, min=1.0)  # set everything < 1 to 1
            for i in range(end):
                grad[i] /= norm

        # The L1 ball for the identity direction
        if self.l1_ratio > 0.0:
            norm = torch.abs(grad[-1])
            norm = torch.clamp(norm, min=1.0)
            grad[-1] /= norm

        return grad

    def _dual_gap_prox_tvl1(self, input_img_norm, new, gap, weight):
        """
        Compute the dual gap of total variation denoising.

        Args:
            input_img_norm (float): Norm of the input image.
            new (torch.Tensor): Updated tensor.
            gap (torch.Tensor): Gap tensor.
            weight (float): Regularization strength.

        Returns:
            float: Dual gap value.
        Notes:
            see "Total variation regularization for fMRI-based prediction of behavior",
            by Michel et al. (2011) for a derivation of the dual gap
        """
        tv_new = self.tvl1_from_grad(self.gradient(new))
        gap = gap.view(-1)
        d_gap = (
            torch.dot(gap, gap)
            + 2 * weight * tv_new
            - input_img_norm
            + torch.sum(new * new)
        )
        return 0.5 * d_gap

    def prox(self, x: torch.Tensor, lr: float) -> torch.Tensor:
        """
        Iterative algorithm to compute the proximal mapping of the tensor.

        Args:
            x (torch.Tensor): Input tensor.
            lr (float): Learning rate.

        Returns:
            torch.Tensor: Tensor after applying the proximal operation.

        Notes
        -----
        Total variation denoising aims to minimize the total variation of the image,
        which can be roughly described as the integral of the norm of the image gradient.
        As a result, it produces "cartoon-like" images, i.e., piecewise-constant images.
        For more details, refer to:
        http://en.wikipedia.org/wiki/Total_variation_denoising

        This function implements the FISTA (Fast Iterative Shrinkage
        Thresholding Algorithm) algorithm of Beck et Teboulle, adapted to
        total variation denoising in "Fast gradient-based algorithms for
        constrained total variation image denoising and deblurring problems"
        (2009).

        For more on bound constraints implementation, see the aforementioned Beck and Teboulle paper.
        """
        fista = True
        weight = self.alpha * lr
        input_shape = x.shape
        input_img_norm = torch.norm(x) ** 2
        lipschitz_constant = 1.1 * (4 * 3)
        negated_output = -x
        grad_aux = torch.zeros_like(self.gradient(x))
        grad_im = torch.zeros_like(grad_aux)
        t = 1.0
        i = 0
        dgap = torch.tensor(float("inf")).to(x.device)
        while i < self.max_iter:
            # tv_prev = self.tv_from_grad(self.gradient(output))
            grad_tmp = self.gradient(negated_output)
            grad_tmp *= 1.0 / (lipschitz_constant * weight)
            grad_aux += grad_tmp
            grad_tmp = self._projector_on_tvl1_dual(grad_aux)

            # Careful, in the next few lines, grad_tmp and grad_aux are a
            # view on the same array, as _projector_on_tvl1_dual returns a view
            # on the input array
            t_new = 0.5 * (1 + sqrt(1 + 4 * t**2))
            t_factor = (t - 1) / t_new
            if fista:
                # fista
                grad_aux = (1 + t_factor) * grad_tmp - t_factor * grad_im
            else:
                # ista
                grad_aux = grad_tmp
            grad_im = grad_tmp
            t = t_new
            gap = weight * self.divergence(grad_aux)
            negated_output = gap - x
            if i % 4 == 0:
                old_dgap = dgap
                dgap = self._dual_gap_prox_tvl1(
                    input_img_norm, -negated_output, gap, weight
                )
                if dgap < self.tol:
                    break
                if old_dgap < dgap:
                    fista = False
            i += 1
        output = x - weight * self.divergence(grad_im)
        return output.reshape(input_shape)

    def _nonsmooth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Total Variation (TV) for a given tensor x.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The TV of the tensor x.
        """
        gradients = self.gradient(x)
        return self.tvl1_from_grad(gradients) * self.alpha

    @staticmethod
    def tvl1_from_grad(gradients: torch.Tensor) -> torch.Tensor:
        r"""
        Calculate the TV from gradients.

        Args:
            gradients (torch.Tensor): Gradient tensor.

        Returns:
            float: The TV value computed from the gradients.
        """
        tv = torch.sum(torch.sqrt(torch.sum(gradients[:-1] * gradients[:-1], dim=0)))
        l1 = torch.sum(torch.abs(gradients[-1]))
        return tv + l1
