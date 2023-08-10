import torch.nn.functional as F
import torch

from proxtorch.operators.tv_3d import TV_3DProx


class TV_2DProx(TV_3DProx):
    @staticmethod
    def gradient(x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute gradient in the x and y directions and pad the gradients.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed gradient tensor in 2D.
        """
        grad_x = x[1:, :] - x[:-1, :]
        grad_y = x[:, 1:] - x[:, :-1]

        grad_x = F.pad(grad_x, (0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 1, 0, 0))

        return torch.stack([grad_x, grad_y], dim=0)

    @staticmethod
    def divergence(p: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the divergence (negative adjoint of gradient) for 2D.

        Args:
            p (torch.Tensor): Tensor representing the 2D gradient.

        Returns:
            torch.Tensor: Computed divergence tensor in 2D.
        """
        div_x = p[0, :-1, :] - p[0, 1:, :]
        div_y = p[1, :, :-1] - p[1, :, 1:]

        div_x = F.pad(div_x, (0, 0, 1, 0), "constant", 0)
        div_y = F.pad(div_y, (1, 0, 0, 0), "constant", 0)

        return div_x + div_y

    @staticmethod
    def tv_from_grad(gradients: torch.Tensor) -> float:
        r"""
        Calculate the TV from 2D gradients.

        Args:
            gradients (torch.Tensor): 2D gradient tensor.

        Returns:
            float: The TV value computed from the 2D gradients.
        """
        grad_x, grad_y = gradients[0], gradients[1]
        return torch.sum(torch.sqrt(grad_x**2 + grad_y**2))
