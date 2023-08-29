import torch
import torch.nn.functional as F

from proxtorch.base import ProxOperator


class TV_3DProx(ProxOperator):
    def __init__(
        self, alpha: float, shape=None, max_iter: int = 50, tol: float = 1e-4
    ) -> None:
        """
        Initialize the 3D Total Variation proximal operator.

        Args:
            alpha (float): Regularization strength.
            shape (tuple, optional): Desired shape for the input tensor. Defaults to None.
            max_iter (int, optional): Maximum iterations for the iterative algorithm. Defaults to 50.
            tol (float, optional): Tolerance level for early stopping. Defaults to 1e-2.
        """
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.shape = shape

    @staticmethod
    def gradient(x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute gradient in the x, y, and z directions and pad the gradients.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed gradient tensor.
        """
        grad_x = x[1:, :, :] - x[:-1, :, :]
        grad_y = x[:, 1:, :] - x[:, :-1, :]
        grad_z = x[:, :, 1:] - x[:, :, :-1]

        grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_z = F.pad(grad_z, (0, 1, 0, 0, 0, 0))

        return torch.stack([grad_x, grad_y, grad_z], dim=0)

    @staticmethod
    def divergence(p: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the divergence (negative adjoint of gradient).

        Args:
            p (torch.Tensor): Tensor representing the gradient.

        Returns:
            torch.Tensor: Computed divergence tensor.
        """
        div_x = p[0, :-1, :, :] - p[0, 1:, :, :]
        div_y = p[1, :, :-1, :] - p[1, :, 1:, :]
        div_z = p[2, :, :, :-1] - p[2, :, :, 1:]

        div_x = F.pad(div_x, (0, 0, 0, 0, 1, 0), "constant", 0)
        div_y = F.pad(div_y, (0, 0, 1, 0, 0, 0), "constant", 0)
        div_z = F.pad(div_z, (1, 0, 0, 0, 0, 0), "constant", 0)

        return div_x + div_y + div_z

    def prox(self, x: torch.Tensor, lr: float) -> torch.Tensor:
        r"""
        Iterative algorithm to compute the proximal mapping of the tensor.

        Args:
            x (torch.Tensor): Input tensor.
            lr (float): Learning rate.

        Returns:
            torch.Tensor: Tensor after applying the proximal operation.
        """
        input_shape = x.shape
        # check if x has shape self.shape if not try to reshape
        if self.shape and x.shape != self.shape:
            x = x.reshape(self.shape)
        # Define constants and initial values
        tau = 1.0 / (2.0 * x.ndim)
        total_elements = torch.numel(x)

        # Initialize dual variable and divergence tensor
        p = torch.zeros_like(self.gradient(x))

        E_init = None  # Initial energy value for convergence checking

        # Iterative algorithm to compute the proximal mapping
        for i in range(self.max_iter):
            # Compute divergence (negative adjoint of gradient)
            d = self.divergence(p)

            # Compute the "denoised" tensor
            out = x + d

            # Compute energy associated with current state
            E = torch.sum(d**2)
            gradient_of_out = self.gradient(out)

            # Compute norm of the gradient
            norm = torch.sqrt(torch.sum(gradient_of_out**2, dim=0, keepdim=True))
            E += lr * self.alpha * torch.sum(norm)

            # Update step for the dual variable p
            norm_scaling = tau / (self.alpha * lr)
            p -= tau * gradient_of_out
            p /= norm * norm_scaling + 1.0

            # Normalize energy by total number of elements
            E /= total_elements

            # Check for convergence based on relative change in energy
            if i > 0 and torch.abs(E - E_prev) < self.tol * E_init:
                break

            # Store the current energy for the next iteration's comparison
            E_prev = E

            # Initialize E_init during the first iteration
            if i == 0:
                E_init = E

        return out.reshape(input_shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Total Variation (TV) for a given tensor x.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The TV of the tensor x.
        """
        # Check if self.shape is not None and x has different shape, then try to reshape
        if self.shape and x.shape != self.shape:
            x = x.reshape(self.shape)
        gradients = self.gradient(x)
        return self.tv_from_grad(gradients)

    @staticmethod
    def tv_from_grad(gradients: torch.Tensor) -> float:
        r"""
        Calculate the TV from gradients.

        Args:
            gradients (torch.Tensor): Gradient tensor.

        Returns:
            float: The TV value computed from the gradients.
        """
        grad_x, grad_y, grad_z = gradients[0], gradients[1], gradients[2]
        return torch.sum(torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2))
