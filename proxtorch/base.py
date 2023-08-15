import torch


class ProxOperator:
    r"""Base class for proximal operators in proxtorch.

    This class provides the basic structure and enforces the implementation
    of the required methods. Subclasses should implement the 'prox' and
    '__call__' methods to provide specific proximal operators.

    Note:
        This is an abstract class and should not be instantiated directly.
    """

    def prox(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        r"""Proximal mapping of the operator.

        Args:
            x (torch.Tensor): Input tensor.
            tau (float): Proximal parameter.

        Returns:
            torch.Tensor: Result of the proximal mapping.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the 'prox' method.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r"""Function call to evaluate the operator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of the operator.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the '__call__' method.")


class Constraint:
    r"""Base class for constraints in proxtorch.

    This class provides the basic structure and enforces the implementation
    of the required methods. Subclasses should implement the 'is_satisfied' and
    'project' methods to provide specific constraints.

    Note:
        This is an abstract class and should not be instantiated directly.
    """

    def __call__(self, x: torch.Tensor) -> bool:
        r"""Check if the constraint is satisfied for the given tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            bool: True if constraint is satisfied, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'is_satisfied' method."
        )

    def prox(self, x: torch.Tensor) -> torch.Tensor:
        r"""Projects the tensor onto the feasible set defined by the constraint.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after projection.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the 'project' method.")
