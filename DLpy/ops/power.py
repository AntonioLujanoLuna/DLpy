from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray

from ..core import Function, Tensor
from ..core.context import Context  # Import Context for type hints


class Power(Function):
    @staticmethod
    def forward(
        ctx: Context,
        base: Union[Tensor, NDArray[Any], float, int],
        exponent: Union[Tensor, float, int],
    ) -> Tensor:
        """
        Computes element-wise power operation: base ^ exponent.

        Args:
            ctx: Context object for storing data needed in backward pass
            base: The base values, can be a tensor or scalar
            exponent: The power to raise the base to, must be a scalar value

        Returns:
            A new tensor containing the result of base raised to exponent

        Raises:
            TypeError: If exponent is not a Tensor, int, or float
            ValueError: If exponent is a non-scalar tensor
        """
        if not isinstance(base, Tensor):
            base = Tensor(base)
        if not isinstance(exponent, (Tensor, int, float)):
            raise TypeError("Exponent must be a Tensor, int, or float")

        # Convert Tensor exponent to scalar if possible
        if isinstance(exponent, Tensor):
            if exponent.data.size == 1:
                exponent = float(exponent.data)
            else:
                raise ValueError("Only scalar exponents are supported")

        ctx.save_for_backward(base)
        ctx.save_arguments(exponent=exponent)

        return Tensor(np.power(base.data, exponent))

    @staticmethod
    def backward(
        ctx: Context, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]
    ) -> None:
        """
        Computes gradients for the power operation.

        For f(x) = x^n, the derivative is f'(x) = nx^(n-1)

        Args:
            ctx: Context containing saved tensors and arguments
            grad_output: Gradient from upstream operations
            grad_dict: Dictionary mapping tensor IDs to their gradients
        """
        (base,) = ctx.saved_tensors
        exponent = ctx.saved_arguments["exponent"]

        if base.requires_grad:
            grad = grad_output * exponent * np.power(base.data, exponent - 1)
            grad_dict[id(base)] = grad


class Divide(Function):
    @staticmethod
    def forward(
        ctx: Context,
        numerator: Union[Tensor, NDArray[Any], float, int],
        denominator: Union[Tensor, NDArray[Any], float, int],
    ) -> Tensor:
        """
        Computes element-wise division: numerator / denominator.

        Args:
            ctx: Context object for storing data needed in backward pass
            numerator: The values to be divided
            denominator: The values to divide by

        Returns:
            A new tensor containing the quotient

        Raises:
            ValueError: If any element in denominator is zero
        """
        if not isinstance(numerator, Tensor):
            numerator = Tensor(numerator)
        if not isinstance(denominator, Tensor):
            denominator = Tensor(denominator)

        # Check for division by zero
        if np.any(denominator.data == 0):
            raise ValueError("Division by zero encountered")

        ctx.save_for_backward(numerator, denominator)
        return Tensor(numerator.data / denominator.data)

    @staticmethod
    def backward(
        ctx: Context, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]
    ) -> None:
        """
        Computes gradients for the division operation.

        For f(x,y) = x/y:
        - df/dx = 1/y
        - df/dy = -x/y^2

        Args:
            ctx: Context containing saved tensors
            grad_output: Gradient from upstream operations
            grad_dict: Dictionary mapping tensor IDs to their gradients
        """
        numerator, denominator = ctx.saved_tensors

        if numerator.requires_grad:
            grad_dict[id(numerator)] = grad_output / denominator.data

        if denominator.requires_grad:
            grad_dict[id(denominator)] = -grad_output * numerator.data / (denominator.data**2)
