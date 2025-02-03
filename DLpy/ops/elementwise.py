from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray

from ..core import Function, Tensor
from ..core.context import Context


class Log(Function):
    """
    Natural logarithm operation.

    Forward: f(x) = ln(x)
    Backward: f'(x) = 1/x

    Note: This operation requires positive input values as log is undefined
    for negative numbers and zero.
    """

    @staticmethod
    def forward(ctx: Context, x: Union[Tensor, NDArray[Any], float, int]) -> Tensor:
        """
        Computes the natural logarithm of the input tensor.

        Args:
            ctx: Context object for storing information needed in backward pass
            x: Input tensor or scalar value, must be strictly positive

        Returns:
            A new tensor containing the natural logarithm of x

        Raises:
            ValueError: If any elements of x are less than or equal to zero
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Check for negative values
        if np.any(x.data <= 0):
            raise ValueError("Log of negative numbers or zero is undefined")

        ctx.save_for_backward(x)
        return Tensor(np.log(x.data))

    @staticmethod
    def backward(
        ctx: Context, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]
    ) -> None:
        """
        Computes the gradient of the natural logarithm.

        The derivative of ln(x) is 1/x.

        Args:
            ctx: Context containing saved tensors from forward pass
            grad_output: Gradient from upstream operations
            grad_dict: Dictionary mapping tensor IDs to their gradients
        """
        (x,) = ctx.saved_tensors
        if x.requires_grad:
            # d/dx(log(x)) = 1/x
            grad_dict[id(x)] = grad_output / x.data


class Exp(Function):
    """
    Exponential operation.

    Forward: f(x) = exp(x)
    Backward: f'(x) = exp(x)

    Note: This operation can potentially produce very large numbers,
    so care should be taken with input values to avoid overflow.
    """

    @staticmethod
    def forward(ctx: Context, x: Union[Tensor, NDArray[Any], float, int]) -> Tensor:
        """
        Computes the exponential of the input tensor.

        Args:
            ctx: Context object for storing information needed in backward pass
            x: Input tensor or scalar value

        Returns:
            A new tensor containing exp(x)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)

        result = np.exp(x.data)
        ctx.save_for_backward(x)  # Save x for backward pass
        ctx.save_arguments(exp_x=result)  # Save exp(x) as argument
        return Tensor(result)

    @staticmethod
    def backward(
        ctx: Context, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]
    ) -> None:
        """
        Computes the gradient of the exponential function.

        The derivative of exp(x) is exp(x).

        Args:
            ctx: Context containing saved tensors and arguments from forward pass
            grad_output: Gradient from upstream operations
            grad_dict: Dictionary mapping tensor IDs to their gradients
        """
        (x,) = ctx.saved_tensors
        exp_x = ctx.saved_arguments["exp_x"]

        if x.requires_grad:
            # d/dx(exp(x)) = exp(x)
            grad_dict[id(x)] = grad_output * exp_x
