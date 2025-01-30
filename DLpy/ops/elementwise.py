from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..core import Function, Tensor


class Log(Function):
    """
    Natural logarithm operation.

    Forward: f(x) = ln(x)
    Backward: f'(x) = 1/x
    """

    @staticmethod
    def forward(ctx, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Check for negative values
        if np.any(x.data <= 0):
            raise ValueError("Log of negative numbers or zero is undefined")

        ctx.save_for_backward(x)
        return Tensor(np.log(x.data))

    @staticmethod
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
        (x,) = ctx.saved_tensors
        if x.requires_grad:
            # d/dx(log(x)) = 1/x
            grad_dict[id(x)] = grad_output / x.data


class Exp(Function):
    """
    Exponential operation.

    Forward: f(x) = exp(x)
    Backward: f'(x) = exp(x)
    """

    @staticmethod
    def forward(ctx, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        result = np.exp(x.data)
        ctx.save_for_backward(x)  # Save x for backward pass
        ctx.save_arguments(exp_x=result)  # Save exp(x) as argument
        return Tensor(result)

    @staticmethod
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
        (x,) = ctx.saved_tensors
        exp_x = ctx.saved_arguments["exp_x"]

        if x.requires_grad:
            # d/dx(exp(x)) = exp(x)
            grad_dict[id(x)] = grad_output * exp_x
