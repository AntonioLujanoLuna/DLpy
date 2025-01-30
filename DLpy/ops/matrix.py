from typing import Dict, Optional, Tuple

import numpy as np

from ..core import Function, Tensor


class Transpose(Function):
    @staticmethod
    def forward(ctx, x, axes: Optional[Tuple[int, ...]] = None):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        ctx.save_for_backward(x)
        ctx.save_arguments(axes=axes)

        if axes is None:
            return Tensor(np.transpose(x.data))
        return Tensor(np.transpose(x.data, axes))

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        (x,) = ctx.saved_tensors
        axes = ctx.saved_arguments["axes"]

        if x.requires_grad:
            if axes is None:
                # For standard transpose, just transpose the gradient
                grad_dict[id(x)] = np.transpose(grad_output)
            else:
                # For specific axes, need to invert the permutation
                inverse_axes = np.argsort(axes)
                grad_dict[id(x)] = np.transpose(grad_output, inverse_axes)


class Compare(Function):
    """Base class for comparison operations"""

    @staticmethod
    def _compare(op, x1, x2):
        if not isinstance(x1, Tensor):
            x1 = Tensor(x1)
        if not isinstance(x2, Tensor):
            x2 = Tensor(x2)

        return Tensor(op(x1.data, x2.data))

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        # Comparison operations have no gradient
        pass


class Greater(Compare):
    @staticmethod
    def forward(ctx, x1, x2):
        return Compare._compare(np.greater, x1, x2)


class GreaterEqual(Compare):
    @staticmethod
    def forward(ctx, x1, x2):
        return Compare._compare(np.greater_equal, x1, x2)


class Less(Compare):
    @staticmethod
    def forward(ctx, x1, x2):
        return Compare._compare(np.less, x1, x2)


class LessEqual(Compare):
    @staticmethod
    def forward(ctx, x1, x2):
        return Compare._compare(np.less_equal, x1, x2)


class Equal(Compare):
    @staticmethod
    def forward(ctx, x1, x2):
        return Compare._compare(np.equal, x1, x2)


class NotEqual(Compare):
    @staticmethod
    def forward(ctx, x1, x2):
        return Compare._compare(np.not_equal, x1, x2)
