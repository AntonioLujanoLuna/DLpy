# DLpy/ops/reshape.py

# DLpy/ops/reshape.py
# DLpy/ops/reshape.py
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from numpy.typing import NDArray

from ..core.function import Function
from ..core.tensor import Tensor

# DLpy/ops/reshape.py


# DLpy/ops/reshape.py


# DLpy/ops/reshape.py


class Reshape(Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        # Ensure all dimensions are integers
        final_shape = tuple(int(d) if d != -1 else -1 for d in shape)

        ctx.save_for_backward(tensor)
        ctx.save_arguments(target_shape=final_shape)
        return Tensor(tensor.data.reshape(final_shape))

    @staticmethod
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
        (original_tensor,) = ctx.saved_tensors
        if original_tensor.requires_grad:
            grad_dict[id(original_tensor)] = grad_output.reshape(original_tensor.shape)
