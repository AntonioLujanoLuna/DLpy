# DLpy/ops/reshape.py
from ..core.function import Function
from ..core.tensor import Tensor
import numpy as np
from typing import Dict

class Reshape(Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        # Ensure all dimensions are integers
        final_shape = tuple(int(d) if d != -1 else -1 for d in shape)
        
        ctx.save_for_backward(tensor)
        ctx.save_arguments(target_shape=final_shape)
        return Tensor(tensor.data.reshape(final_shape))
    
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        original_tensor, = ctx.saved_tensors
        if original_tensor.requires_grad:
            grad_dict[id(original_tensor)] = grad_output.reshape(original_tensor.shape)