# DLpy/ops/reshape.py
from ..core.function import Function
from ..core.tensor import Tensor
import numpy as np
from typing import Dict

class Reshape(Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        # Save both the input tensor and the target shape
        ctx.save_for_backward(tensor)
        ctx.save_arguments(target_shape=shape)
        # Create and return a new tensor with the reshaped data
        return Tensor(tensor.data.reshape(shape))
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        # Get the original tensor and reshape the gradient back to its shape
        original_tensor, = ctx.saved_tensors
        if original_tensor.requires_grad:
            # Reshape gradient back to the original tensor's shape
            grad_dict[id(original_tensor)] = grad_output.reshape(original_tensor.shape)