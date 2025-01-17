from typing import Dict  # Add this import at the top
from ..core.function import Function
from ..core.tensor import Tensor
import numpy as np

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not isinstance(a, Tensor):
            a = Tensor(a)
        if not isinstance(b, Tensor):
            b = Tensor(b)
            
        shape_a = a.data.shape
        shape_b = b.data.shape

        # Check valid broadcasting manually
        if len(shape_a) == 2 and shape_a[0] == 1 and len(shape_b) == 1:
            # Special case: (1,N) matrix with (M,) vector requires N==M
            if shape_a[1] != shape_b[0]:
                raise ValueError(f"Cannot broadcast shape {shape_a} with {shape_b}")
        elif len(shape_a) == 1 and len(shape_b) == 2 and shape_b[0] == 1:
            # Special case: (N,) vector with (1,M) matrix requires N==M
            if shape_a[0] != shape_b[1]:
                raise ValueError(f"Cannot broadcast shape {shape_a} with {shape_b}")
                
        # Save tensors for backward pass
        ctx.save_for_backward(a, b)
        
        # If we get here, try the operation
        try:
            result = a.data + b.data
            return Tensor(result)
        except ValueError:
            raise ValueError(f"Cannot broadcast shape {shape_a} with {shape_b}")
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        a, b = ctx.saved_tensors

        if a.requires_grad:
            grad_a = grad_output
            grad_a = Add._reduce_grad(grad_a, a.data.shape)
            if id(a) not in grad_dict or grad_dict[id(a)] is None:
                grad_dict[id(a)] = grad_a
            else:
                grad_dict[id(a)] += grad_a  # Accumulate gradients

        if b.requires_grad:
            grad_b = grad_output
            grad_b = Add._reduce_grad(grad_b, b.data.shape)
            if id(b) not in grad_dict or grad_dict[id(b)] is None:
                grad_dict[id(b)] = grad_b
            else:
                grad_dict[id(b)] += grad_b  # Accumulate gradients

    @staticmethod
    def _reduce_grad(grad, target_shape):
        """
        Reduces the gradient to match the target shape by summing over broadcasted dimensions.
        """
        # Convert target_shape to a tuple if it's not
        if not isinstance(target_shape, tuple):
            target_shape = tuple(target_shape)
        
        # Align the dimensions by prepending 1s if necessary
        grad_shape = grad.shape
        target_shape = (1,) * (len(grad_shape) - len(target_shape)) + target_shape
        for axis, (grad_dim, target_dim) in enumerate(zip(grad_shape, target_shape)):
            if target_dim == 1 and grad_dim != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

class Multiply(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not isinstance(a, Tensor):
            a = Tensor(a)
        if not isinstance(b, Tensor):
            b = Tensor(b)
            
        shape_a = a.data.shape
        shape_b = b.data.shape
        
        # Check if shapes can be broadcast according to NumPy rules
        try:
            # Test broadcast compatibility without actually performing the operation
            np.broadcast_shapes(shape_a, shape_b)
            # If we get here, shapes are compatible
            result = a.data * b.data
            ctx.save_for_backward(a, b)
            return Tensor(result)
        except ValueError:
            raise ValueError(f"Cannot broadcast shape {shape_a} with {shape_b}")
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        a, b = ctx.saved_tensors

        if a.requires_grad:
            grad_a = grad_output * b.data
            grad_a = Multiply._reduce_grad(grad_a, a.data.shape)
            if id(a) not in grad_dict or grad_dict[id(a)] is None:
                grad_dict[id(a)] = grad_a
            else:
                grad_dict[id(a)] += grad_a  # Accumulate gradients

        if b.requires_grad:
            grad_b = grad_output * a.data
            grad_b = Multiply._reduce_grad(grad_b, b.data.shape)
            if id(b) not in grad_dict or grad_dict[id(b)] is None:
                grad_dict[id(b)] = grad_b
            else:
                grad_dict[id(b)] += grad_b  # Accumulate gradients

    @staticmethod
    def _reduce_grad(grad, target_shape):
        """
        Reduces the gradient to match the target shape by summing over broadcasted dimensions.
        """
        # Convert target_shape to a tuple if it's not
        if not isinstance(target_shape, tuple):
            target_shape = tuple(target_shape)
        
        # Align the dimensions by prepending 1s if necessary
        grad_shape = grad.shape
        target_shape = (1,) * (len(grad_shape) - len(target_shape)) + target_shape
        for axis, (grad_dim, target_dim) in enumerate(zip(grad_shape, target_shape)):
            if target_dim == 1 and grad_dim != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad
