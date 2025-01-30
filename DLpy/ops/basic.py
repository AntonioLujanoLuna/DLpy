from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from ..core.function import Function
from ..core.tensor import Tensor


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
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
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
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
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


class MatMul(Function):
    """Matrix multiplication operation with support for batched operations."""

    @staticmethod
    def forward(ctx, a, b):
        if not isinstance(a, Tensor):
            a = Tensor(a)
        if not isinstance(b, Tensor):
            b = Tensor(b)

        ctx.save_for_backward(a, b)
        return Tensor(np.matmul(a.data, b.data))

    @staticmethod
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
        a, b = ctx.saved_tensors

        if a.requires_grad:
            # Handle batched matrix multiplication for gradients
            if len(grad_output.shape) == len(b.data.shape):
                grad_a = np.matmul(grad_output, b.data.swapaxes(-1, -2))
            else:
                # Reshape grad_output to match dimensions if necessary
                grad_a = np.matmul(
                    grad_output.reshape(-1, grad_output.shape[-1]),
                    b.data.reshape(-1, b.data.shape[-1]).T,
                )
                grad_a = grad_a.reshape(grad_output.shape[:-1] + (b.data.shape[-2],))

            if id(a) not in grad_dict:
                grad_dict[id(a)] = grad_a
            else:
                grad_dict[id(a)] += grad_a

        if b.requires_grad:
            # Handle batched matrix multiplication for gradients
            if len(grad_output.shape) == len(a.data.shape):
                grad_b = np.matmul(a.data.swapaxes(-1, -2), grad_output)
            else:
                # Reshape grad_output to match dimensions if necessary
                grad_b = np.matmul(
                    a.data.reshape(-1, a.data.shape[-2]).T,
                    grad_output.reshape(-1, grad_output.shape[-1]),
                )
                grad_b = grad_b.reshape(
                    a.data.shape[:-2] + (a.data.shape[-1], grad_output.shape[-1])
                )

            if id(b) not in grad_dict:
                grad_dict[id(b)] = grad_b
            else:
                grad_dict[id(b)] += grad_b


class Softmax(Function):
    """Softmax activation function operation."""

    @staticmethod
    def forward(ctx, x, dim=-1):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Convert negative dim to positive for consistency
        if dim < 0:
            dim = len(x.data.shape) + dim

        # Compute max along specified dimension for numerical stability
        x_max = np.max(x.data, axis=dim, keepdims=True)
        exp_x = np.exp(x.data - x_max)
        softmax_out = exp_x / np.sum(exp_x, axis=dim, keepdims=True)

        # Save inputs and outputs for backward pass
        ctx.save_for_backward(x, Tensor(softmax_out))
        ctx.dim = dim  # Save dim as a regular integer

        return Tensor(softmax_out)

    @staticmethod
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
        x, softmax_out = ctx.saved_tensors
        dim = ctx.dim  # Get dim from context as regular integer

        if x.requires_grad:
            # Gradient of softmax:
            # dsoftmax_i/dx_j = softmax_i * (1{i=j} - softmax_j)
            grad_x = softmax_out.data * (
                grad_output - np.sum(grad_output * softmax_out.data, axis=dim, keepdims=True)
            )

            if id(x) not in grad_dict:
                grad_dict[id(x)] = grad_x
            else:
                grad_dict[id(x)] += grad_x


class Clip(Function):
    """Clips tensor values between minimum and maximum."""

    @staticmethod
    def forward(ctx, x, min_val, max_val):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Save input and clip values for backward pass
        ctx.save_for_backward(x)
        ctx.min_val = min_val
        ctx.max_val = max_val

        return Tensor(np.clip(x.data, min_val, max_val))

    @staticmethod
    def backward(ctx, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]) -> None:
        (x,) = ctx.saved_tensors
        min_val = ctx.min_val
        max_val = ctx.max_val

        if x.requires_grad:
            # Gradient is zero where input was clipped
            grad = grad_output * ((x.data >= min_val) & (x.data <= max_val))

            if id(x) not in grad_dict:
                grad_dict[id(x)] = grad
            else:
                grad_dict[id(x)] += grad  # Accumulate gradients
