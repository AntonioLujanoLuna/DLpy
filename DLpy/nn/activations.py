"""
Activation functions module for DLpy.

This module contains both Function and Module implementations of standard activation functions.
Functions can be used directly (relu(x)), while Modules can be used in Sequential layers (ReLU()).
Each activation function is implemented with full autograd support.
"""

from typing import Dict

import numpy as np

from ..core import Function, Module, Tensor

# Function implementations (for functional usage)


class ReLUFunction(Function):
    """
    Rectified Linear Unit activation function.

    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0 else 0
    """

    @staticmethod
    def forward(ctx, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        ctx.save_for_backward(x)
        return Tensor(np.maximum(0, x.data))

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        (x,) = ctx.saved_tensors
        if x.requires_grad:
            grad = grad_output * (x.data > 0)
            grad_dict[id(x)] = grad


class LeakyReLUFunction(Function):
    """
    Leaky Rectified Linear Unit activation function.

    Forward: f(x) = x if x > 0 else negative_slope * x
    Backward: f'(x) = 1 if x > 0 else negative_slope

    Args:
        negative_slope: Controls slope for negative values. Default: 0.01
    """

    @staticmethod
    def forward(ctx, x, negative_slope: float = 0.01):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        ctx.save_for_backward(x)
        ctx.save_arguments(negative_slope=negative_slope)

        return Tensor(np.where(x.data > 0, x.data, negative_slope * x.data))

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        (x,) = ctx.saved_tensors
        negative_slope = ctx.saved_arguments["negative_slope"]

        if x.requires_grad:
            grad = grad_output * np.where(x.data > 0, 1.0, negative_slope)
            grad_dict[id(x)] = grad


class ELUFunction(Function):
    """
    Exponential Linear Unit activation function.

    Forward: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    Backward: f'(x) = 1 if x > 0 else alpha * exp(x)

    Args:
        alpha: Controls the value to which an ELU saturates for negative inputs. Default: 1.0
    """

    @staticmethod
    def forward(ctx, x, alpha: float = 1.0):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        ctx.save_for_backward(x)
        ctx.save_arguments(alpha=alpha)

        return Tensor(np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1)))

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        (x,) = ctx.saved_tensors
        alpha = ctx.saved_arguments["alpha"]

        if x.requires_grad:
            grad = grad_output * np.where(x.data > 0, 1.0, alpha * np.exp(x.data))
            grad_dict[id(x)] = grad


class GELUFunction(Function):
    """
    Gaussian Error Linear Unit activation function.

    Forward: f(x) = x * Φ(x)
    where Φ(x) is the Gaussian cumulative distribution function.

    This implementation uses the approximation:
    f(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
    """

    @staticmethod
    def forward(ctx, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Constants for the approximation
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        coeff = 0.044715

        # Compute intermediate values
        x_cubed = x.data**3
        inner = sqrt_2_over_pi * (x.data + coeff * x_cubed)
        tanh_inner = np.tanh(inner)

        # Compute output
        result = 0.5 * x.data * (1 + tanh_inner)

        # Save for backward pass
        ctx.save_for_backward(x)
        ctx.save_arguments(tanh_inner=tanh_inner)

        return Tensor(result)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        (x,) = ctx.saved_tensors
        tanh_inner = ctx.saved_arguments["tanh_inner"]

        if x.requires_grad:
            sqrt_2_over_pi = np.sqrt(2 / np.pi)
            coeff = 0.044715

            # Compute derivative
            x_cubed = x.data**3
            sqrt_2_over_pi * (x.data + coeff * x_cubed)

            # d/dx[GELU(x)] = 0.5 * (1 + tanh(inner)) +
            #                 0.5x * (1 - tanh²(inner)) * sqrt(2/π) * (1 + 3 * 0.044715x²)
            grad = 0.5 * (1 + tanh_inner)
            grad += (
                0.5 * x.data * (1 - tanh_inner**2) * sqrt_2_over_pi * (1 + 3 * coeff * x.data**2)
            )

            grad_dict[id(x)] = grad_output * grad


class SigmoidFunction(Function):
    """
    Sigmoid activation function.

    Forward: f(x) = 1 / (1 + exp(-x))
    Backward: f'(x) = f(x) * (1 - f(x))
    """

    @staticmethod
    def forward(ctx, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Compute sigmoid with numerical stability
        x_data = x.data
        exp_neg_x = np.exp(-np.abs(x_data))
        sigmoid_x = np.where(x_data >= 0, 1 / (1 + exp_neg_x), exp_neg_x / (1 + exp_neg_x))

        ctx.save_for_backward(x)
        ctx.save_arguments(sigmoid_x=sigmoid_x)
        return Tensor(sigmoid_x)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        (x,) = ctx.saved_tensors
        sigmoid_x = ctx.saved_arguments["sigmoid_x"]

        if x.requires_grad:
            grad = grad_output * sigmoid_x * (1 - sigmoid_x)
            grad_dict[id(x)] = grad


class TanhFunction(Function):
    """
    Hyperbolic tangent activation function.

    Forward: f(x) = tanh(x)
    Backward: f'(x) = 1 - tanh²(x)
    """

    @staticmethod
    def forward(ctx, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        tanh_x = np.tanh(x.data)
        ctx.save_for_backward(x)
        ctx.save_arguments(tanh_x=tanh_x)
        return Tensor(tanh_x)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        (x,) = ctx.saved_tensors
        tanh_x = ctx.saved_arguments["tanh_x"]

        if x.requires_grad:
            grad = grad_output * (1 - tanh_x**2)
            grad_dict[id(x)] = grad


# Module implementations (for use in Sequential and other Module-based architectures)


class ReLU(Module):
    """Applies the rectified linear unit function element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return ReLUFunction.apply(x)


class LeakyReLU(Module):
    """Applies leaky ReLU function element-wise."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return LeakyReLUFunction.apply(x, self.negative_slope)


class ELU(Module):
    """Applies the exponential linear unit function element-wise."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return ELUFunction.apply(x, self.alpha)


class GELU(Module):
    """Applies the Gaussian Error Linear Units function."""

    def forward(self, x: Tensor) -> Tensor:
        return GELUFunction.apply(x)


class Sigmoid(Module):
    """Applies the sigmoid function element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return SigmoidFunction.apply(x)


class Tanh(Module):
    """Applies the hyperbolic tangent function element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return TanhFunction.apply(x)


# Functional interface (for direct usage)


def relu(x: Tensor) -> Tensor:
    """Applies ReLU activation function."""
    return ReLUFunction.apply(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Applies Leaky ReLU activation function."""
    return LeakyReLUFunction.apply(x, negative_slope)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Applies ELU activation function."""
    return ELUFunction.apply(x, alpha)


def gelu(x: Tensor) -> Tensor:
    """Applies GELU activation function."""
    return GELUFunction.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    """Applies Sigmoid activation function."""
    return SigmoidFunction.apply(x)


def tanh(x: Tensor) -> Tensor:
    """Applies Tanh activation function."""
    return TanhFunction.apply(x)
