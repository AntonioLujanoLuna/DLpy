# layer_norm.py

# layer_norm.py
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core import Module, Tensor

# layer_norm.py


# layer_norm.py


# layer_norm.py


# layer_norm.py


class LayerNorm(Module):
    """
    Applies Layer Normalization over a mini-batch of inputs

    Args:
        normalized_shape: Input shape from an expected input of size
        eps: Small constant for numerical stability
        elementwise_affine: If True, use learnable affine parameters
    """

    def __init__(
        self, normalized_shape: List[int], eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Tensor(np.ones(normalized_shape), requires_grad=True)
            self.bias = Tensor(np.zeros(normalized_shape), requires_grad=True)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # Check input dimensions
        input_shape = x.shape
        ndim = len(input_shape)
        normalized_ndim = len(self.normalized_shape)

        if ndim < normalized_ndim:
            raise ValueError(f"Expected {normalized_ndim}D or higher input, got {ndim}D input")

        for i, s in enumerate(reversed(self.normalized_shape)):
            if input_shape[ndim - i - 1] != s:
                raise ValueError(
                    f"Expected normalized_shape={self.normalized_shape}, "
                    f"got input shape={input_shape}"
                )

        # Calculate statistics over the normalized dimensions
        stats_shape = input_shape[:-normalized_ndim] + (1,) * normalized_ndim
        reduction_axes = tuple(range(ndim - normalized_ndim, ndim))

        mean = np.mean(x.data, axis=reduction_axes, keepdims=True)
        # Use ddof=0 for layer normalization
        var = np.var(x.data, axis=reduction_axes, keepdims=True, ddof=0)

        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)

        # Apply affine transform if specified
        if self.elementwise_affine:
            # Reshape weight and bias to broadcast correctly
            weight_shape = (1,) * (ndim - normalized_ndim) + self.normalized_shape
            bias_shape = (1,) * (ndim - normalized_ndim) + self.normalized_shape

            # Handle broadcasting for weight and bias
            weight = np.broadcast_to(self.weight.data.reshape(weight_shape), x_norm.shape)
            bias = np.broadcast_to(self.bias.data.reshape(bias_shape), x_norm.shape)

            x_norm = x_norm * weight + bias

        return Tensor(x_norm, requires_grad=x.requires_grad)
