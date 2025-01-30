from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core import Function, Tensor
from ..core.context import Context


def _compute_output_shape(
    input_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> Tuple[int, int]:
    """Calculate output shape for pooling operations."""
    H_out = (input_size[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (input_size[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return (H_out, W_out)


def _pad_input(x: NDArray[Any], padding: Tuple[int, int]) -> NDArray[Any]:
    """Add padding to input tensor."""
    if padding[0] == 0 and padding[1] == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
        constant_values=0,
    )


class MaxPool2dFunction(Function):
    """Function implementing 2D max pooling."""

    @staticmethod
    def forward(
        ctx: Context,
        x: Tensor,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Tensor:
        # Save params for backward pass
        ctx.save_arguments(kernel_size=kernel_size, stride=stride, padding=padding)

        # Apply padding
        x_padded = _pad_input(x.data, padding)
        N, C, H, W = x_padded.shape
        kH, kW = kernel_size
        sH, sW = stride

        # Calculate output dimensions
        H_out = ((H - kH) // sH) + 1
        W_out = ((W - kW) // sW) + 1

        # Initialize output and max indices for backward pass
        output = np.zeros((N, C, H_out, W_out))
        max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32)

        # Compute max pooling
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * sH
                        h_end = h_start + kH
                        w_start = w * sW
                        w_end = w_start + kW

                        window = x_padded[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h, w] = np.max(window)
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_indices[n, c, h, w] = [h_start + max_idx[0], w_start + max_idx[1]]

        ctx.save_for_backward(x)
        ctx.save_arguments(max_indices=max_indices)
        return Tensor(output)

    @staticmethod
    def backward(
        ctx: Context, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]
    ) -> None:
        (x,) = ctx.saved_tensors
        max_indices = ctx.saved_arguments["max_indices"]

        if x.requires_grad:
            grad = np.zeros_like(x.data)
            N, C, H_out, W_out = grad_output.shape

            # Distribute gradients to max positions
            for n in range(N):
                for c in range(C):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_max, w_max = max_indices[n, c, h, w]
                            grad[n, c, h_max, w_max] += grad_output[n, c, h, w]

            grad_dict[id(x)] = grad


class AvgPool2dFunction(Function):
    """Function implementing 2D average pooling."""

    @staticmethod
    def forward(
        ctx: Context,
        x: Tensor,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Tensor:
        # Save params for backward pass
        ctx.save_arguments(kernel_size=kernel_size, stride=stride, padding=padding)

        # Apply padding
        x_padded = _pad_input(x.data, padding)
        N, C, H, W = x_padded.shape
        kH, kW = kernel_size
        sH, sW = stride

        # Calculate output dimensions
        H_out = ((H - kH) // sH) + 1
        W_out = ((W - kW) // sW) + 1

        output = np.zeros((N, C, H_out, W_out))

        # Compute average pooling
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * sH
                        h_end = h_start + kH
                        w_start = w * sW
                        w_end = w_start + kW
                        window = x_padded[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h, w] = np.mean(window)

        ctx.save_for_backward(x)
        return Tensor(output)

    @staticmethod
    def backward(
        ctx: Context, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]
    ) -> None:
        (x,) = ctx.saved_tensors
        kernel_size = ctx.saved_arguments["kernel_size"]
        stride = ctx.saved_arguments["stride"]
        ctx.saved_arguments["padding"]

        if x.requires_grad:
            kH, kW = kernel_size
            grad = np.zeros_like(x.data)
            N, C, H_out, W_out = grad_output.shape

            # Distribute gradients uniformly within each pooling window
            scale = 1.0 / (kH * kW)
            for n in range(N):
                for c in range(C):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * stride[0]
                            h_end = h_start + kH
                            w_start = w * stride[1]
                            w_end = w_start + kW
                            grad[n, c, h_start:h_end, w_start:w_end] += (
                                grad_output[n, c, h, w] * scale
                            )

            grad_dict[id(x)] = grad
