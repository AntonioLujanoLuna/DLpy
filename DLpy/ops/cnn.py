from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core import Context, Function, Tensor


class ConvMode:
    """Enumeration of convolution modes."""

    STANDARD = "standard"
    TRANSPOSED = "transposed"
    DEFORMABLE = "deformable"


def _validate_conv_params(
    x_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    mode: str = ConvMode.STANDARD,
    offset: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> None:
    """Validates convolution parameters."""
    N, C_in, H, W = x_shape
    C_out, C_in_per_group, kH, kW = weight_shape

    # Basic validations for all modes
    if mode not in [ConvMode.STANDARD, ConvMode.TRANSPOSED, ConvMode.DEFORMABLE]:
        raise ValueError(f"Invalid convolution mode: {mode}")

    # Validate groups configuration
    if C_in % groups != 0:
        raise ValueError(f"Input channels ({C_in}) must be divisible by groups ({groups})")
    if C_out % groups != 0:
        raise ValueError(f"Output channels ({C_out}) must be divisible by groups ({groups})")

    # Validate kernel dimensions
    if kH <= 0 or kW <= 0:
        raise ValueError(f"Kernel dimensions must be positive, got ({kH}, {kW})")

    # Validate stride and dilation
    if any(s <= 0 for s in stride):
        raise ValueError(f"Stride values must be positive, got {stride}")
    if any(d <= 0 for d in dilation):
        raise ValueError(f"Dilation values must be positive, got {dilation}")

    # Validate padding
    if any(p < 0 for p in padding):
        raise ValueError(f"Padding values must be non-negative, got {padding}")

    if mode == ConvMode.TRANSPOSED:
        # For transposed conv:
        # - x shape is (N, C_in, H, W)
        # - weight shape should be (C_out, C_in, kH, kW)
        # - C_in from x should match C_in_per_group * groups from weight
        if C_in_per_group != C_in // groups:
            raise ValueError(
                f"For transposed conv, expected {C_in // groups} input channels per group, "
                f"got {C_in_per_group}"
            )

        # Calculate and validate output size
        H_out = (H - 1) * stride[0] - 2 * padding[0] + kH
        W_out = (W - 1) * stride[1] - 2 * padding[1] + kW
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Transposed conv output size would be negative or zero: ({H_out}, {W_out})"
            )

    else:  # Standard and deformable validation
        # Validate channels per group
        if C_in_per_group != C_in // groups:
            raise ValueError(
                f"Expected {C_in // groups} input channels per group, got {C_in_per_group}"
            )

        # Calculate and validate output size
        H_out = ((H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0]) + 1
        W_out = ((W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1]) + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Conv output size would be negative or zero: ({H_out}, {W_out})")

    if mode == ConvMode.DEFORMABLE:
        # Validate offset tensor presence and shape
        if offset is None and weight is not None:
            offset = getattr(weight, "offset", None)

        if offset is None:
            raise ValueError("Deformable convolution requires offset parameter")

        # Calculate output size for offset validation
        H_out = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

        # Validate offset tensor shape
        expected_offset_shape = (N, 2 * kH * kW, H_out, W_out)
        if offset.shape != expected_offset_shape:
            raise ValueError(f"Expected offset shape {expected_offset_shape}, got {offset.shape}")

        # Validate mask tensor if present
        if mask is not None:
            expected_mask_shape = (N, kH * kW, H_out, W_out)
            if mask.shape != expected_mask_shape:
                raise ValueError(f"Expected mask shape {expected_mask_shape}, got {mask.shape}")


def _pad_input(x: NDArray[Any], padding: Tuple[int, int]) -> NDArray[Any]:
    """
    Pads input tensor with zeros.

    Args:
        x: Input tensor
        padding: (padding_height, padding_width)
    """
    if padding[0] == 0 and padding[1] == 0:
        return x
    pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
    return np.pad(x, pad_width, mode="constant", constant_values=0)


def _get_output_shape(
    input_shape: Tuple[int, ...],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    mode: str = ConvMode.STANDARD,
    output_padding: Tuple[int, int] = (0, 0),
) -> Tuple[int, int]:
    """
    Calculates output shape for different convolution types.

    Args:
        input_shape: Shape of input tensor (N, C, H, W)
        kernel_size: Size of convolution kernel (kH, kW)
        stride: Stride of convolution (sH, sW)
        padding: Zero-padding size (pH, pW)
        dilation: Dilation rate (dH, dW)
        mode: Convolution mode (standard, transposed, or deformable)
        output_padding: Additional size added to output shape (only for transposed conv)

    Returns:
        Tuple of output height and width (H_out, W_out)
    """
    if mode == ConvMode.STANDARD:
        H = (input_shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[
            0
        ] + 1
        W = (input_shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[
            1
        ] + 1
    elif mode == ConvMode.TRANSPOSED:
        H = (input_shape[2] - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
        W = (input_shape[3] - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]
    else:  # Deformable follows standard conv shape
        H = (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        W = (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return H, W


def _get_deformable_offsets(
    offset_data: NDArray[Any],
    kernel_size: Tuple[int, int],
    input_shape: Tuple[int, ...],
    dilation: Tuple[int, int] = (1, 1),
) -> NDArray[Any]:
    """
    Computes sampling locations for deformable convolution.

    Args:
        offset_data: Offset tensor of shape (N, 2*kH*kW, H_out, W_out)
        kernel_size: Tuple of (kH, kW)
        input_shape: Shape of input tensor (N, C, H, W)
        dilation: Dilation rate

    Returns:
        Sampling locations of shape (N, H_out*W_out, kH*kW, 2)
    """
    N, _, H_out, W_out = offset_data.shape
    kH, kW = kernel_size

    # Generate base grid for the kernel
    h_range = np.arange(kH) * dilation[0]
    w_range = np.arange(kW) * dilation[1]
    h_grid, w_grid = np.meshgrid(h_range, w_range, indexing="ij")
    kernel_grid = np.stack([h_grid, w_grid], axis=-1)  # (kH, kW, 2)
    kernel_grid = kernel_grid.reshape(-1, 2)  # (kH*kW, 2)

    # Reshape offsets
    offset = offset_data.reshape(N, 2, kH * kW, H_out, W_out)
    offset = offset.transpose(0, 3, 4, 2, 1)  # (N, H_out, W_out, kH*kW, 2)
    offset = offset.reshape(N, H_out * W_out, kH * kW, 2)

    # Add base grid to offsets
    kernel_grid = np.expand_dims(np.expand_dims(kernel_grid, 0), 0)  # (1, 1, kH*kW, 2)
    sampling_locations = kernel_grid + offset

    return sampling_locations


def _im2col_dilated(
    x: NDArray[Any],
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    dilation: Tuple[int, ...],
    padding: Tuple[int, ...],
    mode: str = ConvMode.STANDARD,
    sampling_locations: Optional[NDArray[Any]] = None,
) -> NDArray[Any]:
    """Rearranges dilated image blocks into columns."""
    N, C, H, W = x.shape
    kH, kW = kernel_size

    # Calculate output dimensions
    H_out = ((H - dilation[0] * (kH - 1) - 1) // stride[0]) + 1
    W_out = ((W - dilation[1] * (kW - 1) - 1) // stride[1]) + 1

    # Initialize output array
    # For standard convolution:
    # - Each column represents a dot product of the kernel with a specific output position
    # - Number of rows = C * kH * kW (all values needed for one output value)
    # - Number of columns = N * H_out * W_out (total number of output values)
    cols = np.zeros((C * kH * kW, N * H_out * W_out))

    # Process each input position in the kernel
    for c in range(C):
        for kh in range(kH):
            for kw in range(kW):
                h_start = np.arange(H_out) * stride[0]
                w_start = np.arange(W_out) * stride[1]

                h_offset = kh * dilation[0]
                w_offset = kw * dilation[1]

                h_pos, w_pos = np.meshgrid(h_start + h_offset, w_start + w_offset, indexing="ij")
                h_pos = h_pos.reshape(-1)
                w_pos = w_pos.reshape(-1)

                row_idx = c * kH * kW + kh * kW + kw
                for n in range(N):
                    col_idx = n * H_out * W_out + np.arange(H_out * W_out)
                    cols[row_idx, col_idx] = x[n, c, h_pos, w_pos]

    print(f"im2col output shape: {cols.shape}")
    print(f"Expected reshape: C*kH*kW={C*kH*kW}, N={N}, H_out={H_out}, W_out={W_out}")
    return cols


def _get_output_size(
    input_shape: Tuple[int, ...],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    mode: str,
) -> Tuple[int, int]:
    """Calculate output dimensions for different convolution modes."""
    _, _, H, W = input_shape
    kH, kW = kernel_size

    if mode == ConvMode.TRANSPOSED:
        H_out = (H - 1) * stride[0] - 2 * padding[0] + kH
        W_out = (W - 1) * stride[1] - 2 * padding[1] + kW
    else:
        H_out = ((H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0]) + 1
        W_out = ((W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1]) + 1

    return H_out, W_out


def _col2im_dilated(
    cols: NDArray[Any],
    output_size: Tuple[int, ...],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    mode: str = ConvMode.STANDARD,
) -> NDArray[Any]:
    """Convert columns back to dilated image."""
    N, C, H, W = output_size
    kH, kW = kernel_size

    # Calculate output dimensions based on mode
    if mode == ConvMode.TRANSPOSED:
        H_out = H * stride[0]
        W_out = W * stride[1]
    else:
        H_out = ((H - dilation[0] * (kH - 1) - 1) // stride[0]) + 1
        W_out = ((W - dilation[1] * (kW - 1) - 1) // stride[1]) + 1

    output = np.zeros(output_size)
    weights = np.zeros(output_size)  # For averaging overlapping values

    if mode == ConvMode.TRANSPOSED:
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    for kh in range(kH):
                        for kw in range(kW):
                            h_out = h * stride[0] + kh
                            w_out = w * stride[1] + kw

                            if h_out < H_out and w_out < W_out:
                                col_idx = c * kH * kW + kh * kW + kw
                                row_idx = h * W + w

                                for n in range(N):
                                    output[n, c, h_out, w_out] += cols[col_idx, n * H * W + row_idx]
                                    weights[n, c, h_out, w_out] += 1
    else:
        for h_out in range(H_out):
            for w_out in range(W_out):
                for c in range(C):
                    for i in range(kH):
                        for j in range(kW):
                            h_in = h_out * stride[0] + i * dilation[0]
                            w_in = w_out * stride[1] + j * dilation[1]

                            if 0 <= h_in < H and 0 <= w_in < W:
                                col_idx = c * kH * kW + i * kW + j
                                row_idx = h_out * W_out + w_out

                                for n in range(N):
                                    output[n, c, h_in, w_in] += cols[
                                        col_idx, n * H_out * W_out + row_idx
                                    ]
                                    weights[n, c, h_in, w_in] += 1

    # Average overlapping values
    np.divide(output, weights, out=output, where=weights != 0)
    return output


def _compute_conv_output_shape(
    input_size: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> int:
    """Computes output dimension for a single axis."""
    numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1


def _compute_conv_grad_input_padding(
    grad_output_size: int,
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Computes padding needed for gradient computation."""
    grad_input_padding = kernel_size - 1 - padding
    return grad_input_padding


def _compute_output_padding(
    input_size: int, output_size: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> int:
    """Computes additional padding needed for transposed convolution."""
    expected_output = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    return output_size - expected_output


def _unfold(
    input_tensor: NDArray[Any],
    kernel_size: Tuple[int, ...],
    dilation: Tuple[int, ...],
    padding: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> NDArray[Any]:
    """Extracts sliding local blocks from input tensor."""
    N, C, H, W = input_tensor.shape
    kH, kW = kernel_size

    # Apply padding if needed
    if padding[0] > 0 or padding[1] > 0:
        input_tensor = np.pad(
            input_tensor,
            ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
            mode="constant",
        )

    # Calculate output dimensions
    H_out = ((H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0]) + 1
    W_out = ((W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1]) + 1

    # Initialize output array with correct shape
    output = np.zeros((C * kH * kW, N * H_out * W_out))

    # Extract patches
    for h in range(H_out):
        for w in range(W_out):
            for i in range(kH):
                for j in range(kW):
                    h_start = h * stride[0] + i * dilation[0]
                    w_start = w * stride[1] + j * dilation[1]

                    # Extract patch for all channels and batches
                    patch = input_tensor[:, :, h_start : h_start + 1, w_start : w_start + 1]

                    # Place in output array
                    row_idx = (i * kW + j) * C + np.arange(C)
                    col_idx = h * W_out + w + np.arange(N) * H_out * W_out
                    output[row_idx[:, None], col_idx] = patch.reshape(N, C).T

    return output


def _fold(
    input: NDArray[Any],
    output_size: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    dilation: Tuple[int, ...],
    padding: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> NDArray[Any]:
    """Combines an array of sliding local blocks into a large tensor."""
    H, W = output_size
    kH, kW = kernel_size
    C = input.shape[0] // (kH * kW)
    N = input.shape[1] // ((H + 2 * padding[0] - kH + 1) * (W + 2 * padding[1] - kW + 1))

    # Initialize output tensor
    output = np.zeros((N, C, H + 2 * padding[0], W + 2 * padding[1]))
    divisor = np.zeros_like(output)  # For averaging overlapping values

    # Calculate output dimensions
    H_out = ((H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0]) + 1
    W_out = ((W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1]) + 1

    # Fold patches back
    for h in range(H_out):
        for w in range(W_out):
            for i in range(kH):
                for j in range(kW):
                    h_start = h * stride[0] + i * dilation[0]
                    w_start = w * stride[1] + j * dilation[1]

                    row_idx = (i * kW + j) * C + np.arange(C)
                    col_idx = h * W_out + w + np.arange(N) * H_out * W_out

                    patch = input[row_idx[:, None], col_idx].T.reshape(N, C, 1, 1)
                    output[:, :, h_start : h_start + 1, w_start : w_start + 1] += patch
                    divisor[:, :, h_start : h_start + 1, w_start : w_start + 1] += 1

    # Average overlapping values
    output = np.divide(output, divisor, where=divisor != 0)

    # Remove padding if necessary
    if padding[0] > 0 or padding[1] > 0:
        output = output[
            :,
            :,
            padding[0] : -padding[0] if padding[0] > 0 else None,
            padding[1] : -padding[1] if padding[1] > 0 else None,
        ]

    return output


def _dilate(input: NDArray[Any], dilation: Tuple[int, ...]) -> NDArray[Any]:
    """
    Dilates the input tensor by inserting zeros between elements.

    Args:
        input: Input tensor
        dilation: Dilation factors for each dimension

    Returns:
        Dilated tensor
    """
    if all(d == 1 for d in dilation):
        return input

    N, C, H, W = input.shape
    dH, dW = dilation

    H_dilated = H + (H - 1) * (dH - 1)
    W_dilated = W + (W - 1) * (dW - 1)

    output = np.zeros((N, C, H_dilated, W_dilated))
    output[:, :, ::dH, ::dW] = input

    return output


def _bilinear_interpolate(
    input: NDArray[Any], points: NDArray[Any], align_corners: bool = True
) -> NDArray[Any]:
    """
    Performs bilinear interpolation on the input tensor at specified points.

    Args:
        input: Input tensor (N, C, H, W)
        points: Points to sample (N, P, 2) in normalized coordinates [-1, 1]
        align_corners: Whether to align corners in interpolation

    Returns:
        Interpolated values (N, C, P)
    """
    N, C, H, W = input.shape
    _, P, _ = points.shape

    # Convert normalized coordinates to pixel coordinates
    if align_corners:
        x = (points[..., 0] + 1) * (W - 1) / 2
        y = (points[..., 1] + 1) * (H - 1) / 2
    else:
        x = ((points[..., 0] + 1) * W - 1) / 2
        y = ((points[..., 1] + 1) * H - 1) / 2

    # Get corner indices
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    # Clip to image boundaries
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # Calculate interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Gather corner values
    Ia = np.zeros((N, C, P))
    Ib = np.zeros((N, C, P))
    Ic = np.zeros((N, C, P))
    Id = np.zeros((N, C, P))

    for n in range(N):
        for p in range(P):
            Ia[n, :, p] = input[n, :, y0[n, p], x0[n, p]]
            Ib[n, :, p] = input[n, :, y1[n, p], x0[n, p]]
            Ic[n, :, p] = input[n, :, y0[n, p], x1[n, p]]
            Id[n, :, p] = input[n, :, y1[n, p], x1[n, p]]

    # Reshape weights for broadcasting
    wa = wa.reshape(N, 1, P)
    wb = wb.reshape(N, 1, P)
    wc = wc.reshape(N, 1, P)
    wd = wd.reshape(N, 1, P)

    # Interpolate
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


def _bilinear_interpolate_gradient(
    grad_output: NDArray[Any],
    points: NDArray[Any],
    input_size: Tuple[int, ...],
    input_tensor: NDArray[Any],
    align_corners: bool = True,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Computes gradients for bilinear interpolation.

    Args:
        grad_output: Gradient of loss with respect to interpolated values (can be any shape)
        points: Points that were sampled (N, P, 2) in normalized coordinates [-1, 1]
        input_size: Size of the input tensor (H, W)
        input_tensor: The input tensor being sampled from (N, C, H, W)
        align_corners: Whether corners were aligned in interpolation

    Returns:
        Tuple of:
            - grad_input: Gradient with respect to input tensor (N, C, H, W)
            - grad_points: Gradient with respect to sampling points (N, P, 2)

    Raises:
        ValueError: If input shapes are incompatible
    """
    # Ensure points is properly shaped first
    if points.ndim == 2:
        points = points.reshape(1, *points.shape)

    # Get number of points from points tensor
    N, P, _ = points.shape

    # Ensure grad_output is properly shaped (N, C, P) to match points
    if grad_output.ndim == 1:
        grad_output = grad_output.reshape(1, 1, -1)
    elif grad_output.ndim == 2:
        grad_output = grad_output.reshape(1, -1, 1)

    # Broadcast grad_output to match number of points if necessary
    if grad_output.shape[2] == 1:
        grad_output = np.broadcast_to(grad_output, (grad_output.shape[0], grad_output.shape[1], P))
    elif grad_output.shape[2] != P:
        raise ValueError(
            f"Gradient shape {grad_output.shape} cannot be broadcast to number of points {P}"
        )

    C = grad_output.shape[1]
    H, W = input_size

    # Validate input_tensor shape
    if input_tensor.shape[2:] != input_size:
        raise ValueError(
            f"Input tensor spatial dimensions {input_tensor.shape[2:]} "
            f"don't match input_size {input_size}"
        )

    # Convert normalized coordinates to pixel coordinates
    if align_corners:
        x = (points[..., 0] + 1) * (W - 1) / 2
        y = (points[..., 1] + 1) * (H - 1) / 2
    else:
        x = ((points[..., 0] + 1) * W - 1) / 2
        y = ((points[..., 1] + 1) * H - 1) / 2

    # Get corner indices
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    # Clip to image boundaries
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # Compute weights for bilinear interpolation
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Reshape weights for broadcasting with channel dimension
    wa = wa[..., None]  # Shape: (N, P, 1)
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]

    # Initialize gradients
    grad_input = np.zeros((N, C, H, W))
    grad_points = np.zeros_like(points)  # Shape: (N, P, 2)

    # Compute gradients with respect to input
    for n in range(N):
        for c in range(C):
            grad_chan = grad_output[n, c]  # Shape: (P,)
            for p in range(P):
                grad = grad_chan[p]
                grad_input[n, c, y0[n, p], x0[n, p]] += grad * wa[n, p, 0]
                grad_input[n, c, y1[n, p], x0[n, p]] += grad * wb[n, p, 0]
                grad_input[n, c, y0[n, p], x1[n, p]] += grad * wc[n, p, 0]
                grad_input[n, c, y1[n, p], x1[n, p]] += grad * wd[n, p, 0]

    # Compute scaling factors for coordinate gradients
    if align_corners:
        dx = (W - 1) / 2
        dy = (H - 1) / 2
    else:
        dx = W / 2
        dy = H / 2

    # Compute gradients with respect to sampling points
    for n in range(N):
        for p in range(P):
            grad = grad_output[n, :, p].sum()  # Sum over channels

            # Gradient with respect to x
            gx = (
                grad
                * (
                    (y1[n, p] - y[n, p])
                    * (
                        input_tensor[n, :, y0[n, p], x1[n, p]]
                        - input_tensor[n, :, y0[n, p], x0[n, p]]
                    ).sum()
                    + (y[n, p] - y0[n, p])
                    * (
                        input_tensor[n, :, y1[n, p], x1[n, p]]
                        - input_tensor[n, :, y1[n, p], x0[n, p]]
                    ).sum()
                )
                * dx
            )

            # Gradient with respect to y
            gy = (
                grad
                * (
                    (x1[n, p] - x[n, p])
                    * (
                        input_tensor[n, :, y1[n, p], x0[n, p]]
                        - input_tensor[n, :, y0[n, p], x0[n, p]]
                    ).sum()
                    + (x[n, p] - x0[n, p])
                    * (
                        input_tensor[n, :, y1[n, p], x1[n, p]]
                        - input_tensor[n, :, y0[n, p], x1[n, p]]
                    ).sum()
                )
                * dy
            )

            grad_points[n, p] = [gx, gy]

    return grad_input, grad_points


def _generate_grid(
    batch_size: int, height: int, width: int, align_corners: bool = True
) -> NDArray[Any]:
    """
    Generates a coordinate grid for grid sampling.

    Args:
        batch_size: Number of samples in batch
        height: Height of the grid
        width: Width of the grid
        align_corners: Whether to align corners

    Returns:
        Grid tensor of shape (N, H, W, 2) with normalized coordinates
    """
    if align_corners:
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
    else:
        x = np.linspace(-1 + (1 / width), 1 - (1 / width), width)
        y = np.linspace(-1 + (1 / height), 1 - (1 / height), height)

    x_coords, y_coords = np.meshgrid(x, y)
    grid = np.stack([x_coords, y_coords], axis=-1)
    grid = np.tile(grid[None], (batch_size, 1, 1, 1))

    return grid


def _deform_grid(grid: NDArray[Any], offset: NDArray[Any]) -> NDArray[Any]:
    """
    Deforms a regular grid using offset values.

    Args:
        grid: Regular coordinate grid (N, H, W, 2)
        offset: Offset values for deformation (N, 2, H, W)

    Returns:
        Deformed grid (N, H, W, 2)
    """
    N, H, W, _ = grid.shape

    # Reshape offset to match grid shape
    offset = offset.transpose(0, 2, 3, 1)

    # Add offset to grid
    deformed_grid = grid + offset

    # Clamp values to [-1, 1] to ensure valid sampling
    return np.clip(deformed_grid, -1, 1)


def _modulated_deform_grid(
    grid: NDArray[Any], offset: NDArray[Any], mask: NDArray[Any]
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Deforms a regular grid using offset values and modulation mask.
    Used in Deformable ConvNets v2.

    Args:
        grid: Regular coordinate grid (N, H, W, 2)
        offset: Offset values for deformation (N, 2, H, W)
        mask: Modulation mask (N, 1, H, W)

    Returns:
        Tuple of deformed grid and modulation mask
    """
    # Deform grid
    deformed_grid = _deform_grid(grid, offset)

    # Reshape mask to match sampling points
    mask = mask.transpose(0, 2, 3, 1)

    return deformed_grid, mask


def _compute_indices_weights(
    points: NDArray[Any], size: Tuple[int, int]
) -> Tuple[NDArray[Any], ...]:
    """
    Computes indices and weights for bilinear interpolation.

    Args:
        points: Sampling points (N, H, W, 2)
        size: Size of the input feature map (H, W)

    Returns:
        Tuple of indices and weights for bilinear interpolation
    """
    H, W = size

    # Convert points to pixel coordinates
    x = (points[..., 0] + 1) * (W - 1) / 2
    y = (points[..., 1] + 1) * (H - 1) / 2

    # Get corner indices
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    # Clip to image boundaries
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # Compute weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (x0, x1, y0, y1), (wa, wb, wc, wd)


def _apply_deform_conv(
    input: NDArray[Any],
    weight: NDArray[Any],
    offset: NDArray[Any],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    mask: Optional[NDArray[Any]] = None,
) -> NDArray[Any]:
    """
    Applies deformable convolution operation.

    Args:
        input: Input feature map (N, C_in, H, W)
        weight: Convolution weights (C_out, C_in, kH, kW)
        offset: Sampling offsets (N, 2*kH*kW, H_out, W_out)
        stride: Convolution stride
        padding: Zero-padding size
        dilation: Dilation rate
        mask: Optional modulation mask for v2 (N, kH*kW, H_out, W_out)

    Returns:
        Output feature map (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = input.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    # Generate sampling grid
    grid = _generate_grid(N, H_out, W_out)

    # Deform grid using offsets
    if mask is not None:
        deformed_grid, modulation = _modulated_deform_grid(grid, offset, mask)
    else:
        deformed_grid = _deform_grid(grid, offset)
        modulation = None

    # Get sampling indices and weights
    indices, weights = _compute_indices_weights(deformed_grid, (H, W))
    x0, x1, y0, y1 = indices
    wa, wb, wc, wd = weights

    # Initialize output
    output = np.zeros((N, C_out, H_out, W_out))

    # Apply convolution with deformed sampling
    for i in range(kH):
        for j in range(kW):
            # Get values from input feature map
            values = (
                wa[..., None] * input[:, :, y0, x0]
                + wb[..., None] * input[:, :, y1, x0]
                + wc[..., None] * input[:, :, y0, x1]
                + wd[..., None] * input[:, :, y1, x1]
            )

            # Apply modulation if available
            if modulation is not None:
                values = values * modulation[:, i * kW + j, ..., None]

            # Accumulate weighted values
            for cout in range(C_out):
                output[:, cout] += np.sum(values * weight[cout, :, i, j], axis=1)

    return output


class Conv2dFunction(Function):
    @staticmethod
    def forward(
        ctx: Context,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        mode: str = ConvMode.STANDARD,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        output_padding: Tuple[int, int] = (0, 0),
    ) -> Tensor:
        """Forward pass of flexible 2D convolution."""
        # Validate parameters
        _validate_conv_params(
            x.shape, weight.shape, stride, padding, dilation, groups, mode, offset, weight, mask
        )

        # Save tensors and info for backward pass
        if mode == ConvMode.DEFORMABLE:
            ctx.save_for_backward(x, weight, bias, offset)
        else:
            ctx.save_for_backward(x, weight, bias)

        ctx.save_arguments(
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            mode=mode,
            sampling_locations=None,
            output_padding=output_padding,
        )

        if mode == ConvMode.TRANSPOSED:
            # Get shapes
            batch_size, C_in, H_in, W_in = x.shape
            C_out, C_in_per_group, kH, kW = weight.shape
            C_in_per_group = C_in // groups
            C_out_per_group = C_out // groups

            # Calculate output dimensions
            H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kH + output_padding[0]
            W_out = (W_in - 1) * stride[1] - 2 * padding[1] + kW + output_padding[1]

            # Initialize output tensor
            output = np.zeros((batch_size, C_out, H_out, W_out))

            # Process each group
            for g in range(groups):
                # Get input and weight for current group
                x_g = x.data[:, g * C_in_per_group : (g + 1) * C_in_per_group]
                w_g = weight.data[g * C_out_per_group : (g + 1) * C_out_per_group]

                # Flip kernel for transposed convolution
                w_g = np.flip(np.flip(w_g, 2), 3).transpose(1, 0, 2, 3)

                # Create the output tensor for this group
                output_g = np.zeros((batch_size, C_out_per_group, H_out, W_out))

                # Process each input element
                for n in range(batch_size):
                    for h in range(H_in):
                        for w in range(W_in):
                            # Get input value for all channels
                            x_val = x_g[n, :, h, w]  # Shape: (C_in_per_group,)

                            # Calculate output position
                            h_start = h * stride[0] - padding[0]
                            w_start = w * stride[1] - padding[1]

                            # Apply kernel at this position
                            for c_out in range(C_out_per_group):
                                for c_in in range(C_in_per_group):
                                    for kh in range(kH):
                                        for kw in range(kW):
                                            h_out = h_start + kh
                                            w_out = w_start + kw

                                            if 0 <= h_out < H_out and 0 <= w_out < W_out:
                                                output_g[n, c_out, h_out, w_out] += (
                                                    x_val[c_in] * w_g[c_in, c_out, kh, kw]
                                                )

                # Add this group's output to the final output
                output[:, g * C_out_per_group : (g + 1) * C_out_per_group] = output_g

            if bias is not None:
                output += bias.data.reshape(1, -1, 1, 1)

            return Tensor(output)
        else:
            # Rest of the code for standard/deformable convolution remains the same
            x_padded = _pad_input(x.data, padding)
            C_in_per_group = x.shape[1] // groups
            C_out_per_group = weight.shape[0] // groups

            H_out = (
                x.shape[2] + 2 * padding[0] - dilation[0] * (weight.shape[2] - 1) - 1
            ) // stride[0] + 1
            W_out = (
                x.shape[3] + 2 * padding[1] - dilation[1] * (weight.shape[3] - 1) - 1
            ) // stride[1] + 1

            output = np.zeros((x.shape[0], weight.shape[0], H_out, W_out))

            for g in range(groups):
                x_g = x_padded[:, g * C_in_per_group : (g + 1) * C_in_per_group]
                w_g = weight.data[g * C_out_per_group : (g + 1) * C_out_per_group]

                x_cols = _im2col_dilated(
                    x_g,
                    weight.shape[2:],
                    stride,
                    dilation,
                    padding,
                    mode=mode,
                    sampling_locations=None,
                )
                w_reshaped = w_g.reshape(C_out_per_group, -1)
                out = w_reshaped @ x_cols

                out = out.reshape(C_out_per_group, H_out * W_out, x.shape[0])
                out = out.transpose(2, 0, 1).reshape(x.shape[0], C_out_per_group, H_out, W_out)
                output[:, g * C_out_per_group : (g + 1) * C_out_per_group] = out

            if bias is not None:
                output += bias.data.reshape(1, -1, 1, 1)

            return Tensor(output)

    @staticmethod
    def backward(
        ctx: Context, grad_output: NDArray[Any], grad_dict: Dict[int, NDArray[Any]]
    ) -> None:
        """Backward pass of 2D convolution."""
        # Retrieve saved tensors and arguments
        saved_tensors = ctx.saved_tensors
        num_saved = len(saved_tensors)

        # Initialize variables
        offset_tensor = None
        mask = None
        grad_offset = None
        grad_mask = None

        # Get saved tensors based on mode
        if num_saved == 4:  # Deformable case
            x, weight, bias, offset_tensor = saved_tensors
        else:  # Standard/transposed case
            x, weight, bias = saved_tensors

        # Get saved arguments
        stride = ctx.saved_arguments["stride"]
        padding = ctx.saved_arguments["padding"]
        dilation = ctx.saved_arguments["dilation"]
        groups = ctx.saved_arguments["groups"]
        mode = ctx.saved_arguments["mode"]
        sampling_locations = ctx.saved_arguments.get("sampling_locations", None)

        # Calculate dimensions
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        C_in_per_group = C_in // groups
        C_out_per_group = C_out // groups
        H_out, W_out = _get_output_shape(x.shape, weight.shape[2:], stride, padding, dilation, mode)

        # Initialize gradients based on requires_grad flags
        grad_x = None
        grad_x_padded = None
        grad_weight = None
        grad_bias = None

        if x.requires_grad:
            if mode in [ConvMode.STANDARD, ConvMode.DEFORMABLE]:
                x_padded = _pad_input(x.data, padding)
                grad_x_padded = np.zeros_like(x_padded)
            else:  # Transposed
                grad_x = np.zeros_like(x.data)

        if weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)

        if bias is not None and bias.requires_grad:
            grad_bias = np.zeros_like(bias.data)

        if mode == ConvMode.DEFORMABLE:
            if offset_tensor is not None and offset_tensor.requires_grad:
                grad_offset = np.zeros_like(offset_tensor.data)

        # Compute gradients based on mode
        if mode == ConvMode.STANDARD:
            Conv2dFunction._backward_standard(
                grad_output,
                grad_x_padded,
                grad_weight,
                grad_bias,
                x_padded,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                N,
                C_in_per_group,
                C_out_per_group,
                kH,
                kW,
                H_out,
                W_out,
            )
        elif mode == ConvMode.TRANSPOSED:
            Conv2dFunction._backward_transposed(
                grad_output,
                grad_x,
                grad_weight,
                grad_bias,
                x,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                N,
                C_in_per_group,
                C_out_per_group,
                kH,
                kW,
                H_out,
                W_out,
            )
        elif mode == ConvMode.DEFORMABLE:
            Conv2dFunction._backward_deformable(
                grad_output,
                grad_x_padded,
                grad_weight,
                grad_bias,
                grad_offset,
                grad_mask,
                x_padded,
                weight,
                bias,
                offset_tensor,
                mask,
                stride,
                padding,
                dilation,
                groups,
                sampling_locations,
                N,
                C_in_per_group,
                C_out_per_group,
                kH,
                kW,
                H_out,
                W_out,
            )

        # Assign gradients to grad_dict
        if x.requires_grad:
            if mode in [ConvMode.STANDARD, ConvMode.DEFORMABLE]:
                if padding[0] > 0 or padding[1] > 0:
                    if grad_x_padded is not None:
                        grad_x = grad_x_padded[
                            :,
                            :,
                            padding[0] : grad_x_padded.shape[2] - padding[0],
                            padding[1] : grad_x_padded.shape[3] - padding[1],
                        ]
                else:
                    grad_x = grad_x_padded
            grad_dict[id(x)] = grad_x

        if weight.requires_grad:
            grad_dict[id(weight)] = grad_weight

        if bias is not None and bias.requires_grad:
            grad_dict[id(bias)] = grad_bias

        if offset_tensor is not None and offset_tensor.requires_grad:
            grad_dict[id(offset_tensor)] = grad_offset

    @staticmethod
    def _backward_standard(
        grad_output: NDArray[Any],
        grad_x_padded: Optional[NDArray[Any]],
        grad_weight: Optional[NDArray[Any]],
        grad_bias: Optional[NDArray[Any]],
        x_padded: NDArray[Any],
        weight: Tensor,
        bias: Optional[Tensor],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        N: int,
        C_in_per_group: int,
        C_out_per_group: int,
        kH: int,
        kW: int,
        H_out: int,
        W_out: int,
    ) -> None:
        """Backward pass for standard convolution."""
        for g in range(groups):
            # Get weight and grad_output for current group
            w_g = weight.data[g * C_out_per_group : (g + 1) * C_out_per_group]
            grad_out_g = grad_output[:, g * C_out_per_group : (g + 1) * C_out_per_group]

            # Convert grad_output to columns
            grad_out_col = grad_out_g.transpose(1, 0, 2, 3).reshape(C_out_per_group, -1)

            # Get input columns
            x_cols = _im2col_dilated(
                x_padded[:, g * C_in_per_group : (g + 1) * C_in_per_group],
                (kH, kW),
                stride,
                dilation,
                padding,
                ConvMode.STANDARD,
            )

            # Compute weight gradients
            if grad_weight is not None:
                grad_weight[g * C_out_per_group : (g + 1) * C_out_per_group] = (
                    grad_out_col @ x_cols.T
                ).reshape(C_out_per_group, C_in_per_group, kH, kW)

            # Compute input gradients
            w_reshaped = w_g.reshape(C_out_per_group, -1).T
            grad_cols = w_reshaped @ grad_out_col

            # Reshape grad_cols to match the expected shape
            grad_cols = grad_cols.reshape(-1, N * H_out * W_out)

            # Convert columns back to image format
            if grad_x_padded is not None:
                grad_x_padded[:, g * C_in_per_group : (g + 1) * C_in_per_group] += _col2im_dilated(
                    grad_cols, x_padded.shape, (kH, kW), stride, dilation
                )

        # Compute bias gradients if needed
        if bias is not None and grad_bias is not None:
            grad_bias[:] = grad_output.sum(axis=(0, 2, 3))

    @staticmethod
    def _backward_transposed(
        grad_output_padded: NDArray[Any],
        grad_x: Optional[NDArray[Any]],
        grad_weight: Optional[NDArray[Any]],
        grad_bias: Optional[NDArray[Any]],
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        N: int,
        C_in_per_group: int,
        C_out_per_group: int,
        kH: int,
        kW: int,
        H_out: int,
        W_out: int,
    ) -> None:
        """
        Backward pass for transposed convolution.
        """
        for g in range(groups):
            # Slice weights and grad_output for the current group
            w_g = weight.data[g * C_out_per_group : (g + 1) * C_out_per_group]
            grad_out_g = grad_output_padded[:, g * C_out_per_group : (g + 1) * C_out_per_group]

            # Reshape weights for gradient computation
            w_flipped = np.flip(np.flip(w_g, 2), 3).transpose(1, 0, 2, 3)  # Flip kernel
            w_reshaped = w_flipped.reshape(
                -1, C_out_per_group
            )  # Shape: (C_in_per_group * kH * kW, C_out_per_group)

            # Compute gradient columns with swapped stride and dilation
            grad_cols = _im2col_dilated(
                grad_out_g,
                weight.shape[2:],
                dilation,  # Use dilation as stride
                stride,  # Use stride as dilation
                (0, 0),  # Change to tuple for padding parameter
            )

            # Compute gradient for input
            if grad_x is not None:
                grad_x[:, g * C_in_per_group : (g + 1) * C_in_per_group] += (
                    w_reshaped @ grad_cols
                ).reshape(N, C_in_per_group, *x.shape[2:])

            # Compute gradient for weights
            x_original = x.data[:, g * C_in_per_group : (g + 1) * C_in_per_group]
            x_cols = _im2col_dilated(
                x_original,
                weight.shape[2:],
                stride,
                dilation,
                (0, 0),
                sampling_locations=None,
            )
            for n in range(N):
                grad_out_n = grad_out_g[n].reshape(
                    C_out_per_group, -1
                )  # Shape: (C_out_per_group, H_out * W_out)
                if grad_weight is not None:
                    grad_weight[g * C_out_per_group : (g + 1) * C_out_per_group] += (
                        grad_out_n @ x_cols[:, n * H_out * W_out : (n + 1) * H_out * W_out].T
                    ).reshape(C_out_per_group, C_in_per_group, kH, kW)

        # Compute gradient for bias
        if bias is not None:
            grad_bias += grad_output_padded.sum(axis=(0, 2, 3))

    @staticmethod
    def _backward_deformable(
        grad_output: NDArray[Any],
        grad_x_padded: Optional[NDArray[Any]],
        grad_weight: Optional[NDArray[Any]],
        grad_bias: Optional[NDArray[Any]],
        grad_offset: Optional[NDArray[Any]],
        grad_mask: Optional[NDArray[Any]],
        x_padded: NDArray[Any],
        weight: Tensor,
        bias: Optional[Tensor],
        offset_tensor: Optional[Tensor],
        mask: Optional[NDArray[Any]],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        sampling_locations: Optional[NDArray[Any]],
        N: int,
        C_in_per_group: int,
        C_out_per_group: int,
        kH: int,
        kW: int,
        H_out: int,
        W_out: int,
        align_corners: bool = True,
    ) -> None:
        """Backward pass for deformable convolution."""

        if sampling_locations is None and offset_tensor is not None:
            # If sampling_locations weren't provided but we have offset tensor, compute them
            sampling_locations = _get_deformable_offsets(
                offset_tensor.data, (kH, kW), x_padded.shape, dilation
            )

        # Now proceed with backward pass using sampling_locations
        for g in range(groups):
            w_g = weight.data[g * C_out_per_group : (g + 1) * C_out_per_group]
            grad_out_g = grad_output[:, g * C_out_per_group : (g + 1) * C_out_per_group]

            for n in range(N):
                for h in range(H_out):
                    for w_ in range(W_out):
                        i = h * W_out + w_
                        grad_out_slice = grad_out_g[n, :, h, w_]

                        for c_out in range(C_out_per_group):
                            grad = grad_out_slice[c_out]

                            for c_in in range(C_in_per_group):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        k_idx = kh * kW + kw

                                        if sampling_locations is not None:
                                            loc = sampling_locations[n, i, k_idx]
                                            h_in = int(loc[0])
                                            w_in = int(loc[1])

                                            if (
                                                0 <= h_in < x_padded.shape[2]
                                                and 0 <= w_in < x_padded.shape[3]
                                            ):
                                                # Update gradients
                                                if grad_x_padded is not None:
                                                    grad_x_padded[
                                                        n, g * C_in_per_group + c_in, h_in, w_in
                                                    ] += (grad * w_g[c_out, c_in, kh, kw])
                                                if grad_weight is not None:
                                                    grad_weight[
                                                        g * C_out_per_group + c_out, c_in, kh, kw
                                                    ] += (
                                                        grad
                                                        * x_padded[
                                                            n, g * C_in_per_group + c_in, h_in, w_in
                                                        ]
                                                    )

        # Compute gradients for bias if needed
        if grad_bias is not None and bias is not None:
            grad_bias[:] = grad_output.sum((0, 2, 3))
