from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from ..core import Function, Tensor

class ConvMode:
    """Enumeration of convolution modes."""
    STANDARD = "standard"
    TRANSPOSED = "transposed"
    DEFORMABLE = "deformable"

def _validate_conv_params(x_shape: Tuple[int, ...], weight_shape: Tuple[int, ...],
                       stride: Tuple[int, int], padding: Tuple[int, int],
                       dilation: Tuple[int, int], groups: int,
                       weight: Optional[Tensor] = None,  # Add weight tensor parameter
                       mode: str = ConvMode.STANDARD) -> None:
    """
    Validates convolution parameters.
    
    Args:
        x_shape: Input shape (N, C_in, H, W)
        weight_shape: Weight shape (C_out, C_in/groups, kH, kW)
        stride: Stride in height and width directions
        padding: Padding in height and width directions
        dilation: Dilation in height and width directions
        groups: Number of groups for grouped convolution
        mode: Convolution mode (standard, transposed, or deformable)
    """
    N, C_in, H, W = x_shape
    C_out, C_in_per_group, kH, kW = weight_shape

    if C_in % groups != 0:
            raise ValueError(f"Input channels ({C_in}) must be divisible by groups ({groups})")
            
    if C_out % groups != 0:
        raise ValueError(f"Output channels ({C_out}) must be divisible by groups ({groups})")
        
    if C_in_per_group != C_in // groups:
        raise ValueError(f"Expected {C_in // groups} input channels per group, got {C_in_per_group}")
        
    if mode not in [ConvMode.STANDARD, ConvMode.TRANSPOSED, ConvMode.DEFORMABLE]:
        raise ValueError(f"Invalid convolution mode: {mode}")
        
    if mode == ConvMode.DEFORMABLE and (weight is None or not hasattr(weight, 'offset')):
        raise ValueError("Deformable convolution requires offset parameters")

    if mode == ConvMode.DEFORMABLE and offset is None:
        raise ValueError("Deformable convolution requires offset parameter")

def _pad_input(x: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    """
    Pads input tensor with zeros.
    
    Args:
        x: Input tensor
        padding: (padding_height, padding_width)
    """
    if padding[0] == 0 and padding[1] == 0:
        return x
    pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
    return np.pad(x, pad_width, mode='constant', constant_values=0)

def _get_output_shape(input_shape: Tuple[int, ...], kernel_size: Tuple[int, int],
                    stride: Tuple[int, int], padding: Tuple[int, int],
                    dilation: Tuple[int, int], mode: str = ConvMode.STANDARD) -> Tuple[int, int]:
    """
    Calculates output shape for different convolution types.
    """
    if mode == ConvMode.STANDARD:
        H = ((input_shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) 
             // stride[0] + 1)
        W = ((input_shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) 
             // stride[1] + 1)
    elif mode == ConvMode.TRANSPOSED:
        H = (input_shape[2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1
        W = (input_shape[3] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1
    else:  # Deformable follows standard conv shape
        H = ((input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1)
        W = ((input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1)
    return H, W

def _get_deformable_offsets(offset_tensor: Tensor, kernel_size: Tuple[int, int],
                         input_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Computes sampling locations for deformable convolution.
    """
    N, C, H, W = input_shape
    kH, kW = kernel_size
    
    # Generate base sampling grid
    base_h = np.arange(kH)
    base_w = np.arange(kW)
    mesh_h, mesh_w = np.meshgrid(base_h, base_w)
    base_grid = np.stack([mesh_h, mesh_w], axis=-1)
    
    # Add offsets to base grid
    offsets = offset_tensor.data.reshape(N, 2 * kH * kW, H, W)
    sampling_locations = base_grid.reshape(-1, 2) + offsets
    
    return sampling_locations

def _bilinear_interpolate(input_tensor: np.ndarray, sampling_locations: np.ndarray) -> np.ndarray:
    """
    Performs bilinear interpolation for deformable convolution.
    """
    N, C, H, W = input_tensor.shape
    
    # Clip sampling locations to valid range
    h = sampling_locations[:, :, :, 0].clip(0, H - 1)
    w = sampling_locations[:, :, :, 1].clip(0, W - 1)
    
    # Get corner points
    h0 = np.floor(h).astype(np.int32)
    w0 = np.floor(w).astype(np.int32)
    h1 = h0 + 1
    w1 = w0 + 1
    
    # Clip again
    h0 = h0.clip(0, H - 1)
    w0 = w0.clip(0, W - 1)
    h1 = h1.clip(0, H - 1)
    w1 = w1.clip(0, W - 1)
    
    # Get weights for bilinear interpolation
    wa = (w1 - w) * (h1 - h)
    wb = (w1 - w) * (h - h0)
    wc = (w - w0) * (h1 - h)
    wd = (w - w0) * (h - h0)
    
    # Gather and weight values
    output = (input_tensor[:, :, h0, w0] * wa +
             input_tensor[:, :, h0, w1] * wb +
             input_tensor[:, :, h1, w0] * wc +
             input_tensor[:, :, h1, w1] * wd)
             
    return output

def _im2col_dilated(x: np.ndarray, kernel_size: Tuple[int, int],
                    stride: Tuple[int, int], dilation: Tuple[int, int],
                    mode: str = ConvMode.STANDARD,
                    sampling_locations: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rearranges dilated image blocks into columns with support for different convolution types.
    """
    N, C, H, W = x.shape
    kH, kW = kernel_size
    dH, dW = dilation
    
    # Calculate output size based on mode
    out_h, out_w = _get_output_shape((N, C, H, W), kernel_size, stride, (0, 0), dilation, mode)
    
    # Initialize output array
    cols = np.zeros((C * kH * kW, N * out_h * out_w))
    
    if mode == ConvMode.DEFORMABLE and sampling_locations is not None:
        # Handle deformable convolution
        for n in range(N):
            for h_out in range(out_h):
                for w_out in range(out_w):
                    # Get sampling locations for this output position
                    loc = sampling_locations[n, :, h_out, w_out]
                    
                    # Sample using bilinear interpolation
                    sampled_values = _bilinear_interpolate(
                        x[n:n+1], 
                        loc.reshape(1, -1, 1, 2)
                    )
                    
                    # Store in cols array
                    col_idx = np.arange(C * kH * kW)
                    row_idx = n * out_h * out_w + h_out * out_w + w_out
                    cols[col_idx, row_idx] = sampled_values.flatten()
    else:
        # Standard or transposed convolution
        for h_out in range(out_h):
            for w_out in range(out_w):
                for c in range(C):
                    for i in range(kH):
                        for j in range(kW):
                            if mode == ConvMode.STANDARD:
                                h_in = h_out * stride[0] + i * dH
                                w_in = w_out * stride[1] + j * dW
                            else:  # TRANSPOSED
                                h_in = h_out * dH + i * stride[0]
                                w_in = w_out * dW + j * stride[1]
                            
                            col_idx = (c * kH * kW + i * kW + j)
                            row_idx = h_out * out_w + w_out
                            
                            for n in range(N):
                                if 0 <= h_in < H and 0 <= w_in < W:
                                    cols[col_idx, n * out_h * out_w + row_idx] = x[n, c, h_in, w_in]
                                    
    return cols

def _compute_conv_output_shape(input_size: int, kernel_size: int, stride: int,
                             padding: int, dilation: int) -> int:
    """Computes output dimension for a single axis."""
    numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1

def _compute_conv_grad_input_padding(grad_output_size: int, input_size: int,
                                   kernel_size: int, stride: int, padding: int,
                                   dilation: int) -> Tuple[int, int]:
    """Computes padding needed for gradient computation."""
    grad_input_padding = kernel_size - 1 - padding
    return grad_input_padding

def _compute_output_padding(input_size: int, output_size: int, kernel_size: int,
                          stride: int, padding: int, dilation: int) -> int:
    """Computes additional padding needed for transposed convolution."""
    expected_output = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    return output_size - expected_output

def _unfold(input_tensor: np.ndarray,
           kernel_size: Tuple[int, ...],
           dilation: Tuple[int, ...],
           padding: Tuple[int, ...],
           stride: Tuple[int, ...]) -> np.ndarray:
    """Extracts sliding local blocks from input tensor."""
    N, C, H, W = input_tensor.shape
    kH, kW = kernel_size
    
    # Apply padding if needed
    if padding[0] > 0 or padding[1] > 0:
        input_tensor = np.pad(input_tensor,
                          ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                          mode='constant')
    
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
                    patch = input_tensor[:, :, h_start:h_start+1, w_start:w_start+1]
                    
                    # Place in output array
                    row_idx = (i * kW + j) * C + np.arange(C)
                    col_idx = h * W_out + w + np.arange(N) * H_out * W_out
                    output[row_idx[:, None], col_idx] = patch.reshape(N, C).T
    
    return output

def _fold(input: np.ndarray,
         output_size: Tuple[int, ...],
         kernel_size: Tuple[int, ...],
         dilation: Tuple[int, ...],
         padding: Tuple[int, ...],
         stride: Tuple[int, ...]) -> np.ndarray:
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
                    output[:, :, h_start:h_start+1, w_start:w_start+1] += patch
                    divisor[:, :, h_start:h_start+1, w_start:w_start+1] += 1
    
    # Average overlapping values
    output = np.divide(output, divisor, where=divisor != 0)
    
    # Remove padding if necessary
    if padding[0] > 0 or padding[1] > 0:
        output = output[:, :, padding[0]:-padding[0] if padding[0] > 0 else None,
                       padding[1]:-padding[1] if padding[1] > 0 else None]
    
    return output

def _dilate(input: np.ndarray, dilation: Tuple[int, ...]) -> np.ndarray:
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

def _bilinear_interpolate(input: np.ndarray,
                       points: np.ndarray,
                       align_corners: bool = True) -> np.ndarray:
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

def _bilinear_interpolate_gradient(grad_output: np.ndarray,
                                points: np.ndarray,
                                input_size: Tuple[int, ...],
                                align_corners: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes gradients for bilinear interpolation.
    
    Args:
        grad_output: Gradient of the loss with respect to interpolated values (N, C, P)
        points: Points that were sampled (N, P, 2)
        input_size: Size of the input tensor (H, W)
        align_corners: Whether corners were aligned in interpolation
        
    Returns:
        Tuple of gradients with respect to input and points
    """
    N, C, P = grad_output.shape
    H, W = input_size

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

    # Reshape for broadcasting
    wa = wa.reshape(N, 1, P)
    wb = wb.reshape(N, 1, P)
    wc = wc.reshape(N, 1, P)
    wd = wd.reshape(N, 1, P)

    # Initialize gradients
    grad_input = np.zeros((N, C, H, W))
    grad_points = np.zeros_like(points)

    # Accumulate gradients for input
    np.add.at(grad_input, (slice(None), slice(None), y0, x0), grad_output * wa)
    np.add.at(grad_input, (slice(None), slice(None), y1, x0), grad_output * wb)
    np.add.at(grad_input, (slice(None), slice(None), y0, x1), grad_output * wc)
    np.add.at(grad_input, (slice(None), slice(None), y1, x1), grad_output * wd)

    # Compute gradients for sampling points
    if align_corners:
        dx = (W - 1) / 2
        dy = (H - 1) / 2
    else:
        dx = W / 2
        dy = H / 2

    grad_x = grad_output * ((y1 - y).reshape(N, 1, P) * \
                         (input[:, :, y0, x1] - input[:, :, y0, x0]) + \
                         (y - y0).reshape(N, 1, P) * \
                         (input[:, :, y1, x1] - input[:, :, y1, x0]))
    grad_y = grad_output * ((x1 - x).reshape(N, 1, P) * \
                         (input[:, :, y1, x0] - input[:, :, y0, x0]) + \
                         (x - x0).reshape(N, 1, P) * \
                         (input[:, :, y1, x1] - input[:, :, y0, x1]))

    grad_points[..., 0] = np.sum(grad_x, axis=1) * dx
    grad_points[..., 1] = np.sum(grad_y, axis=1) * dy

    return grad_input, grad_points

def _generate_grid(batch_size: int, height: int, width: int,
                 align_corners: bool = True) -> np.ndarray:
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
        x = np.linspace(-1 + (1/width), 1 - (1/width), width)
        y = np.linspace(-1 + (1/height), 1 - (1/height), height)

    x_coords, y_coords = np.meshgrid(x, y)
    grid = np.stack([x_coords, y_coords], axis=-1)
    grid = np.tile(grid[None], (batch_size, 1, 1, 1))
    
    return grid

def _deform_grid(grid: np.ndarray, offset: np.ndarray) -> np.ndarray:
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

def _modulated_deform_grid(grid: np.ndarray, offset: np.ndarray, 
                        mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def _compute_indices_weights(points: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, ...]:
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

def _apply_deform_conv(input: np.ndarray, weight: np.ndarray, offset: np.ndarray,
                    stride: Tuple[int, int], padding: Tuple[int, int],
                    dilation: Tuple[int, int], mask: Optional[np.ndarray] = None) -> np.ndarray:
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
            values = (wa[..., None] * input[:, :, y0, x0] +
                     wb[..., None] * input[:, :, y1, x0] +
                     wc[..., None] * input[:, :, y0, x1] +
                     wd[..., None] * input[:, :, y1, x1])
                     
            # Apply modulation if available
            if modulation is not None:
                values = values * modulation[:, i*kW + j, ..., None]
                
            # Accumulate weighted values
            for cout in range(C_out):
                output[:, cout] += np.sum(values * weight[cout, :, i, j], axis=1)
    
    return output

class Conv2dFunction(Function):
    """
    Implements 2D convolution with support for:
    - Asymmetric kernels
    - Different strides for height and width
    - Different padding for height and width
    - Different dilation rates for height and width
    - Transposed convolution
    - Deformable convolution
    - Grouped convolution
    """
    
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            mode: str = ConvMode.STANDARD,
            offset: Optional[Tensor] = None,
            mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of flexible 2D convolution.
        """
        # Convert scalar params to tuples
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        
        # Validate parameters
        _validate_conv_params(x.shape, weight.shape, stride, padding, dilation, groups, mode, offset, mask)

        # Save tensors and parameters
        ctx.save_for_backward(x, weight, bias, offset)
        ctx.save_arguments(stride=stride, padding=padding, dilation=dilation,
                        groups=groups, mode=mode)
        
        # Save tensors and parameters for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.save_arguments(stride=stride, padding=padding, dilation=dilation, 
                         groups=groups, mode=mode)
        
        x_padded = _pad_input(x.data, padding)
        N = x.shape[0]
        C_out = weight.shape[0]
        
        # Get output dimensions
        out_h, out_w = _get_output_shape(x.shape, weight.shape[2:], stride, padding, dilation, mode)
        
        # Handle grouped convolution
        C_in_per_group = x.shape[1] // groups
        C_out_per_group = C_out // groups
        
        # Initialize output
        output = np.zeros((N, C_out, out_h, out_w))
        
        # Get sampling locations for deformable convolution
        sampling_locations = None
        if mode == ConvMode.DEFORMABLE:
            sampling_locations = _get_deformable_offsets(offset.data, weight.shape[2:], x.shape)
        
        # Process each group separately
        for g in range(groups):
            x_g = x_padded[:, g*C_in_per_group:(g+1)*C_in_per_group]
            w_g = weight.data[g*C_out_per_group:(g+1)*C_out_per_group]
            
            # Reshape weight for matrix multiplication
            w_reshaped = w_g.reshape(C_out_per_group, -1)
            
            # Convert input patches to columns
            x_cols = _im2col_dilated(x_g, weight.shape[2:], stride, dilation, 
                                   mode, sampling_locations)
            
            # Perform convolution as matrix multiplication
            out = w_reshaped @ x_cols
            
            # Reshape output and handle batch dimension
            out = out.reshape(C_out_per_group, -1)
            for n in range(N):
                start_idx = n * out_h * out_w
                end_idx = (n + 1) * out_h * out_w
                output[n, g*C_out_per_group:(g+1)*C_out_per_group] = \
                    out[:, start_idx:end_idx].reshape(C_out_per_group, out_h, out_w)
        
        # Add bias if present
        if bias is not None:
            output += bias.data.reshape(1, -1, 1, 1)
            
        return Tensor(output)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        """
        Backward pass of 2D convolution.
        
        Computes gradients for:
        - Input tensor (x)
        - Weight tensor (w)
        - Bias (if present)
        - Offsets (for deformable convolution)
        
        Handles different convolution modes:
        - Standard convolution
        - Transposed convolution
        - Deformable convolution
        """
        x, weight, bias = ctx.saved_tensors
        stride = ctx.saved_arguments['stride']
        padding = ctx.saved_arguments['padding']
        dilation = ctx.saved_arguments['dilation']
        groups = ctx.saved_arguments['groups']
        mode = ctx.saved_arguments['mode']

        # Get dimensions
        N = x.shape[0]
        C_in_per_group = x.shape[1] // groups
        C_out_per_group = weight.shape[0] // groups
        out_h, out_w = _get_output_shape(x.shape, weight.shape[2:], stride, padding, dilation, mode)

        # Pad input for standard convolution
        if mode == ConvMode.STANDARD:
            x_padded = _pad_input(x.data, padding)
        elif mode == ConvMode.TRANSPOSED:
            # For transposed convolution, we pad the gradient instead
            grad_output_padded = _pad_input(grad_output, padding)
        else:  # DEFORMABLE
            x_padded = _pad_input(x.data, padding)
            sampling_locations = _get_deformable_offsets(weight.offset, weight.shape[2:], x.shape)

        # Compute input gradient if required
        if x.requires_grad:
            if mode == ConvMode.STANDARD:
                grad_x_padded = np.zeros_like(x_padded)

                # Process each group
                for g in range(groups):
                    w_g = weight.data[g*C_out_per_group:(g+1)*C_out_per_group]
                    grad_out_g = grad_output[:, g*C_out_per_group:(g+1)*C_out_per_group]

                    # Reshape weight and gradient
                    w_reshaped = w_g.reshape(C_out_per_group, -1).T
                    grad_out_reshaped = grad_out_g.reshape(N, -1).T

                    # Compute gradient using transposed operations
                    grad_cols = w_reshaped @ grad_out_reshaped.reshape(C_out_per_group, -1)
                    grad_x_g = _col2im_dilated(
                        grad_cols,
                        x_padded[:, g*C_in_per_group:(g+1)*C_in_per_group].shape,
                        weight.shape[2:],
                        stride,
                        dilation
                    )
                    grad_x_padded[:, g*C_in_per_group:(g+1)*C_in_per_group] = grad_x_g

                # Remove padding from gradient
                if padding[0] > 0 or padding[1] > 0:
                    grad_dict[id(x)] = grad_x_padded[:, :,
                                                   padding[0]:grad_x_padded.shape[2]-padding[0],
                                                   padding[1]:grad_x_padded.shape[3]-padding[1]]
                else:
                    grad_dict[id(x)] = grad_x_padded

            elif mode == ConvMode.TRANSPOSED:
                grad_x = np.zeros_like(x.data)

                # Process each group
                for g in range(groups):
                    w_g = weight.data[g*C_out_per_group:(g+1)*C_out_per_group]
                    grad_out_g = grad_output_padded[:, g*C_out_per_group:(g+1)*C_out_per_group]

                    # For transposed conv, we swap stride and dilation
                    grad_cols = _im2col_dilated(
                        grad_out_g,
                        weight.shape[2:],
                        dilation,  # Use dilation as stride
                        stride,    # Use stride as dilation
                        ConvMode.STANDARD
                    )

                    # Compute gradient using flipped weights
                    w_flipped = np.flip(np.flip(w_g, 2), 3).transpose(1, 0, 2, 3)
                    w_reshaped = w_flipped.reshape(-1, C_out_per_group)
                    grad_x[:, g*C_in_per_group:(g+1)*C_in_per_group] = \
                        (w_reshaped @ grad_cols).reshape(N, C_in_per_group, *x.shape[2:])

                grad_dict[id(x)] = grad_x

            else:  # DEFORMABLE
                grad_x = np.zeros_like(x.data)

                # Process each group
                for g in range(groups):
                    w_g = weight.data[g*C_out_per_group:(g+1)*C_out_per_group]
                    grad_out_g = grad_output[:, g*C_out_per_group:(g+1)*C_out_per_group]

                    # Compute gradient for regular convolution first
                    grad_standard = np.zeros_like(x_padded)
                    w_reshaped = w_g.reshape(C_out_per_group, -1).T
                    grad_out_reshaped = grad_out_g.reshape(N, -1).T
                    grad_cols = w_reshaped @ grad_out_reshaped.reshape(C_out_per_group, -1)

                    # Apply deformable sampling gradient
                    for n in range(N):
                        for h in range(out_h):
                            for w in range(out_w):
                                loc = sampling_locations[n, :, h, w]
                                grad_standard[n, :, h, w] = _bilinear_interpolate_gradient(
                                    grad_cols[:, n*out_h*out_w + h*out_w + w],
                                    loc,
                                    x.shape[2:]
                                )

                    grad_x[:, g*C_in_per_group:(g+1)*C_in_per_group] = grad_standard[:, 
                        g*C_in_per_group:(g+1)*C_in_per_group]

                grad_dict[id(x)] = grad_x

        # Compute weight gradient if required
        if weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)

            # Process each group
            for g in range(groups):
                if mode == ConvMode.STANDARD:
                    x_g = x_padded[:, g*C_in_per_group:(g+1)*C_in_per_group]
                    grad_out_g = grad_output[:, g*C_out_per_group:(g+1)*C_out_per_group]

                    # Compute gradient using im2col
                    x_cols = _im2col_dilated(x_g, weight.shape[2:], stride, dilation, mode)
                    grad_out_reshaped = grad_out_g.reshape(N, C_out_per_group, -1)

                    # Accumulate gradients for each batch
                    for n in range(N):
                        grad_w = grad_out_reshaped[n] @ x_cols[:, n*out_h*out_w:(n+1)*out_h*out_w].T
                        grad_weight[g*C_out_per_group:(g+1)*C_out_per_group] += \
                            grad_w.reshape(C_out_per_group, C_in_per_group, *weight.shape[2:])

                elif mode == ConvMode.TRANSPOSED:
                    x_g = x[:, g*C_in_per_group:(g+1)*C_in_per_group]
                    grad_out_g = grad_output_padded[:, g*C_out_per_group:(g+1)*C_out_per_group]

                    # For transposed conv, we compute gradient using rotated operations
                    x_cols = _im2col_dilated(
                        x_g.transpose(1, 0, 2, 3),
                        weight.shape[2:],
                        dilation,
                        stride,
                        ConvMode.STANDARD
                    )
                    grad_out_reshaped = grad_out_g.transpose(1, 0, 2, 3).reshape(C_out_per_group, -1)
                    grad_weight[g*C_out_per_group:(g+1)*C_out_per_group] = \
                        (grad_out_reshaped @ x_cols.T).reshape(C_out_per_group, C_in_per_group, 
                                                             *weight.shape[2:])

                else:  # DEFORMABLE
                    x_g = x_padded[:, g*C_in_per_group:(g+1)*C_in_per_group]
                    grad_out_g = grad_output[:, g*C_out_per_group:(g+1)*C_out_per_group]

                    # Similar to standard conv but with deformable sampling
                    for n in range(N):
                        deform_cols = _im2col_dilated(
                            x_g[n:n+1],
                            weight.shape[2:],
                            stride,
                            dilation,
                            mode,
                            sampling_locations[n:n+1]
                        )
                        grad_out_n = grad_out_g[n].reshape(C_out_per_group, -1)
                        grad_weight[g*C_out_per_group:(g+1)*C_out_per_group] += \
                            (grad_out_n @ deform_cols.T).reshape(C_out_per_group, C_in_per_group, 
                                                               *weight.shape[2:])

            grad_dict[id(weight)] = grad_weight

            # Compute offset gradients for deformable convolution
            if mode == ConvMode.DEFORMABLE and hasattr(weight, 'offset') and weight.offset.requires_grad:
                grad_offset = _compute_offset_gradients(
                    grad_output, x_padded, weight, sampling_locations,
                    stride, padding, dilation, groups
                )
                grad_dict[id(weight.offset)] = grad_offset

        # Compute bias gradient if required
        if bias is not None and bias.requires_grad:
            grad_dict[id(bias)] = grad_output.sum(axis=(0, 2, 3))