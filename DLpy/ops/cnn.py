from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from ..core import Function, Tensor, Context

class ConvMode:
    """Enumeration of convolution modes."""
    STANDARD = "standard"
    TRANSPOSED = "transposed"
    DEFORMABLE = "deformable"

def _validate_conv_params(
    x_shape: tuple,
    weight_shape: tuple,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
    mode: str = ConvMode.STANDARD,
    offset: Optional[Tensor] = None,
    weight: Optional[Tensor] = None, # Added weight parameter
    mask: Optional[Tensor] = None
) -> None:
    """Validates convolution parameters."""
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
        
    if mode == ConvMode.DEFORMABLE:
        # Check for offset either in direct parameter or weight.offset
        offset_tensor = offset
        if offset_tensor is None and weight is not None:
            offset_tensor = getattr(weight, 'offset', None)
            
        if offset_tensor is None:
            raise ValueError("Deformable convolution requires offset parameter")
        
        # Calculate output size
        H_out = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
        
        # Check offset tensor shape
        expected_offset_shape = (N, 2 * kH * kW, H_out, W_out)
        if offset_tensor.shape != expected_offset_shape:
            raise ValueError(f"Expected offset shape {expected_offset_shape}, got {offset_tensor.shape}")
            
        # Check mask tensor shape if provided
        if mask is not None:
            expected_mask_shape = (N, kH * kW, H_out, W_out)
            if mask.shape != expected_mask_shape:
                raise ValueError(f"Expected mask shape {expected_mask_shape}, got {mask.shape}")


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

def _get_deformable_offsets(offset_tensor: np.ndarray, kernel_size: Tuple[int, int],
                         input_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Computes sampling locations for deformable convolution.
    
    Args:
        offset_tensor: Offset values of shape (N, 2*kH*kW, H_out, W_out)
        kernel_size: Size of the convolving kernel (kH, kW)
        input_shape: Shape of input tensor (N, C, H, W)
        
    Returns:
        Sampling locations of shape (N, H_out*W_out, kH*kW, 2)
    """
    N, C, H, W = input_shape
    kH, kW = kernel_size
    H_out, W_out = offset_tensor.shape[2:]
    
    # Convert memoryview to numpy array if needed
    if isinstance(offset_tensor, memoryview):
        offset_tensor = np.array(offset_tensor)
    
    # Generate base sampling grid for each output position
    base_h = np.arange(kH)
    base_w = np.arange(kW)
    mesh_h, mesh_w = np.meshgrid(base_h, base_w, indexing='ij')
    base_grid = np.stack([mesh_h, mesh_w], axis=-1)  # (kH, kW, 2)
    
    # Reshape for broadcasting
    kHW = kH * kW
    base_grid = base_grid.reshape(-1, 2).T  # (2, kH*kW)
    base_grid = base_grid[None, None, :, :]  # (1, 1, 2, kH*kW)
    
    # Reshape offset tensor to (N, H_out*W_out, 2, kH*kW)
    offset_tensor = offset_tensor.reshape(N, 2, kHW, H_out, W_out)
    offset_tensor = offset_tensor.transpose(0, 3, 4, 1, 2)  # (N, H_out, W_out, 2, kH*kW)
    offset_tensor = offset_tensor.reshape(N, H_out * W_out, 2, kHW)
    
    # Broadcast base grid to match offset tensor shape
    base_grid = np.broadcast_to(base_grid, (N, H_out * W_out, 2, kHW))
    
    # Add offsets to base grid
    sampling_locations = base_grid + offset_tensor
    
    # Ensure output is in correct format (N, H_out*W_out, kH*kW, 2)
    return sampling_locations.transpose(0, 1, 3, 2)

def _bilinear_interpolate(input: np.ndarray, points: np.ndarray, align_corners: bool = True) -> np.ndarray:
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
    
    # Ensure points is correct shape (N, P, 2)
    if points.ndim == 4:
        points = points.reshape(points.shape[0], -1, 2)
    
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
    
    # Reshape weights for broadcasting
    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]
    
    # Gather corner values and compute weighted sum
    output = np.zeros((N, C, P))
    for n in range(N):
        output[n] = (wa[n] * input[n, :, y0[n], x0[n]] +
                    wb[n] * input[n, :, y1[n], x0[n]] +
                    wc[n] * input[n, :, y0[n], x1[n]] +
                    wd[n] * input[n, :, y1[n], x1[n]])
    
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
        for n in range(N):
            for h_out in range(out_h):
                for w_out in range(out_w):
                    # Reshape to (1, kH*kW, 2) for bilinear interpolation
                    loc = sampling_locations[n, h_out*out_w + w_out].reshape(1, -1, 2)
                    
                    # Sample using bilinear interpolation
                    sampled_values = _bilinear_interpolate(
                        x[n:n+1],
                        loc
                    )
                    
                    # Store in cols array
                    idx = n * out_h * out_w + h_out * out_w + w_out
                    cols[:, idx] = sampled_values.reshape(-1)
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
                                input_tensor: np.ndarray,
                                align_corners: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(f"Gradient shape {grad_output.shape} cannot be broadcast to number of points {P}")
        
    C = grad_output.shape[1]
    H, W = input_size

    # Validate input_tensor shape
    if input_tensor.shape[2:] != input_size:
        raise ValueError(f"Input tensor spatial dimensions {input_tensor.shape[2:]} "
                        f"don't match input_size {input_size}")
    
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
            gx = grad * (
                (y1[n, p] - y[n, p]) * (
                    input_tensor[n, :, y0[n, p], x1[n, p]] - 
                    input_tensor[n, :, y0[n, p], x0[n, p]]
                ).sum() +
                (y[n, p] - y0[n, p]) * (
                    input_tensor[n, :, y1[n, p], x1[n, p]] - 
                    input_tensor[n, :, y1[n, p], x0[n, p]]
                ).sum()
            ) * dx
            
            # Gradient with respect to y
            gy = grad * (
                (x1[n, p] - x[n, p]) * (
                    input_tensor[n, :, y1[n, p], x0[n, p]] - 
                    input_tensor[n, :, y0[n, p], x0[n, p]]
                ).sum() +
                (x[n, p] - x0[n, p]) * (
                    input_tensor[n, :, y1[n, p], x1[n, p]] - 
                    input_tensor[n, :, y0[n, p], x1[n, p]]
                ).sum()
            ) * dy
            
            grad_points[n, p] = [gx, gy]
    
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
    def forward(ctx: Context, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
            stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
            dilation: Tuple[int, int] = (1, 1), groups: int = 1,
            mode: str = ConvMode.STANDARD, offset: Optional[Tensor] = None,
            mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of flexible 2D convolution."""
        # Validate parameters
        _validate_conv_params(x.shape, weight.shape, stride, padding, dilation, groups, 
                            mode, offset, weight, mask)
        
        # Get required offset tensor and compute sampling locations for deformable conv
        if mode == ConvMode.DEFORMABLE:
            offset_tensor = offset if offset is not None else getattr(weight, 'offset', None)
            sampling_locations = _get_deformable_offsets(
                offset_tensor.data,
                weight.shape[2:],
                x.shape
            )
            # Ensure sampling_locations has correct shape for _im2col_dilated
            N, HW, kHW, _ = sampling_locations.shape
            H_out = W_out = int(np.sqrt(HW))  # Assuming square output
            sampling_locations = sampling_locations.reshape(N, H_out * W_out, kHW, 2)
        else:
            sampling_locations = None
            offset_tensor = None  # Explicitly set to None if not deformable
        
        # Save tensors and info for backward pass
        if mode == ConvMode.DEFORMABLE:
            ctx.save_for_backward(x, weight, bias, offset_tensor)
        else:
            ctx.save_for_backward(x, weight, bias)
            
        ctx.save_arguments(stride=stride, padding=padding, dilation=dilation, 
                        groups=groups, mode=mode, sampling_locations=sampling_locations)
        
        # Process each group
        x_padded = _pad_input(x.data, padding)
        C_in_per_group = x.shape[1] // groups
        C_out_per_group = weight.shape[0] // groups
        H_out, W_out = _get_output_shape(x.shape, weight.shape[2:], stride, padding, dilation, mode)
        output = np.zeros((x.shape[0], weight.shape[0], H_out, W_out))
        
        for g in range(groups):
            x_g = x_padded[:, g*C_in_per_group:(g+1)*C_in_per_group]
            w_g = weight.data[g*C_out_per_group:(g+1)*C_out_per_group]
            
            # Convert input patches to columns
            x_cols = _im2col_dilated(x_g, weight.shape[2:], stride, dilation, mode, 
                                sampling_locations)
            
            # Reshape weight for matrix multiplication
            w_reshaped = w_g.reshape(C_out_per_group, -1)
            
            # Perform convolution
            out = w_reshaped @ x_cols
            
            # Reshape output
            output[:, g*C_out_per_group:(g+1)*C_out_per_group] = out.reshape(
                C_out_per_group, x.shape[0], *output.shape[2:]
            ).transpose(1, 0, 2, 3)
        
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
        # Retrieve saved tensors and arguments
        saved_tensors = ctx.saved_tensors
        num_saved = len(saved_tensors)

        # Initialize mask_tensor to None
        mask_tensor = None

        if num_saved == 4:  # Deformable case without mask
            x, weight, bias, offset_tensor = saved_tensors
        elif num_saved == 5:  # Deformable case with mask (if supported)
            x, weight, bias, offset_tensor, mask_tensor = saved_tensors
        else:  # Standard or other modes
            x, weight, bias = saved_tensors
            offset_tensor = None

        # Retrieve saved arguments
        stride = ctx.saved_arguments['stride']
        padding = ctx.saved_arguments['padding']
        dilation = ctx.saved_arguments['dilation']
        groups = ctx.saved_arguments['groups']
        mode = ctx.saved_arguments['mode']
        sampling_locations = ctx.saved_arguments.get('sampling_locations', None)

        # Retrieve mask if present
        mask = ctx.saved_arguments.get('mask', None)
        if mask_tensor is not None:
            mask = mask_tensor.data

        # Determine dimensions
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        C_in_per_group = C_in // groups
        C_out_per_group = C_out // groups
        H_out, W_out = _get_output_shape(x.shape, weight.shape[2:], stride, padding, dilation, mode)

        # Apply padding based on convolution mode
        if mode in [ConvMode.STANDARD, ConvMode.DEFORMABLE]:
            x_padded = _pad_input(x.data, padding)
        elif mode == ConvMode.TRANSPOSED:
            grad_output_padded = _pad_input(grad_output, padding)

        # Initialize gradients if required
        if x.requires_grad:
            if mode in [ConvMode.STANDARD, ConvMode.DEFORMABLE]:
                grad_x_padded = np.zeros_like(x_padded)
            elif mode == ConvMode.TRANSPOSED:
                grad_x = np.zeros_like(x.data)
        if weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)
        if bias is not None and bias.requires_grad:
            grad_bias = np.zeros_like(bias.data)
        if mode == ConvMode.DEFORMABLE and offset_tensor is not None and offset_tensor.requires_grad:
            grad_offset = np.zeros_like(offset_tensor.data)
        if mask is not None and mode == ConvMode.DEFORMABLE and mask_tensor is not None and mask_tensor.requires_grad:
            grad_mask = np.zeros_like(mask_tensor.data)
        else:
            grad_mask = None  # Initialize grad_mask to None if mask is not provided

        # Delegate backward computation based on mode
        if mode == ConvMode.STANDARD:
            Conv2dFunction._backward_standard(
                grad_output, grad_x_padded, grad_weight, grad_bias,
                x_padded, weight, bias, stride, padding, dilation, groups,
                N, C_in_per_group, C_out_per_group, kH, kW, H_out, W_out
            )
        elif mode == ConvMode.TRANSPOSED:
            Conv2dFunction._backward_transposed(
                grad_output_padded, grad_x, grad_weight, grad_bias,
                x, weight, bias, stride, padding, dilation, groups,
                N, C_in_per_group, C_out_per_group, kH, kW, H_out, W_out
            )
        elif mode == ConvMode.DEFORMABLE:
            Conv2dFunction._backward_deformable(
                grad_output, grad_x_padded, grad_weight, grad_bias, grad_offset, grad_mask,
                x_padded, weight, bias, offset_tensor, mask, stride, padding, dilation, groups, sampling_locations,
                N, C_in_per_group, C_out_per_group, kH, kW, H_out, W_out
            )

        # Assign gradients to grad_dict
        if x.requires_grad:
            if mode in [ConvMode.STANDARD, ConvMode.DEFORMABLE]:
                if mode == ConvMode.STANDARD:
                    # Remove padding from grad_x_padded if necessary
                    if padding[0] > 0 or padding[1] > 0:
                        grad_x = grad_x_padded[:, :,
                                            padding[0]:grad_x_padded.shape[2]-padding[0],
                                            padding[1]:grad_x_padded.shape[3]-padding[1]]
                    else:
                        grad_x = grad_x_padded
                elif mode == ConvMode.DEFORMABLE:
                    # Directly assign grad_x_padded to grad_x for deformable convolution
                    grad_x = grad_x_padded
                grad_dict[id(x)] = grad_x
            elif mode == ConvMode.TRANSPOSED:
                grad_dict[id(x)] = grad_x

        if weight.requires_grad:
            grad_dict[id(weight)] = grad_weight

        if bias is not None and bias.requires_grad:
            grad_dict[id(bias)] = grad_bias

        if mode == ConvMode.DEFORMABLE and offset_tensor is not None and offset_tensor.requires_grad:
            grad_dict[id(offset_tensor)] = grad_offset

        if mask is not None and mode == ConvMode.DEFORMABLE and mask_tensor is not None and mask_tensor.requires_grad:
            grad_dict[id(mask_tensor)] = grad_mask

    @staticmethod
    def _backward_standard(
        grad_output, grad_x_padded, grad_weight, grad_bias,
        x_padded, weight, bias, stride, padding, dilation, groups,
        N, C_in_per_group, C_out_per_group, kH, kW, H_out, W_out
    ):
        """
        Backward pass for standard convolution.
        """
        for g in range(groups):
            # Slice weights and grad_output for the current group
            w_g = weight.data[g * C_out_per_group:(g + 1) * C_out_per_group]
            grad_out_g = grad_output[:, g * C_out_per_group:(g + 1) * C_out_per_group]

            # Reshape weights and grad_output for matrix multiplication
            w_reshaped = w_g.reshape(C_out_per_group, -1).T  # Shape: (C_in_per_group * kH * kW, C_out_per_group)
            grad_out_reshaped = grad_out_g.reshape(N, C_out_per_group, -1).transpose(1, 0, 2).reshape(C_out_per_group, -1)  # Shape: (C_out_per_group, N * H_out * W_out)

            # Compute gradient columns
            grad_cols = w_reshaped @ grad_out_reshaped  # Shape: (C_in_per_group * kH * kW, N * H_out * W_out)

            # Convert columns back to image
            grad_x_g = _col2im_dilated(
                grad_cols,
                x_padded[:, g * C_in_per_group:(g + 1) * C_in_per_group].shape,
                weight.shape[2:],
                stride,
                dilation
            )
            grad_x_padded[:, g * C_in_per_group:(g + 1) * C_in_per_group] += grad_x_g

            # Compute gradient for weights
            x_cols = _im2col_dilated(
                x_padded[:, g * C_in_per_group:(g + 1) * C_in_per_group],
                weight.shape[2:], stride, dilation, ConvMode.STANDARD, sampling_locations=None
            )
            for n in range(N):
                grad_out_n = grad_out_g[n].reshape(C_out_per_group, -1)  # Shape: (C_out_per_group, H_out * W_out)
                grad_weight[g * C_out_per_group:(g + 1) * C_out_per_group] += \
                    (grad_out_n @ x_cols[:, n * H_out * W_out:(n + 1) * H_out * W_out].T).reshape(
                        C_out_per_group, C_in_per_group, kH, kW
                    )

        # Compute gradient for bias
        if bias is not None:
            grad_bias += grad_output.sum(axis=(0, 2, 3))

    @staticmethod
    def _backward_transposed(
        grad_output_padded, grad_x, grad_weight, grad_bias,
        x, weight, bias, stride, padding, dilation, groups,
        N, C_in_per_group, C_out_per_group, kH, kW, H_out, W_out
    ):
        """
        Backward pass for transposed convolution.
        """
        for g in range(groups):
            # Slice weights and grad_output for the current group
            w_g = weight.data[g * C_out_per_group:(g + 1) * C_out_per_group]
            grad_out_g = grad_output_padded[:, g * C_out_per_group:(g + 1) * C_out_per_group]

            # Reshape weights for gradient computation
            w_flipped = np.flip(np.flip(w_g, 2), 3).transpose(1, 0, 2, 3)  # Flip kernel
            w_reshaped = w_flipped.reshape(-1, C_out_per_group)  # Shape: (C_in_per_group * kH * kW, C_out_per_group)

            # Compute gradient columns with swapped stride and dilation
            grad_cols = _im2col_dilated(
                grad_out_g,
                weight.shape[2:],
                dilation,  # Use dilation as stride
                stride,    # Use stride as dilation
                ConvMode.STANDARD
            )

            # Compute gradient for input
            grad_x[:, g * C_in_per_group:(g + 1) * C_in_per_group] += \
                (w_reshaped @ grad_cols).reshape(N, C_in_per_group, *x.shape[2:])

            # Compute gradient for weights
            x_original = x.data[:, g * C_in_per_group:(g + 1) * C_in_per_group]
            x_cols = _im2col_dilated(
                x_original,
                weight.shape[2:], stride, dilation, ConvMode.STANDARD, sampling_locations=None
            )
            for n in range(N):
                grad_out_n = grad_out_g[n].reshape(C_out_per_group, -1)  # Shape: (C_out_per_group, H_out * W_out)
                grad_weight[g * C_out_per_group:(g + 1) * C_out_per_group] += \
                    (grad_out_n @ x_cols[:, n * H_out * W_out:(n + 1) * H_out * W_out].T).reshape(
                        C_out_per_group, C_in_per_group, kH, kW
                    )

        # Compute gradient for bias
        if bias is not None:
            grad_bias += grad_output_padded.sum(axis=(0, 2, 3))

    @staticmethod
    def _backward_deformable(
        grad_output, grad_x_padded, grad_weight, grad_bias, grad_offset, grad_mask,
        x_padded, weight, bias, offset_tensor, mask, stride, padding, dilation, groups, sampling_locations,
        N, C_in_per_group, C_out_per_group, kH, kW, H_out, W_out
    ):
        """
        Backward pass for deformable convolution.
        """
        for g in range(groups):
            # Slice weights and grad_output for the current group
            w_g = weight.data[g * C_out_per_group:(g + 1) * C_out_per_group]  # Shape: (C_out_per_group, C_in_per_group, kH, kW)
            grad_out_g = grad_output[:, g * C_out_per_group:(g + 1) * C_out_per_group]  # Shape: (N, C_out_per_group, H_out, W_out)

            for n in range(N):
                for h in range(H_out):
                    for w_ in range(W_out):
                        # Flat index for the current spatial location
                        i = h * W_out + w_

                        # Extract grad_out for current (n, g, h, w_)
                        grad_out_slice = grad_out_g[n, :, h, w_]  # Shape: (C_out_per_group,)

                        for c_out in range(C_out_per_group):
                            # Get the gradient for the current output channel
                            grad = grad_out_slice[c_out]  # Scalar

                            for c_in in range(C_in_per_group):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        # Index for the kernel position
                                        k_idx = kh * kW + kw

                                        # Get the corresponding sampling location
                                        loc = sampling_locations[n, i, k_idx, :]  # Shape: (2,)
                                        y, x = loc  # y and x are the sampling locations in input space

                                        # Compute integer coordinates
                                        y0 = int(np.floor(y))
                                        x0 = int(np.floor(x))
                                        y1 = y0 + 1
                                        x1 = x0 + 1

                                        # Compute fractional parts
                                        dy = y - y0
                                        dx = x - x0

                                        # Compute interpolation weights
                                        wa = (1 - dy) * (1 - dx)
                                        wb = dy * (1 - dx)
                                        wc = (1 - dy) * dx
                                        wd = dy * dx

                                        # Compute input slice with batch and channel dimensions
                                        input_slice = x_padded[n:n+1, g * C_in_per_group + c_in : g * C_in_per_group + c_in + 1, :, :]  # Shape: (1, 1, H, W)

                                        # Reshape loc to (N=1, P=1, 2)
                                        points = loc.reshape(1, 1, 2)  # Shape: (1, 1, 2)

                                        # Perform bilinear interpolation
                                        interpolated = _bilinear_interpolate(input_slice, points, align_corners=True)  # Shape: (1, 1, 1)
                                        interpolated_sum = interpolated.sum()  # Scalar

                                        # Update grad_weight
                                        grad_weight[g * C_out_per_group + c_out, c_in, kh, kw] += grad * interpolated_sum

                                        # Update grad_x_padded with interpolation weights
                                        # Ensure that the indices are within bounds
                                        if 0 <= y0 < x_padded.shape[2] and 0 <= x0 < x_padded.shape[3]:
                                            grad_x_padded[n, g * C_in_per_group + c_in, y0, x0] += grad * w_g[c_out, c_in, kh, kw] * wa
                                        if 0 <= y1 < x_padded.shape[2] and 0 <= x0 < x_padded.shape[3]:
                                            grad_x_padded[n, g * C_in_per_group + c_in, y1, x0] += grad * w_g[c_out, c_in, kh, kw] * wb
                                        if 0 <= y0 < x_padded.shape[2] and 0 <= x1 < x_padded.shape[3]:
                                            grad_x_padded[n, g * C_in_per_group + c_in, y0, x1] += grad * w_g[c_out, c_in, kh, kw] * wc
                                        if 0 <= y1 < x_padded.shape[2] and 0 <= x1 < x_padded.shape[3]:
                                            grad_x_padded[n, g * C_in_per_group + c_in, y1, x1] += grad * w_g[c_out, c_in, kh, kw] * wd

                                        # Compute gradients w.r.t. offset if applicable
                                        if offset_tensor.requires_grad:
                                            # Placeholder for actual gradient computation w.r.t. offset
                                            # This requires computing derivatives based on how offsets affect sampling locations
                                            # For simplicity, this implementation does not compute these gradients
                                            pass  # Replace with actual gradient computation for offsets

                                        # Compute gradients w.r.t. mask if applicable
                                        if mask is not None and grad_mask is not None:
                                            # Assuming mask modulates the convolution, compute its gradient
                                            # The gradient is grad_output * weight * sampled_input
                                            # First, perform bilinear interpolation to get sampled input
                                            sampled_input = _bilinear_interpolate(input_slice, points, align_corners=True).squeeze(0).squeeze(0)  # Shape: (1,)
                                            
                                            # Compute grad_mask
                                            grad_mask[g * kH * kW + kh * kW + kw, n, h, w_] += grad * w_g[c_out, c_in, kh, kw] * sampled_input

        # After accumulating all gradients, compute grad_bias if needed
        if bias is not None and bias.requires_grad:
            grad_bias += grad_output.sum(axis=(0, 2, 3))  # Sum over N, H_out, W_out