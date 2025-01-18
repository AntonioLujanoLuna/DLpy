import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn import Conv2d
from DLpy.ops.cnn import (
    Conv2dFunction, ConvMode,
    _compute_conv_output_shape,
    _unfold,
    _fold,
    _bilinear_interpolate,
    _generate_grid,
    _deform_grid
)

class TestConv2d:
    """Tests for Conv2d module."""
    
    # [Previous tests remain the same...]

    def test_conv2d_asymmetric_kernel(self):
        """Test Conv2d with asymmetric kernel."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 32
        kernel_size = (3, 5)  # Asymmetric kernel
        
        conv = Conv2d(in_channels, out_channels, kernel_size)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        output = conv(x)
        
        # Check output shape
        expected_height = height - kernel_size[0] + 1
        expected_width = width - kernel_size[1] + 1
        assert output.shape == (batch_size, out_channels, expected_height, expected_width)

    def test_conv2d_asymmetric_stride(self):
        """Test Conv2d with different strides for height and width."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 32
        kernel_size = 3
        stride = (2, 3)  # Different strides
        
        conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        output = conv(x)
        
        expected_height = (height - kernel_size) // stride[0] + 1
        expected_width = (width - kernel_size) // stride[1] + 1
        assert output.shape == (batch_size, out_channels, expected_height, expected_width)

    def test_conv2d_asymmetric_padding(self):
        """Test Conv2d with different padding for height and width."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 32
        kernel_size = 3
        padding = (1, 2)  # Different padding
        
        conv = Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        output = conv(x)
        
        expected_height = height + 2*padding[0] - kernel_size + 1
        expected_width = width + 2*padding[1] - kernel_size + 1
        assert output.shape == (batch_size, out_channels, expected_height, expected_width)

    def test_conv2d_dilated(self):
        """Test dilated convolution."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 32
        kernel_size = 3
        dilation = 2
        
        conv = Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        output = conv(x)
        
        effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        expected_height = height - effective_kernel_size + 1
        expected_width = width - effective_kernel_size + 1
        assert output.shape == (batch_size, out_channels, expected_height, expected_width)

    def test_conv2d_groups(self):
        """Test grouped convolution."""
        batch_size = 2
        in_channels = 4
        out_channels = 4
        height = width = 32
        kernel_size = 3
        groups = 2
        
        conv = Conv2d(in_channels, out_channels, kernel_size, groups=groups)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        output = conv(x)
        
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1
        assert output.shape == (batch_size, out_channels, expected_height, expected_width)

class TestConv2dFunction:
    """Tests for Conv2dFunction and related helper functions."""
    
    def test_deformable_conv_forward(self):
        """Test forward pass of deformable convolution."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 8
        kernel_size = 3
        
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        offset = Tensor(np.random.randn(batch_size, 2*kernel_size*kernel_size, height-2, width-2))
        bias = Tensor(np.random.randn(out_channels))
        
        output = Conv2dFunction.apply(x, weight, bias, (1, 1), (0, 0), (1, 1), 1,
                                    ConvMode.DEFORMABLE, offset)

    def test_modulated_deform_conv(self):
        """Test modulated deformable convolution."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 8
        kernel_size = 3
        
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        offset = Tensor(np.random.randn(batch_size, 2*kernel_size*kernel_size, height-2, width-2))
        mask = Tensor(np.random.randn(batch_size, kernel_size*kernel_size, height-2, width-2))
        bias = Tensor(np.random.randn(out_channels))
        
        output = Conv2dFunction.apply(x, weight, bias, (1, 1), (0, 0), (1, 1), 1,
                                    ConvMode.DEFORMABLE, offset, mask)

class TestHelperFunctions:
    """Tests for CNN helper functions."""
    
    def test_compute_output_shape(self):
        """Test output shape computation."""
        input_size = 32
        kernel_size = 3
        stride = 1
        padding = 0
        dilation = 1
        
        output_size = _compute_conv_output_shape(
            input_size, kernel_size, stride, padding, dilation
        )
        assert output_size == 30  # 32 - 3 + 1

    def test_unfold_operation(self):
        """Test im2col (unfold) operation."""
        batch_size = 2
        channels = 3
        height = width = 8
        kernel_size = (3, 3)
        
        input_tensor = np.random.randn(batch_size, channels, height, width)
        unfolded = _unfold(input_tensor, kernel_size, (1, 1), (0, 0), (1, 1))
        
        # Check unfolded shape
        expected_unfold_shape = (channels * kernel_size[0] * kernel_size[1],
                               batch_size * (height - kernel_size[0] + 1) * 
                               (width - kernel_size[1] + 1))
        assert unfolded.shape == expected_unfold_shape

    def test_fold_operation(self):
        """Test col2im (fold) operation."""
        batch_size = 2
        channels = 3
        height = width = 8
        kernel_size = (3, 3)
        
        # Create random input and unfold it
        input_tensor = np.random.randn(batch_size, channels, height, width)
        unfolded = _unfold(input_tensor, kernel_size, (1, 1), (0, 0), (1, 1))
        
        # Fold back
        folded = _fold(unfolded, (height, width), kernel_size, (1, 1), (0, 0), (1, 1))
        
        # Check folded shape
        assert folded.shape == input_tensor.shape

    def test_bilinear_interpolation(self):
        """Test bilinear interpolation."""
        batch_size = 2
        channels = 3
        height = width = 8
        
        input_tensor = np.random.randn(batch_size, channels, height, width)
        points = np.random.uniform(-1, 1, (batch_size, 4, 2))  # Sample 4 points
        
        interpolated = _bilinear_interpolate(input_tensor, points)
        assert interpolated.shape == (batch_size, channels, 4)

    def test_generate_grid(self):
        """Test sampling grid generation."""
        batch_size = 2
        height = 8
        width = 8
        
        grid = _generate_grid(batch_size, height, width)
        assert grid.shape == (batch_size, height, width, 2)
        assert np.all(grid >= -1) and np.all(grid <= 1)

    def test_deform_grid(self):
        """Test grid deformation."""
        batch_size = 2
        height = 8
        width = 8
        
        grid = _generate_grid(batch_size, height, width)
        offset = np.random.randn(batch_size, 2, height, width) * 0.1
        
        deformed = _deform_grid(grid, offset)
        assert deformed.shape == (batch_size, height, width, 2)
        assert np.all(deformed >= -1) and np.all(deformed <= 1)

    def test_numerical_gradient_deformable(self):
        """Test numerical gradient computation for deformable convolution."""
        batch_size = 2
        in_channels = 2
        out_channels = 3
        height = width = 5
        kernel_size = 3
        
        x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), 
                       requires_grad=True)
        offset = Tensor(np.random.randn(batch_size, 2*kernel_size*kernel_size, height-2, width-2), 
                       requires_grad=True)
        setattr(weight, 'offset', offset)
        bias = Tensor(np.random.randn(out_channels), requires_grad=True)
        
        def compute_loss(x, w, b):
            return np.sum(Conv2dFunction.apply(x, w, b, (1, 1), (0, 0), (1, 1), 1, 
                                             ConvMode.DEFORMABLE).data)
        
        # Compute analytical gradients
        output = Conv2dFunction.apply(x, weight, bias, (1, 1), (0, 0), (1, 1), 1, 
                                    ConvMode.DEFORMABLE)
        output.backward(np.ones_like(output.data))
        
        # Verify offset gradients exist and have correct shape
        assert offset.grad is not None
        assert offset.grad.shape == offset.shape