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
    _deform_grid,
    _get_deformable_offsets, 
    _compute_conv_grad_input_padding,
    _col2im_dilated
)

class TestConv2d:
    """Tests for Conv2d module."""
    
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

    def test_col2im_dilated(self):
        """Test column to image conversion with dilation"""
        batch_size = 2
        in_channels = 3
        height = width = 8
        kernel_size = 3
        
        # Create sample columns
        cols = np.random.randn(in_channels * kernel_size * kernel_size, 
                            batch_size * height * width)
        
        # Test with different dilation rates
        dilations = [(1, 1), (2, 2), (1, 2)]
        for dilation in dilations:
            output = _col2im_dilated(cols, 
                                    (batch_size, in_channels, height, width),
                                    (kernel_size, kernel_size),
                                    (1, 1),  # stride
                                    dilation)
            
            assert output.shape == (batch_size, in_channels, height, width)

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

    def test_conv2d_output_shape_calculation(self):
        """Test the calc_output_shape static method"""
        input_shape = (1, 3, 32, 32)
        out_channels = 16
        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)
        dilation = (1, 1)
        
        output_shape = Conv2d.calc_output_shape(
            input_shape, out_channels, kernel_size, stride, padding, dilation
        )
        assert output_shape == (1, 16, 16, 16)

    def test_conv2d_invalid_groups(self):
        """Test error handling for invalid group configurations"""
        with pytest.raises(ValueError):
            # in_channels not divisible by groups
            Conv2d(5, 10, kernel_size=3, groups=2)
            
        with pytest.raises(ValueError):
            # out_channels not divisible by groups
            Conv2d(4, 5, kernel_size=3, groups=2)

    def test_conv2d_extra_repr(self):
        """Test string representation of Conv2d layer"""
        conv = Conv2d(3, 16, kernel_size=3, stride=2, padding=1, dilation=2, groups=1)
        repr_str = conv.extra_repr()
        assert '3, 16' in repr_str
        assert 'kernel_size=(3, 3)' in repr_str
        assert 'stride=(2, 2)' in repr_str

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

    def test_bilinear_interpolate(self):
        """Test bilinear interpolation function"""
        input_tensor = np.random.randn(1, 3, 4, 4)  # Change batch size to 1
        points = np.array([[[-1.0, -1.0]]])  # Use normalized coordinates in [-1, 1] range
        
        output = _bilinear_interpolate(input_tensor, points)
        assert output.shape == (1, 3, 1)  # (batch_size, channels, num_points)
        
    def test_compute_conv_output_shape(self):
        """Test output shape computation for different configurations"""
        input_size = 32
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1
        
        output_size = _compute_conv_output_shape(
            input_size, kernel_size, stride, padding, dilation
        )
        expected = ((32 + 2*1 - 1*(3-1) - 1) // 2) + 1
        assert output_size == expected

    def test_unfold_fold_operations(self):
        """Test unfold and fold operations are inverses"""
        input_tensor = np.random.randn(2, 3, 8, 8)
        kernel_size = (3, 3)
        stride = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        
        unfolded = _unfold(input_tensor, kernel_size, dilation, padding, stride)
        folded = _fold(unfolded, (8, 8), kernel_size, dilation, padding, stride)
        
        assert np.allclose(input_tensor, folded, atol=1e-6)
    
    def test_backward_standard_comprehensive(self):
        """Test standard convolution backward pass with various configurations"""
        batch_size = 2
        in_channels = 4
        out_channels = 8
        height = width = 8
        kernel_size = 3
        
        # Test different stride/padding combinations
        configs = [
            ((1, 1), (0, 0)),  # Basic case
            ((2, 2), (1, 1)),  # With stride and padding
            ((1, 2), (2, 1)),  # Asymmetric stride and padding
        ]
        
        for stride, padding in configs:
            # Set requires_grad=True for tensors that need gradients
            x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
            weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
            bias = Tensor(np.random.randn(out_channels), requires_grad=True)
            
            # Forward pass
            output = Conv2dFunction.apply(x, weight, bias, stride, padding)
            
            # Backward pass
            grad_output = np.random.randn(*output.shape)
            output.backward(grad_output)
            
            # Verify gradients exist and have correct shapes
            assert x.grad is not None
            assert weight.grad is not None
            assert bias.grad is not None
            
            # Verify gradient shapes
            assert x.grad.shape == x.shape
            assert weight.grad.shape == weight.shape
            assert bias.grad.shape == bias.shape

    def test_deformable_conv_with_mask(self):
        """Test deformable convolution with modulation mask"""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 8
        kernel_size = 3
        
        x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        offset = Tensor(np.random.randn(batch_size, 2*kernel_size*kernel_size, height-2, width-2), requires_grad=True)
        mask = Tensor(np.random.randn(batch_size, kernel_size*kernel_size, height-2, width-2), requires_grad=True)
        bias = Tensor(np.random.randn(out_channels), requires_grad=True)  # Add bias
        
        output = Conv2dFunction.apply(x, weight, bias, (1, 1), (0, 0), (1, 1), 1,
                                    ConvMode.DEFORMABLE, offset, mask)
        
        grad_output = np.random.randn(*output.shape)
        output.backward(grad_output)
    
    def test_transposed_conv_output_padding(self):
        """Test transposed convolution with output padding"""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height = width = 8
        kernel_size = 3
        stride = (2, 2)
        padding = (1, 1)

        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        # For transposed conv, weight shape should be (in_channels, out_channels/groups, kernel_size, kernel_size)
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))  # Not swapped, keep original format

        # Forward pass
        H_out = (height - 1) * stride[0] - 2 * padding[0] + kernel_size
        W_out = (width - 1) * stride[1] - 2 * padding[1] + kernel_size
        
        # Test with different output paddings
        output_paddings = [(0, 0), (1, 1), (1, 0)]
        for output_padding in output_paddings:
            # Add output padding to expected dimensions
            expected_H = H_out + output_padding[0]
            expected_W = W_out + output_padding[1]
            
            output = Conv2dFunction.apply(x, weight, None, stride, padding, (1, 1), 1,
                                    ConvMode.TRANSPOSED, output_padding=output_padding)
            
            assert output.shape == (batch_size, out_channels, expected_H, expected_W)
        
    def test_conv2d_edge_cases(self):
        """Test convolution with edge cases"""
        # Test 1x1 convolution
        x = Tensor(np.random.randn(2, 3, 8, 8))
        weight = Tensor(np.random.randn(4, 3, 1, 1))
        output = Conv2dFunction.apply(x, weight)
        assert output.shape == (2, 4, 8, 8)
        
        # Test with large kernel
        x = Tensor(np.random.randn(2, 3, 16, 16))
        weight = Tensor(np.random.randn(4, 3, 7, 7))
        output = Conv2dFunction.apply(x, weight)
        assert output.shape == (2, 4, 10, 10)
    
    def test_conv2d_validation_errors(self):
        """Test proper error handling"""
        with pytest.raises(ValueError):
            # Test invalid group configuration
            x = Tensor(np.random.randn(2, 3, 8, 8))
            weight = Tensor(np.random.randn(4, 2, 3, 3))  # Wrong in_channels per group
            Conv2dFunction.apply(x, weight, groups=2)

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

    def test_bilinear_interpolate(self):
        """Test bilinear interpolation function"""
        input_tensor = np.random.randn(1, 3, 4, 4)  # Changed batch size to 1
        points = np.array([[[0.5, 0.5]]])  # Shape should be (1, 1, 2)
        
        output = _bilinear_interpolate(input_tensor, points)
        assert output.shape == (1, 3, 1)  # (N, C, P)

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
    
    def test_get_deformable_offsets(self):
        """Test deformable convolution offset computation"""
        batch_size, in_channels = 2, 3
        height, width = 8, 8
        kernel_size = (3, 3)
        
        # Create sample offset tensor
        offset_tensor = np.random.randn(batch_size, 2*kernel_size[0]*kernel_size[1], height-2, width-2)
        input_shape = (batch_size, in_channels, height, width)
        
        sampling_locations = _get_deformable_offsets(offset_tensor, kernel_size, input_shape)
        assert sampling_locations.shape == (batch_size, (height-2)*(width-2), kernel_size[0]*kernel_size[1], 2)

    def test_compute_conv_grad_input_padding(self):
        """Test gradient input padding computation"""
        grad_output_size = 8
        input_size = 10
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        
        grad_padding = _compute_conv_grad_input_padding(
            grad_output_size, input_size, kernel_size, stride, padding, dilation
        )
        assert isinstance(grad_padding, int)

    def test_generate_grid(self):
        """Test sampling grid generation for different align_corners settings"""
        batch_size = 2
        height = 4
        width = 4
        
        # Test with align_corners=True
        grid_aligned = _generate_grid(batch_size, height, width, align_corners=True)
        assert grid_aligned.shape == (batch_size, height, width, 2)
        # Check grid bounds
        assert np.all(grid_aligned >= -1) and np.all(grid_aligned <= 1)
        
        # Test with align_corners=False
        grid_unaligned = _generate_grid(batch_size, height, width, align_corners=False)
        assert grid_unaligned.shape == (batch_size, height, width, 2)
        assert np.all(grid_unaligned >= -1) and np.all(grid_unaligned <= 1)