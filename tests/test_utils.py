import pytest
import numpy as np
from DLpy.utils import calculate_fan_in_fan_out

class TestFanInFanOut:
    """Test suite for the fan-in and fan-out calculation utility."""
    
    def test_linear_layer_dimensions(self):
        """Test fan-in and fan-out calculation for linear layer weights.
        
        For linear layers:
        - fan_in is number of input features (second dimension)
        - fan_out is number of output features (first dimension)
        """
        # Create a weight matrix for linear layer (out_features × in_features)
        weight = np.random.randn(100, 50)  # 100 output features, 50 input features
        
        fan_in, fan_out = calculate_fan_in_fan_out(weight)
        
        assert fan_in == 50, "Fan-in should match input features"
        assert fan_out == 100, "Fan-out should match output features"
        
    def test_conv2d_dimensions(self):
        """Test fan-in and fan-out calculation for 2D convolutional layer weights.
        
        For conv layers:
        - fan_in = channels_in * kernel_size
        - fan_out = channels_out * kernel_size
        """
        # Create a 2D conv weight tensor (out_channels × in_channels × kernel_height × kernel_width)
        weight = np.random.randn(64, 32, 3, 3)  
        # 64 output channels, 32 input channels, 3×3 kernel
        
        fan_in, fan_out = calculate_fan_in_fan_out(weight)
        
        expected_fan_in = 32 * (3 * 3)   # in_channels * kernel_size
        expected_fan_out = 64 * (3 * 3)  # out_channels * kernel_size
        
        assert fan_in == expected_fan_in, "Fan-in should account for input channels and kernel size"
        assert fan_out == expected_fan_out, "Fan-out should account for output channels and kernel size"
        
    def test_conv3d_dimensions(self):
        """Test fan-in and fan-out calculation for 3D convolutional layer weights.
        
        Similar to Conv2D, but with an additional dimension for the kernel depth.
        """
        # Create a 3D conv weight tensor 
        # (out_channels × in_channels × kernel_depth × kernel_height × kernel_width)
        weight = np.random.randn(32, 16, 2, 3, 3)
        
        fan_in, fan_out = calculate_fan_in_fan_out(weight)
        
        expected_fan_in = 16 * (2 * 3 * 3)   # in_channels * kernel_volume
        expected_fan_out = 32 * (2 * 3 * 3)  # out_channels * kernel_volume
        
        assert fan_in == expected_fan_in, "Fan-in should account for 3D kernel volume"
        assert fan_out == expected_fan_out, "Fan-out should account for 3D kernel volume"
        
    def test_1d_tensor_error(self):
        """Test that the function raises an error for 1D tensors."""
        with pytest.raises(ValueError) as exc_info:
            calculate_fan_in_fan_out(np.array([1, 2, 3]))
        
        assert "tensor.shape should have at least 2 dimensions" in str(exc_info.value)
            
    def test_scalar_tensor_error(self):
        """Test that the function raises an error for scalar tensors."""
        with pytest.raises(ValueError) as exc_info:
            calculate_fan_in_fan_out(np.array(5))
            
        assert "tensor.shape should have at least 2 dimensions" in str(exc_info.value)
            
    def test_zero_dimension_tensor(self):
        """Test handling of tensors with zero dimensions."""
        weight = np.zeros((10, 0))  # 10 output features, 0 input features
        
        fan_in, fan_out = calculate_fan_in_fan_out(weight)
        
        assert fan_in == 0, "Fan-in should be zero for zero-dimensional input"
        assert fan_out == 10, "Fan-out should match output features"
        
    def test_large_kernel_conv(self):
        """Test calculation with unusually large convolutional kernels."""
        # Create a conv weight tensor with large kernel size
        weight = np.random.randn(8, 4, 7, 7)  # 7×7 kernel
        
        fan_in, fan_out = calculate_fan_in_fan_out(weight)
        
        expected_fan_in = 4 * (7 * 7)
        expected_fan_out = 8 * (7 * 7)
        
        assert fan_in == expected_fan_in, "Fan-in calculation should work with large kernels"
        assert fan_out == expected_fan_out, "Fan-out calculation should work with large kernels"