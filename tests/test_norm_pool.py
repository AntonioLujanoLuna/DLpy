import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn.pooling import MaxPool2d, AvgPool2d
from DLpy.nn.normalization import InstanceNorm2d, GroupNorm

class TestPoolingLayers:
    """Tests for pooling layers"""
    
    def test_maxpool2d_basic(self):
        """Test basic MaxPool2d functionality"""
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]]), requires_grad=True)
        
        output = pool(x)
        assert output.shape == (1, 1, 2, 2)
        assert np.array_equal(output.data, [[[[6, 8],
                                            [14, 16]]]])
                                            
        # Test backward pass
        output.backward(np.ones_like(output.data))
        assert x.grad is not None
        # Only maximum values should receive gradients
        expected_grad = np.zeros_like(x.data)
        expected_grad[0, 0, 1, 1] = 1
        expected_grad[0, 0, 1, 3] = 1
        expected_grad[0, 0, 3, 1] = 1
        expected_grad[0, 0, 3, 3] = 1
        assert np.array_equal(x.grad, expected_grad)

    def test_avgpool2d_basic(self):
        """Test basic AvgPool2d functionality"""
        pool = AvgPool2d(kernel_size=2, stride=2)
        # Use float values to avoid dtype mismatch during gradient calculation
        x = Tensor(np.array([[[[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0],
                            [13.0, 14.0, 15.0, 16.0]]]]), requires_grad=True)
        
        output = pool(x)
        assert output.shape == (1, 1, 2, 2)
        expected_output = np.array([[[[3.5, 5.5],
                                    [11.5, 13.5]]]])
        assert np.allclose(output.data, expected_output)
        
        # Test backward pass
        output.backward(np.ones_like(output.data))

    def test_pooling_with_padding(self):
        """Test pooling layers with padding"""
        pool_max = MaxPool2d(kernel_size=2, stride=2, padding=1)
        pool_avg = AvgPool2d(kernel_size=2, stride=2, padding=1)
        
        x = Tensor(np.random.randn(1, 3, 4, 4))
        output_max = pool_max(x)
        output_avg = pool_avg(x)
        
        # Check output shapes
        assert output_max.shape == (1, 3, 3, 3)
        assert output_avg.shape == (1, 3, 3, 3)

    def test_pooling_with_different_kernel_stride(self):
        """Test pooling with different kernel and stride sizes"""
        pool_max = MaxPool2d(kernel_size=3, stride=2)
        pool_avg = AvgPool2d(kernel_size=3, stride=2)
        
        x = Tensor(np.random.randn(2, 2, 5, 5))
        output_max = pool_max(x)
        output_avg = pool_avg(x)
        
        expected_shape = (2, 2, 2, 2)
        assert output_max.shape == expected_shape
        assert output_avg.shape == expected_shape

    def test_pooling_edge_cases(self):
        """Test pooling layers with edge cases"""
        # Test with single pixel per channel
        x = Tensor(np.random.randn(1, 1, 1, 1))
        pool_max = MaxPool2d(kernel_size=1)
        pool_avg = AvgPool2d(kernel_size=1)
        
        assert pool_max(x).shape == (1, 1, 1, 1)
        assert pool_avg(x).shape == (1, 1, 1, 1)
        
        # Test with kernel size larger than input
        x = Tensor(np.random.randn(1, 1, 2, 2))
        pool_max = MaxPool2d(kernel_size=3, stride=1, padding=1)
        pool_avg = AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        assert pool_max(x).shape == (1, 1, 2, 2)
        assert pool_avg(x).shape == (1, 1, 2, 2)

class TestNormalizationLayers:
    """Tests for normalization layers"""
    
    def test_instance_norm_basic(self):
        """Test basic InstanceNorm2d functionality"""
        norm = InstanceNorm2d(num_features=2)
        x = Tensor(np.random.randn(4, 2, 8, 8), requires_grad=True)
        
        output = norm(x)
        # Check output shape
        assert output.shape == x.shape
        
        # Check normalization statistics
        # Should be approximately zero mean and unit variance per instance and channel
        x_reshaped = output.data.reshape(4, 2, -1)
        means = x_reshaped.mean(axis=2)
        vars = x_reshaped.var(axis=2)
        assert np.allclose(means, 0, atol=1e-5)
        assert np.allclose(vars, 1, atol=1e-5)

    def test_group_norm_basic(self):
        """Test basic GroupNorm functionality"""
        norm = GroupNorm(num_groups=2, num_channels=4)
        x = Tensor(np.random.randn(3, 4, 8, 8), requires_grad=True)
        
        output = norm(x)
        # Check output shape
        assert output.shape == x.shape
        
        # Check normalization statistics
        # Reshape to (N, G, C/G, H, W) for checking stats
        x_reshaped = output.data.reshape(3, 2, 2, 8, 8)
        means = x_reshaped.mean(axis=(2, 3, 4))
        vars = x_reshaped.var(axis=(2, 3, 4))
        assert np.allclose(means, 0, atol=1e-5)
        assert np.allclose(vars, 1, atol=1e-5)

    def test_instance_norm_affine(self):
        """Test InstanceNorm2d with affine parameters"""
        norm = InstanceNorm2d(num_features=2, affine=True)
        x = Tensor(np.random.randn(3, 2, 4, 4), requires_grad=True)
        
        # Set specific affine parameters
        norm.weight.data = np.array([2.0, 3.0])
        norm.bias.data = np.array([0.5, -0.5])
        
        output = norm(x)
        assert output.shape == x.shape
        
        # Test backward pass
        output.backward(np.ones_like(output.data))
        assert x.grad is not None
        assert norm.weight.grad is not None
        assert norm.bias.grad is not None

    def test_group_norm_invalid_groups(self):
        """Test GroupNorm with invalid number of groups"""
        with pytest.raises(ValueError):
            GroupNorm(num_groups=3, num_channels=4)  # 4 is not divisible by 3

    def test_normalization_eval_mode(self):
        """Test normalization layers in eval mode"""
        # Correct num_features to match input channels (4 instead of 2)
        norm_instance = InstanceNorm2d(num_features=4, track_running_stats=True)
        norm_group = GroupNorm(num_groups=2, num_channels=4)
        
        x = Tensor(np.random.randn(2, 4, 4, 4))  # 4 channels
        
        # Test in train mode
        train_output_instance = norm_instance(x)
        train_output_group = norm_group(x)
        
        # Test in eval mode
        norm_instance.eval()
        norm_group.eval()
        eval_output_instance = norm_instance(x)
        eval_output_group = norm_group(x)
        
        # GroupNorm should behave the same in both modes
        assert np.allclose(train_output_group.data, eval_output_group.data)
        
        # InstanceNorm with track_running_stats=True should use running stats in eval
        if norm_instance.track_running_stats:
            # Check means from running stats vs batch stats
            assert not np.allclose(train_output_instance.data, eval_output_instance.data)

    def test_normalization_no_affine(self):
        """Test normalization layers without affine parameters"""
        norm_instance = InstanceNorm2d(num_features=2, affine=False)
        norm_group = GroupNorm(num_groups=2, num_channels=4, affine=False)
        
        x = Tensor(np.random.randn(2, 4, 4, 4))  
        
        output_instance = norm_instance(x)
        output_group = norm_group(x)
        
        assert norm_instance.weight is None
        assert norm_instance.bias is None
        assert norm_group.weight is None
        assert norm_group.bias is None

    def test_instance_norm_momentum(self):
        """Test InstanceNorm2d with different momentum values"""
        norm = InstanceNorm2d(num_features=2, momentum=0.1, track_running_stats=True)
        x = Tensor(np.random.randn(2, 2, 4, 4))
        
        # Multiple forward passes should update running statistics
        initial_mean = norm.running_mean.data.copy()
        initial_var = norm.running_var.data.copy()
        
        for _ in range(5):
            _ = norm(x)
            
        assert not np.array_equal(norm.running_mean.data, initial_mean)
        assert not np.array_equal(norm.running_var.data, initial_var)

    def test_group_norm_different_groups(self):
        """Test GroupNorm with different group configurations"""
        norm1 = GroupNorm(num_groups=1, num_channels=4)
        norm2 = GroupNorm(num_groups=4, num_channels=4)
        
        x = Tensor(np.random.randn(2, 4, 4, 4))
        
        output1 = norm1(x)
        output2 = norm2(x)
        
        assert output1.shape == x.shape
        assert output2.shape == x.shape
        
        # Check if the actual outputs are different
        assert not np.allclose(output1.data, output2.data, atol=1e-5)