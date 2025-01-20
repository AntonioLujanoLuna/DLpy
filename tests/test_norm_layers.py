import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn import BatchNorm1d, BatchNorm2d, LayerNorm

class TestBatchNorm1d:
    """Tests for BatchNorm1d layer"""
    
    def test_basic_normalization(self):
        """Test basic normalization without affine transform"""
        batch_norm = BatchNorm1d(3, affine=False)
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
        
        output = batch_norm(x)
        
        # Output should be normalized (mean ≈ 0, std ≈ 1)
        assert np.allclose(output.data.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(output.data.std(axis=0), 1, atol=1e-6)
        
    def test_affine_transform(self):
        """Test normalization with affine transform"""
        batch_norm = BatchNorm1d(2)
        batch_norm.weight.data = np.array([2.0, 3.0])
        batch_norm.bias.data = np.array([1.0, 2.0])

        x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float64))
        output = batch_norm(x)

        # Output should be normalized and then transformed
        mean = x.data.mean(axis=0)
        var = x.data.var(axis=0, ddof=0)  # Use ddof=0 to match the implementation
        normalized = (x.data - mean) / np.sqrt(var)
        # No extra broadcasting needed, just direct multiply and add
        expected = normalized * batch_norm.weight.data + batch_norm.bias.data

        assert np.allclose(output.data, expected)
        
    def test_running_stats(self):
        """Test tracking of running statistics"""
        batch_norm = BatchNorm1d(2, momentum=0.1)
        x1 = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float64))
        x2 = Tensor(np.array([[2, 3], [4, 5]], dtype=np.float64))
        
        # Training mode
        batch_norm.train()
        _ = batch_norm(x1)
        initial_mean = batch_norm.running_mean.data.copy()
        _ = batch_norm(x2)
        
        # Running mean should be updated
        assert not np.array_equal(batch_norm.running_mean.data, initial_mean)
        
        # Eval mode
        batch_norm.eval()
        output = batch_norm(x1)
        # Should use running statistics
        expected = (x1.data - batch_norm.running_mean.data) / np.sqrt(batch_norm.running_var.data + batch_norm.eps)
        assert np.allclose(output.data, expected)
        
    def test_backward(self):
        """Test backward pass with gradients"""
        batch_norm = BatchNorm1d(2)
        x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float64), requires_grad=True)
        
        output = batch_norm(x)
        output.backward(np.ones_like(output.data))
        
        assert x.grad is not None
        assert batch_norm.weight.grad is not None
        assert batch_norm.bias.grad is not None

class TestBatchNorm2d:
    """Tests for BatchNorm2d layer"""
    
    def test_shape_handling(self):
        """Test proper handling of 4D input"""
        batch_norm = BatchNorm2d(3)  # 3 channels
        x = Tensor(np.random.randn(2, 3, 4, 4))  # (N, C, H, W)
        
        output = batch_norm(x)
        assert output.shape == x.shape
        
    def test_channel_normalization(self):
        """Test that normalization is applied per channel"""
        batch_norm = BatchNorm2d(2, affine=False)
        x = Tensor(np.random.randn(3, 2, 4, 4))  # (N, C, H, W)
        
        output = batch_norm(x)
        
        # Reshape output to (N*H*W, C) for checking statistics
        output_reshaped = output.data.transpose(0, 2, 3, 1).reshape(-1, 2)
        
        # Each channel should be normalized
        assert np.allclose(output_reshaped.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(output_reshaped.std(axis=0), 1, atol=1e-6)
        
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes"""
        batch_norm = BatchNorm2d(3)
        x = Tensor(np.random.randn(2, 3))  # 2D input
        
        with pytest.raises(AssertionError):
            _ = batch_norm(x)

class TestLayerNorm:
    """Tests for LayerNorm layer"""
    
    def test_basic_normalization(self):
        """Test basic layer normalization without affine transform"""
        norm = LayerNorm(normalized_shape=(3,), elementwise_affine=False)
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
        
        output = norm(x)
        
        # Each sample should be independently normalized
        for i in range(len(x)):
            assert np.allclose(output.data[i].mean(), 0, atol=1e-6)
            assert np.allclose(output.data[i].std(), 1, atol=1e-6)
            
    def test_affine_transform(self):
        """Test normalization with affine transform"""
        norm = LayerNorm(normalized_shape=(2,))
        norm.weight.data = np.array([2.0, 3.0])
        norm.bias.data = np.array([1.0, 2.0])
        
        x = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float64))
        output = norm(x)
        
        # Manually compute expected output
        means = x.data.mean(axis=1, keepdims=True)
        stds = x.data.std(axis=1, keepdims=True)
        normalized = (x.data - means) / (stds + norm.eps)
        expected = normalized * norm.weight.data + norm.bias.data
        
        assert np.allclose(output.data, expected)
        
    def test_different_shapes(self):
        """Test normalization over different shapes"""
        # 2D input
        norm1 = LayerNorm((3,))
        x1 = Tensor(np.random.randn(2, 3))
        out1 = norm1(x1)
        assert out1.shape == x1.shape
        
        # 3D input
        norm2 = LayerNorm((4, 3))
        x2 = Tensor(np.random.randn(2, 4, 3))
        out2 = norm2(x2)
        assert out2.shape == x2.shape
        
        # 4D input
        norm3 = LayerNorm((5, 4, 3))
        x3 = Tensor(np.random.randn(2, 5, 4, 3))
        out3 = norm3(x3)
        assert out3.shape == x3.shape
        
    def test_backward(self):
        """Test backward pass with gradients"""
        norm = LayerNorm((3,))
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), requires_grad=True)
        
        output = norm(x)
        output.backward(np.ones_like(output.data))
        
        assert x.grad is not None
        assert norm.weight.grad is not None
        assert norm.bias.grad is not None
        
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes"""
        norm = LayerNorm((3, 2))
        x = Tensor(np.random.randn(2, 3))  # Wrong shape
        
        with pytest.raises(ValueError):
            _ = norm(x)
            
    def test_scalar_normalized_shape(self):
        """Test initialization with scalar normalized_shape"""
        norm = LayerNorm(3)  # Should be equivalent to (3,)
        x = Tensor(np.random.randn(2, 3))
        
        output = norm(x)
        assert output.shape == x.shape