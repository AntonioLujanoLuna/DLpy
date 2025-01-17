import pytest
import numpy as np
from DLpy.core import Tensor

class TestReshapeOp:
    """Tests for Reshape operation"""

    def test_reshape_edge_cases(self):
        """Test edge cases for Reshape operation"""
        x = Tensor([1.0, 2.0], requires_grad=True)
        
        # Test invalid shape
        with pytest.raises(ValueError):
            _ = x.reshape(3)  # Invalid shape
        
        # Test gradients with different shapes
        y = x.reshape(2, 1)
        y.backward(np.array([[1.0], [1.0]]))
        assert np.array_equal(x.grad, [1.0, 1.0])

class TestAdvancedOperations:
    """Additional tests for basic operations"""
    
    def test_broadcasting_edge_cases(self):
        """Test broadcasting with different dimensions"""
        # Test broadcasting scalar to matrix
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x * y
        z.backward(np.ones_like(x.data))
        assert y.grad.shape == (1,)
        assert np.array_equal(x.grad, [[2.0, 2.0], [2.0, 2.0]])
        
        # Test broadcasting vector to matrix
        x = Tensor([[1.0], [2.0]], requires_grad=True)
        y = Tensor([1.0, 2.0], requires_grad=True)
        z = x + y  # Should broadcast to [[1,2], [2,3]]
        z.backward(np.ones((2, 2)))
        assert x.grad.shape == (2, 1)
        assert y.grad.shape == (2,)
        
    def test_zero_gradient_handling(self):
        """Test operations with zero gradients"""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x * y
        z.backward(np.zeros_like(z.data))
        assert np.all(x.grad == 0)
        assert np.all(y.grad == 0)
        
    def test_non_differentiable_inputs(self):
        """Test operations with non-differentiable inputs"""
        x = Tensor([1.0, 2.0], requires_grad=False)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x * y
        z.backward(np.ones_like(z.data))
        assert x.grad is None  # Non-differentiable input should have no gradient
        assert np.array_equal(y.grad, [1.0, 2.0])

    def test_tensor_creation_edge_cases(self):
        """Test edge cases in tensor creation"""
        # Test with different dtypes
        t1 = Tensor([1, 2, 3], dtype=np.int32)
        assert t1.dtype == np.int32
        
        # Test with nested lists
        t2 = Tensor([[1, 2], [3, 4]])
        assert t2.shape == (2, 2)
        
        # Test with another tensor
        t3 = Tensor(t2)
        assert np.array_equal(t3.data, t2.data)

    def test_backward_edge_cases(self):
        """Test edge cases in backward pass"""
        # Test backward with scalar tensor
        x = Tensor(2.0, requires_grad=True)
        y = x * 2
        y.backward(np.array(3.0))
        assert x.grad is not None
        
        # Test backward with non-scalar tensor without gradient
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2
        with pytest.raises(RuntimeError):
            y.backward()  # Should raise error for non-scalar

    def test_repr_and_str(self):
        """Test string representations"""
        t = Tensor([1.0, 2.0], requires_grad=True)
        assert 'Tensor' in repr(t)
        assert 'requires_grad=True' in repr(t)