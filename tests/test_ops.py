import pytest
from DLpy.ops import Add, Multiply
from DLpy.core import Tensor
import numpy as np

class TestBasicOps:
    """Tests for basic arithmetic operations"""
    
    def test_add_edge_cases(self):
        """Test edge cases for Add operation"""
        # Test broadcasting
        x = Tensor([[1.0]], requires_grad=True)
        y = Tensor([1.0, 2.0], requires_grad=True)
        
        with pytest.raises(ValueError):
            _ = Add.apply(x, y)
        
        # Test gradient accumulation
        x = Tensor([1.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        z = Add.apply(x, y)
        z.backward()
        
        assert np.array_equal(x.grad, [1.0])
        assert np.array_equal(y.grad, [1.0])

    def test_multiply_edge_cases(self):
        """Test edge cases for Multiply operation"""
        # Test scalar multiplication
        x = Tensor([1.0], requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = Multiply.apply(x, y)
        z.backward()
        
        assert np.array_equal(x.grad, [2.0])
        assert np.array_equal(y.grad, [1.0])
    
    def test_add_broadcasting_complex(self):
        """Test complex broadcasting scenarios in Add operation"""
        # Test broadcasting with different dimensions
        x = Tensor([[1.0]], requires_grad=True)
        y = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        with pytest.raises(ValueError):
            _ = x + y  # Incompatible shapes
            
        # Test broadcasting with scalar
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x + y
        z.backward(np.ones_like(x.data))
        assert np.sum(y.grad) == np.prod(x.shape)  # Sum of gradients equals number of elements

    def test_multiply_broadcasting_complex(self):
        """Test complex broadcasting scenarios in Multiply operation"""
        # Test scalar multiplication with matrix
        x = Tensor(2.0, requires_grad=True)
        y = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        z = x * y
        z.backward(np.ones_like(y.data))

        # Check scalar gradient - should be sum of all elements in y
        assert x.grad.shape == (1,)
        assert np.allclose(x.grad, [10.0])  # sum([1,2,3,4])

        # Check matrix gradient - should be uniformly scaled by x
        assert y.grad.shape == y.data.shape
        assert np.allclose(y.grad, np.full_like(y.data, 2.0))

        # Test broadcasting matrix with different shapes
        a = Tensor([[1.0], [2.0]], requires_grad=True)  # Shape: (2,1)
        b = Tensor([[1.0, 2.0]], requires_grad=True)    # Shape: (1,2)
        c = a * b  # Should broadcast to shape (2,2)

        assert c.data.shape == (2, 2)
        expected = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert np.allclose(c.data, expected)

        # Test gradient propagation with broadcasting
        c.backward(np.ones_like(c.data))
        assert a.grad.shape == (2, 1)
        assert b.grad.shape == (1, 2)
        # Correct expected gradients
        assert np.allclose(a.grad, np.array([[3.0], [3.0]]))  # Correct sum of gradients for each row
        assert np.allclose(b.grad, np.array([[3.0, 3.0]]))    # Correct sum of gradients for each column

    def test_add_empty_tensors(self):
        x = Tensor([], requires_grad=True)
        y = Tensor([], requires_grad=True)
        z = Add.apply(x, y)
        assert z.shape == (0,)
    
    def test_multiply_empty_tensors(self):
        x = Tensor([], requires_grad=True)
        y = Tensor([], requires_grad=True)
        z = Multiply.apply(x, y)
        assert z.shape == (0,)

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