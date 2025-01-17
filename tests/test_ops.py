import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.ops import (
    Add, Multiply, Power, Divide, Log, Exp, Sum, Mean, Max, Transpose,
    Greater, GreaterEqual, Less, LessEqual, Equal, NotEqual
)

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

class TestPowerOperations:
    """Tests for power and division operations"""
    
    def test_power_scalar(self):
        """Test power operation with scalar exponent"""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = x ** 2
        y.backward(np.array([1.0, 1.0]))
        
        assert np.allclose(y.data, [4.0, 9.0])
        assert np.allclose(x.grad, [4.0, 6.0])  # d/dx(x^2) = 2x
        
    def test_power_negative(self):
        """Test power operation with negative exponent"""
        x = Tensor([2.0, 4.0], requires_grad=True)
        y = x ** (-1)
        y.backward(np.array([1.0, 1.0]))
        
        assert np.allclose(y.data, [0.5, 0.25])
        assert np.allclose(x.grad, [-0.25, -0.0625])  # d/dx(x^-1) = -x^-2
        
    def test_division(self):
        """Test division operation"""
        x = Tensor([6.0, 8.0], requires_grad=True)
        y = Tensor([2.0, 4.0], requires_grad=True)
        z = x / y
        z.backward(np.array([1.0, 1.0]))
        
        assert np.allclose(z.data, [3.0, 2.0])
        assert np.allclose(x.grad, [0.5, 0.25])  # d/dx(x/y) = 1/y
        assert np.allclose(y.grad, [-1.5, -0.5])  # d/dy(x/y) = -x/y^2
        
    def test_division_by_zero(self):
        """Test division by zero raises error"""
        x = Tensor([1.0, 2.0])
        y = Tensor([1.0, 0.0])
        with pytest.raises(ValueError):
            _ = x / y

class TestElementwiseOperations:
    """Tests for element-wise operations"""
    
    def test_log(self):
        """Test natural logarithm"""
        x = Tensor([1.0, np.e], requires_grad=True)
        y = x.log()
        y.backward(np.array([1.0, 1.0]))
        
        assert np.allclose(y.data, [0.0, 1.0])
        assert np.allclose(x.grad, [1.0, 1/np.e])  # d/dx(log(x)) = 1/x
        
    def test_exp(self):
        """Test exponential function"""
        x = Tensor([0.0, 1.0], requires_grad=True)
        y = x.exp()
        y.backward(np.array([1.0, 1.0]))
        
        assert np.allclose(y.data, [1.0, np.e])
        assert np.allclose(y.data, x.grad)  # d/dx(exp(x)) = exp(x)

class TestReductionOperations:
    """Tests for reduction operations"""
    
    def test_sum(self):
        """Test sum reduction"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        # Test sum all elements
        y1 = x.sum()
        y1.backward()
        assert np.allclose(y1.data, 10.0)
        assert np.allclose(x.grad, np.ones_like(x.data))
        
        # Reset gradients
        x.grad = None
        
        # Test sum along axis
        y2 = x.sum(axis=0)
        y2.backward(np.array([1.0, 1.0]))
        assert np.allclose(y2.data, [4.0, 6.0])
        assert np.allclose(x.grad, np.ones_like(x.data))
        
    def test_mean(self):
        """Test mean reduction"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.mean()
        y.backward()
        
        assert np.allclose(y.data, 2.5)
        assert np.allclose(x.grad, np.full_like(x.data, 0.25))  # 1/n for each element
        
    def test_max(self):
        """Test max reduction"""
        x = Tensor([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
        y = x.max()
        y.backward()
        
        assert np.allclose(y.data, 4.0)
        expected_grad = np.array([[0.0, 1.0], [0.0, 0.0]])
        assert np.allclose(x.grad, expected_grad)

class TestMatrixOperations:
    """Tests for matrix operations"""
    
    def test_transpose(self):
        """Test matrix transpose"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.t()
        y.backward(np.ones_like(y.data))
        
        assert np.allclose(y.data, [[1.0, 3.0], [2.0, 4.0]])
        assert np.allclose(x.grad, np.ones_like(x.data))
        
    def test_transpose_3d(self):
        """Test 3D tensor transpose"""
        x = Tensor(np.arange(8).reshape(2, 2, 2), requires_grad=True)
        y = x.transpose(1, 2, 0)
        y.backward(np.ones_like(y.data))
        
        assert y.data.shape == (2, 2, 2)
        assert np.allclose(x.grad, np.ones_like(x.data))

class TestComparisonOperations:
    """Tests for comparison operations"""
    
    def test_greater(self):
        """Test greater than operation"""
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([2.0, 2.0, 2.0])
        result = x > y
        assert np.allclose(result.data, [False, False, True])
        
    def test_less_equal(self):
        """Test less than or equal operation"""
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([2.0, 2.0, 2.0])
        result = x <= y
        assert np.allclose(result.data, [True, True, False])
        
    def test_equal(self):
        """Test equality operation"""
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([1.0, 2.0, 2.0])
        result = x == y
        assert np.allclose(result.data, [True, True, False])
        
    def test_not_equal(self):
        """Test inequality operation"""
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([1.0, 2.0, 2.0])
        result = x != y
        assert np.allclose(result.data, [False, False, True])

class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_log_negative(self):
        """Test log of negative number raises error"""
        x = Tensor([-1.0])
        with pytest.raises(ValueError):
            _ = x.log()
            
    def test_power_non_scalar(self):
        """Test power with non-scalar exponent raises error"""
        x = Tensor([2.0])
        y = Tensor([1.0, 2.0])
        with pytest.raises(ValueError):
            _ = x ** y
            
    def test_reduction_keepdims(self):
        """Test reduction operations with keepdims=True"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.sum(axis=0, keepdims=True)
        assert y.shape == (1, 2)
        
    def test_broadcasting_division(self):
        """Test division with broadcasting"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = Tensor([2.0])
        z = x / y
        assert z.shape == x.shape
        assert np.allclose(z.data, [[0.5, 1.0], [1.5, 2.0]])