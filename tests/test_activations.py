import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn.base.activations import (
    relu, leaky_relu, elu, gelu, sigmoid, tanh,
)

class TestActivations:
    """Tests for activation functions"""
    
    def test_relu(self):
        """Test ReLU activation"""
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = relu(x)
        y.backward(np.ones_like(x.data))
        
        assert np.array_equal(y.data, [0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.array_equal(x.grad, [0.0, 0.0, 0.0, 1.0, 1.0])
        
    def test_leaky_relu(self):
        """Test Leaky ReLU activation"""
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        slope = 0.1
        y = leaky_relu(x, negative_slope=slope)
        y.backward(np.ones_like(x.data))
        
        expected_forward = [-0.2, -0.1, 0.0, 1.0, 2.0]
        expected_backward = [slope, slope, slope, 1.0, 1.0]
        
        assert np.allclose(y.data, expected_forward)
        assert np.allclose(x.grad, expected_backward)
        
    def test_elu(self):
        """Test ELU activation"""
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        alpha = 1.0
        y = elu(x, alpha=alpha)
        y.backward(np.ones_like(x.data))
        
        expected_forward = [alpha * (np.exp(-2.0) - 1), alpha * (np.exp(-1.0) - 1), 0.0, 1.0, 2.0]
        expected_backward = [alpha * np.exp(-2.0), alpha * np.exp(-1.0), alpha * 1.0, 1.0, 1.0]
        
        assert np.allclose(y.data, expected_forward)
        assert np.allclose(x.grad, expected_backward)
        
    def test_gelu(self):
        """Test GELU activation"""
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = gelu(x)
        y.backward(np.ones_like(x.data))
        
        # Values should be finite and have correct shape
        assert not np.any(np.isnan(y.data))
        assert not np.any(np.isinf(y.data))
        assert y.data.shape == x.data.shape
        assert not np.any(np.isnan(x.grad))
        assert not np.any(np.isinf(x.grad))
        
        # Test specific known values
        assert np.allclose(y.data[2], 0.0)  # GELU(0) = 0
        assert y.data[3] > 0.8  # GELU(1) ≈ 0.841
        assert y.data[1] < 0  # GELU(-1) is negative
        
    def test_sigmoid(self):
        """Test Sigmoid activation"""
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = sigmoid(x)
        y.backward(np.ones_like(x.data))
        
        sigmoid_x = 1 / (1 + np.exp(-x.data))
        expected_backward = sigmoid_x * (1 - sigmoid_x)
        
        assert np.allclose(y.data, sigmoid_x)
        assert np.allclose(x.grad, expected_backward)
        
        # Test special values
        assert np.allclose(y.data[2], 0.5)  # sigmoid(0) = 0.5
        assert y.data[0] < 0.5  # sigmoid(-2) < 0.5
        assert y.data[4] > 0.5  # sigmoid(2) > 0.5
        
    def test_tanh(self):
        """Test Tanh activation"""
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = tanh(x)
        y.backward(np.ones_like(x.data))
        
        expected_forward = np.tanh(x.data)
        expected_backward = 1 - np.tanh(x.data) ** 2
        
        assert np.allclose(y.data, expected_forward)
        assert np.allclose(x.grad, expected_backward)
        
        # Test special values
        assert np.allclose(y.data[2], 0.0)  # tanh(0) = 0
        assert y.data[0] < -0.9  # tanh(-2) ≈ -0.964
        assert y.data[4] > 0.9   # tanh(2) ≈ 0.964

class TestNumericalStability:
    """Tests for numerical stability of activation functions"""
    
    def test_sigmoid_stability(self):
        """Test Sigmoid with large inputs"""
        x = Tensor([1000.0, -1000.0], requires_grad=True)
        y = sigmoid(x)
        y.backward(np.ones_like(x.data))
        
        # Check that values are properly clamped
        assert np.allclose(y.data[0], 1.0)
        assert np.allclose(y.data[1], 0.0)
        assert not np.any(np.isnan(y.data))
        assert not np.any(np.isnan(x.grad))
        
    def test_elu_stability(self):
        """Test ELU with large negative inputs"""
        x = Tensor([-1000.0], requires_grad=True)
        y = elu(x)
        y.backward(np.ones_like(x.data))
        
        # Should be close to -1.0 for large negative values
        assert np.allclose(y.data[0], -1.0, rtol=1e-3)
        assert not np.any(np.isnan(y.data))
        assert not np.any(np.isnan(x.grad))
        
    def test_gelu_stability(self):
        """Test GELU with large inputs"""
        x = Tensor([1000.0, -1000.0], requires_grad=True)
        y = gelu(x)
        y.backward(np.ones_like(x.data))
        
        # Check that outputs are finite
        assert not np.any(np.isnan(y.data))
        assert not np.any(np.isinf(y.data))
        assert not np.any(np.isnan(x.grad))
        assert not np.any(np.isinf(x.grad))

class TestGradientFlow:
    """Tests for gradient flow through activation functions"""
    
    def test_relu_dead_neurons(self):
        """Test ReLU gradient flow for negative inputs"""
        x = Tensor([-1.0], requires_grad=True)
        y = relu(x)
        y.backward(np.array([1.0]))
        
        assert x.grad[0] == 0.0  # Gradient should be zero for negative input
        
    def test_leaky_relu_gradient_flow(self):
        """Test Leaky ReLU gradient flow"""
        x = Tensor([-1.0], requires_grad=True)
        slope = 0.01
        y = leaky_relu(x, negative_slope=slope)
        y.backward(np.array([1.0]))
        
        assert x.grad[0] == slope  # Should have small but non-zero gradient
        
    def test_elu_gradient_flow(self):
        """Test ELU gradient flow"""
        x = Tensor([-1.0, 1.0], requires_grad=True)
        y = elu(x)
        y.backward(np.ones_like(x.data))
        
        assert x.grad[0] > 0.0  # Should have positive gradient for negative input
        assert x.grad[1] == 1.0  # Should have gradient 1 for positive input

class TestShapes:
    """Tests for handling different input shapes"""
    
    def test_batch_input(self):
        """Test activations with batched input"""
        batch_size, features = 32, 10
        x = Tensor(np.random.randn(batch_size, features), requires_grad=True)
        
        # Test all activations with batched input
        activations = [relu, leaky_relu, elu, gelu, sigmoid, tanh]
        for activation in activations:
            y = activation(x)
            assert y.shape == x.shape
            y.backward(np.ones_like(x.data))
            assert x.grad.shape == x.shape
            
    def test_scalar_input_single(self):
        """Test scalar input handling for a single activation"""
        x = Tensor(2.0, requires_grad=True)
        y = relu(x)
        
        # Check forward pass maintains scalar nature
        assert y.data.ndim == 0
        assert isinstance(y.data, np.ndarray)
        assert y.data.shape == ()
        
        # Check backward pass (gradient should be size 1 array as in PyTorch)
        y.backward(np.array(1.0))
        assert x.grad.size == 1
        assert isinstance(x.grad, np.ndarray)

    def test_scalar_input(self):
        """Test activations with scalar input"""
        x = Tensor(2.0, requires_grad=True)
        
        # Test all activations with scalar input
        activations = [relu, leaky_relu, elu, gelu, sigmoid, tanh]
        for activation in activations:
            y = activation(x)
            assert y.data.ndim == 0  # Should preserve scalar nature
            y.backward(np.array(1.0))
            assert x.grad.size == 1  # Gradient should be size 1 array (matching PyTorch behavior)

class TestCustomGradients:
    """Tests for custom gradient computations"""
    
    def test_relu_custom_gradient(self):
        """Test ReLU with custom gradient"""
        x = Tensor([1.0, -1.0], requires_grad=True)
        y = relu(x)
        y.backward(np.array([2.0, 2.0]))  # Custom gradient values
        
        assert np.array_equal(x.grad, [2.0, 0.0])  # Should scale gradient for positive input
        
    def test_sigmoid_custom_gradient(self):
        """Test Sigmoid with custom gradient"""
        x = Tensor([0.0], requires_grad=True)
        y = sigmoid(x)
        y.backward(np.array([2.0]))  # Custom gradient value
        
        expected_grad = 2.0 * 0.25  # 2.0 * sigmoid(0) * (1 - sigmoid(0))
        assert np.allclose(x.grad, expected_grad)