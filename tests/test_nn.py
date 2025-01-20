import pytest
import numpy as np
from DLpy.core import Tensor, Module
from DLpy.nn.linear import Linear


class TestLinearLayer:
    """Test suite for the Linear layer implementation."""
    
    def test_linear_layer_creation(self):
        """Test that linear layers are created with correct shapes and initialization."""
        in_features, out_features = 5, 3
        layer = Linear(in_features, out_features)
        
        # Test weight dimensions
        assert layer.weight.shape == (in_features, out_features)
        assert layer.weight.requires_grad
        
        # Test bias dimensions
        assert layer.bias is not None
        assert layer.bias.shape == (out_features,)
        assert layer.bias.requires_grad
        
        # Test layer without bias
        layer_no_bias = Linear(in_features, out_features, bias=False)
        assert layer_no_bias.bias is None
        
    def test_linear_forward(self):
        """Test the forward pass of the linear layer."""
        # Create a simple linear layer with known weights for testing
        layer = Linear(2, 3)
        layer.weight.data = np.array([[1., 2., 3.], [4., 5., 6.]])
        layer.bias.data = np.array([0.1, 0.2, 0.3])
        
        # Create input tensor
        x = Tensor([[1., 2.]])  # Batch size 1, 2 features
        
        # Compute expected output manually
        expected_output = np.array([[9.1, 12.2, 15.3]])  # (1×2) @ (2×3) + bias
        
        # Get actual output
        output = layer(x)
        
        # Compare results
        assert isinstance(output, Tensor)
        assert output.shape == (1, 3)
        assert np.allclose(output.data, expected_output)
        
    def test_linear_backward(self):
        """Test the backward pass and gradient computation of the linear layer."""
        # Create a layer with specific weights for testing
        layer = Linear(2, 1)
        layer.weight.data = np.array([[1.], [2.]])
        layer.bias.data = np.array([0.])
        
        # Forward pass
        x = Tensor([[1., 2.]], requires_grad=True)
        output = layer(x)
        
        # Backward pass
        output.backward(np.array([[1.]]))
        
        # Check input gradients
        expected_input_grad = np.array([[1., 2.]])  # Gradient w.r.t input
        assert np.allclose(x.grad, expected_input_grad)
        
        # Check weight gradients
        expected_weight_grad = np.array([[1.], [2.]])  # Gradient w.r.t weights
        assert np.allclose(layer.weight.grad, expected_weight_grad)
        
        # Check bias gradients
        expected_bias_grad = np.array([1.])  # Gradient w.r.t bias
        assert np.allclose(layer.bias.grad, expected_bias_grad)
        
    def test_linear_batch_processing(self):
        """Test that the linear layer correctly handles batched inputs."""
        layer = Linear(3, 2)
        batch_size = 4
        x = Tensor(np.random.randn(batch_size, 3))
        
        output = layer(x)
        assert output.shape == (batch_size, 2)
        
    def test_weight_initialization(self):
        """Test that weights are properly initialized using He initialization."""
        in_features, out_features = 100, 100
        layer = Linear(in_features, out_features)
        
        # Check if weights follow He initialization statistics
        weights = layer.weight.data
        mean = np.mean(weights)
        std = np.std(weights)
        
        # He initialization should have mean ≈ 0 and std ≈ sqrt(2/in_features)
        expected_std = np.sqrt(2.0 / in_features)
        assert abs(mean) < 0.1  # Mean should be close to 0
        assert abs(std - expected_std) < 0.1  # Std should be close to expected


class TestModule:
    """Test suite for the base Module class."""
    
    class SimpleModule(Module):
        """A simple module for testing purposes."""
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(2, 3)
            self.linear2 = Linear(3, 1)
            self.register_buffer('running_mean', Tensor(np.zeros(3)))
            
        def forward(self, x):
            x = self.linear1(x)
            return self.linear2(x)
    
    def test_module_parameter_registration(self):
        """Test that parameters are correctly registered and tracked."""
        model = self.SimpleModule()
        
        # Count parameters
        params = list(model.parameters())
        assert len(params) == 4  # 2 weights + 2 biases
        
        # Check named parameters
        named_params = dict(model.named_parameters())
        assert 'linear1.weight' in named_params
        assert 'linear1.bias' in named_params
        assert 'linear2.weight' in named_params
        assert 'linear2.bias' in named_params
        
    def test_module_buffer_registration(self):
        """Test that buffers are correctly registered."""
        model = self.SimpleModule()
        assert 'running_mean' in model._buffers
        assert isinstance(model._buffers['running_mean'], Tensor)
        
    def test_module_train_eval_modes(self):
        """Test switching between training and evaluation modes."""
        model = self.SimpleModule()
        
        # Test train mode
        model.train()
        assert model.training
        assert model.linear1.training
        assert model.linear2.training
        
        # Test eval mode
        model.eval()
        assert not model.training
        assert not model.linear1.training
        assert not model.linear2.training
        
    def test_module_repr(self):
        """Test the string representation of modules."""
        model = self.SimpleModule()
        repr_str = repr(model)
        
        # Check that repr includes important information
        assert 'SimpleModule' in repr_str
        assert 'linear1' in repr_str
        assert 'linear2' in repr_str


class TestEndToEnd:
    """End-to-end tests for neural network components."""
    
    def test_simple_network(self):
        """Test a simple network with multiple layers."""
        # Create a simple network
        class SimpleNet(Module):
            def __init__(self):
                super().__init__()
                self.linear1 = Linear(2, 3)
                self.linear2 = Linear(3, 1)
                
            def forward(self, x):
                x = self.linear1(x)
                return self.linear2(x)
        
        # Create model and input
        model = SimpleNet()
        x = Tensor([[1., 2.]], requires_grad=True)
        
        # Forward pass
        output = model(x)
        assert output.shape == (1, 1)
        
        # Backward pass
        output.backward(np.array([[1.]]))
        
        # Check that all parameters have gradients
        for param in model.parameters():
            assert param.grad is not None