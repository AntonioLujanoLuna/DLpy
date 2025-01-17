import pytest
from DLpy.core import Function, Tensor
import numpy as np

class TestFunction:
    """Tests for Function base class and utilities"""
    
    class TestFunction(Function):
        """Simple test function implementation"""
        
        @staticmethod
        def forward(ctx, x, y=None):
            ctx.save_for_backward(x)
            ctx.save_arguments(y=y)
            return Tensor(x.data * 2)
            
        @staticmethod
        def backward(ctx, grad_output, grad_dict):
            x, = ctx.saved_tensors
            y = ctx.saved_arguments["y"]
            
            if x.requires_grad:
                grad_dict[id(x)] = grad_output * 2

    def test_function_application(self):
        """Test applying a function to inputs"""
        x = Tensor([1.0], requires_grad=True)
        result = self.TestFunction.apply(x, y=2.0)
        
        assert isinstance(result, Tensor)
        assert np.array_equal(result.data, [2.0])
        assert result.requires_grad
        assert result._backward_fn is not None

    def test_verify_backward(self):
        """Test gradient verification utility"""
        def forward_fn(x):
            return x * 2
            
        def correct_backward_fn(ctx, grad_output):
            return grad_output * 2
            
        def incorrect_backward_fn(ctx, grad_output):
            return grad_output * 3
        
        # Test with correct gradients
        x = np.array([1.0])
        assert Function.verify_backward(forward_fn, correct_backward_fn, (x,))
        
        # Test with incorrect gradients
        assert not Function.verify_backward(forward_fn, incorrect_backward_fn, (x,))

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        
        class IncompleteFunction(Function):
            pass
            
        with pytest.raises(TypeError):
            IncompleteFunction()