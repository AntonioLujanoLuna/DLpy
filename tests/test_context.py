import pytest
from DLpy.core import Context, Tensor
import numpy as np

class TestContext:
    """Tests for Context class functionality"""
    
    def test_save_and_retrieve_tensors(self):
        """Test saving and retrieving tensors"""
        ctx = Context()
        tensor1 = Tensor([1.0])
        tensor2 = Tensor([2.0])
        
        ctx.save_for_backward(tensor1, tensor2)
        saved = ctx.saved_tensors
        
        assert len(saved) == 2
        assert np.array_equal(saved[0].data, tensor1.data)
        assert np.array_equal(saved[1].data, tensor2.data)

    def test_save_and_retrieve_arguments(self):
        """Test saving and retrieving non-tensor arguments"""
        ctx = Context()
        ctx.save_arguments(arg1="test", arg2=42)
        
        args = ctx.saved_arguments
        assert args["arg1"] == "test"
        assert args["arg2"] == 42
        assert isinstance(args, dict)

    def test_intermediate_values(self):
        """Test storing and retrieving intermediate values"""
        ctx = Context()
        
        # Store various types of values
        ctx.store_intermediate("scalar", 42)
        ctx.store_intermediate("list", [1, 2, 3])
        ctx.store_intermediate("tensor", Tensor([1.0]))
        
        # Retrieve and verify values
        assert ctx.get_intermediate("scalar") == 42
        assert ctx.get_intermediate("list") == [1, 2, 3]
        assert isinstance(ctx.get_intermediate("tensor"), Tensor)
        
        # Test retrieving non-existent key
        with pytest.raises(KeyError):
            ctx.get_intermediate("nonexistent")

    def test_clear_functionality(self):
        """Test clearing all stored data"""
        ctx = Context()
        
        # Store various types of data
        ctx.save_for_backward(Tensor([1.0]))
        ctx.save_arguments(arg1="test")
        ctx.store_intermediate("key", "value")
        
        # Clear all data
        ctx.clear()
        
        # Verify everything is cleared
        assert len(ctx._saved_tensors) == 0
        assert len(ctx._non_tensor_args) == 0
        assert len(ctx._intermediate_values) == 0