import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn import Dropout, Dropout2d

class TestDropout:
    """Tests for Dropout layer"""
    
    def test_training_mode(self):
        """Test dropout behavior in training mode"""
        dropout = Dropout(p=0.5)
        x = Tensor(np.ones((100, 100)))  # Large tensor for stable statistics
        
        dropout.train()  # Ensure training mode
        output = dropout(x)
        
        # Check that approximately p% of elements are zeroed
        zero_ratio = np.mean(output.data == 0)
        assert np.abs(zero_ratio - 0.5) < 0.1  # Allow some statistical variation
        
        # Check scaling of non-zero elements
        nonzero_vals = output.data[output.data != 0]
        assert np.allclose(nonzero_vals, 2.0)  # Scale factor should be 1/(1-p) = 2
        
    def test_eval_mode(self):
        """Test dropout behavior in evaluation mode"""
        dropout = Dropout(p=0.5)
        x = Tensor(np.ones((10, 10)))
        
        dropout.eval()  # Set to evaluation mode
        output = dropout(x)
        
        # In eval mode, should return input unchanged
        assert np.array_equal(output.data, x.data)
        
    def test_invalid_p(self):
        """Test error handling for invalid dropout probability"""
        with pytest.raises(ValueError):
            Dropout(p=1.5)
        with pytest.raises(ValueError):
            Dropout(p=-0.5)
            
    def test_stochastic_behavior(self):
        """Test that different calls produce different masks"""
        dropout = Dropout(p=0.5)
        x = Tensor(np.ones((10, 10)))
        
        dropout.train()
        output1 = dropout(x)
        output2 = dropout(x)
        
        # Different calls should produce different results
        assert not np.array_equal(output1.data, output2.data)
        
    def test_inplace_operation(self):
        """Test inplace dropout operation"""
        dropout = Dropout(p=0.5, inplace=True)
        x = Tensor(np.ones((10, 10)))
        original_data = x.data.copy()
        
        dropout.train()
        output = dropout(x)
        
        # Output should be same object as input
        assert output is x
        # Data should be modified
        assert not np.array_equal(x.data, original_data)

class TestDropout2d:
    """Tests for Dropout2d layer"""
    
    def test_channel_dropout(self):
        """Test that entire channels are dropped"""
        dropout = Dropout2d(p=0.5)
        x = Tensor(np.ones((2, 4, 3, 3)))  # (N, C, H, W)
        
        dropout.train()
        output = dropout(x)
        
        # Check that channels are entirely zero or scaled
        for n in range(x.shape[0]):
            for c in range(x.shape[1]):
                channel = output.data[n, c]
                # Channel should either be all zero or all scaled
                assert np.all(channel == 0) or np.all(np.isclose(channel, 2.0))
                
    def test_invalid_dimensions(self):
        """Test error handling for invalid input dimensions"""
        dropout = Dropout2d(p=0.5)
        x = Tensor(np.ones((2, 3)))  # 2D input
        
        with pytest.raises(AssertionError):
            dropout(x)
            
    def test_eval_mode_2d(self):
        """Test Dropout2d behavior in evaluation mode"""
        dropout = Dropout2d(p=0.5)
        x = Tensor(np.ones((2, 3, 4, 4)))
        
        dropout.eval()
        output = dropout(x)
        
        # In eval mode, should return input unchanged
        assert np.array_equal(output.data, x.data)
        
    def test_inplace_2d(self):
        """Test inplace Dropout2d operation"""
        dropout = Dropout2d(p=0.5, inplace=True)
        x = Tensor(np.ones((2, 3, 4, 4)))
        original_data = x.data.copy()
        
        dropout.train()
        output = dropout(x)
        
        # Output should be same object as input
        assert output is x
        # Data should be modified
        assert not np.array_equal(x.data, original_data)