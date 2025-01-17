import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.ops.loss import (
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    L1Loss,
    HuberLoss,
    KLDivLoss,
    CosineSimilarityLoss,
    HingeLoss,
    FocalLoss
)

class TestMSELoss:
    """Tests for Mean Squared Error Loss"""
    
    def test_forward(self):
        """Test forward pass of MSE loss"""
        predictions = Tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = Tensor([[2.0, 1.0], [4.0, 3.0]])
        
        # Test mean reduction
        loss = MSELoss.apply(predictions, targets, 'mean')
        expected = np.mean((predictions.data - targets.data) ** 2)
        assert np.allclose(loss.data, expected)
        
        # Test sum reduction
        loss = MSELoss.apply(predictions, targets, 'sum')
        expected = np.sum((predictions.data - targets.data) ** 2)
        assert np.allclose(loss.data, expected)
        
    def test_backward(self):
        """Test backward pass of MSE loss"""
        predictions = Tensor([[1.0]], requires_grad=True)
        targets = Tensor([[2.0]])
        
        loss = MSELoss.apply(predictions, targets, 'mean')
        loss.backward()
        
        # For MSE, gradient should be 2(pred - target)/N
        expected_grad = 2 * (predictions.data - targets.data) / np.prod(predictions.shape)
        assert np.allclose(predictions.grad, expected_grad)

class TestCrossEntropyLoss:
    """Tests for Cross Entropy Loss"""
    
    def test_forward(self):
        """Test forward pass of cross entropy loss"""
        predictions = Tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = Tensor([[1.0, 0.0], [0.0, 1.0]])  # One-hot encoded
        
        loss = CrossEntropyLoss.apply(predictions, targets)
        assert loss.data >= 0  # Loss should be non-negative
        
    def test_numerical_stability(self):
        """Test numerical stability with large inputs"""
        predictions = Tensor([[1000., -1000.], [-1000., 1000.]])
        targets = Tensor([[1., 0.], [0., 1.]])
        
        loss = CrossEntropyLoss.apply(predictions, targets)
        assert not np.isnan(loss.data)
        assert not np.isinf(loss.data)
        
    def test_gradient(self):
        """Test gradient computation"""
        predictions = Tensor([[1.0, 0.0]], requires_grad=True)
        targets = Tensor([[1.0, 0.0]])
        
        loss = CrossEntropyLoss.apply(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None
        assert not np.isnan(predictions.grad).any()
        assert not np.isinf(predictions.grad).any()

class TestBinaryCrossEntropyLoss:
    """Tests for Binary Cross Entropy Loss"""
    
    def test_forward(self):
        """Test forward pass of binary cross entropy loss"""
        predictions = Tensor([0.7, 0.3])
        targets = Tensor([1.0, 0.0])
        
        loss = BinaryCrossEntropyLoss.apply(predictions, targets)
        assert loss.data >= 0  # Loss should be non-negative
        
    def test_gradient(self):
        """Test gradient computation"""
        predictions = Tensor([0.7], requires_grad=True)
        targets = Tensor([1.0])
        
        loss = BinaryCrossEntropyLoss.apply(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None
        assert not np.isnan(predictions.grad).any()
        
    def test_reductions(self):
        """Test different reduction methods"""
        predictions = Tensor([[0.7, 0.3], [0.2, 0.8]])
        targets = Tensor([[1.0, 0.0], [0.0, 1.0]])
        
        loss_none = BinaryCrossEntropyLoss.apply(predictions, targets, 'none')
        loss_mean = BinaryCrossEntropyLoss.apply(predictions, targets, 'mean')
        loss_sum = BinaryCrossEntropyLoss.apply(predictions, targets, 'sum')
        
        assert loss_none.shape == predictions.shape
        # Check if scalar by ensuring it's a 0-dimensional array or float
        assert loss_mean.data.ndim == 0 or isinstance(loss_mean.data, float)
        assert loss_sum.data.ndim == 0 or isinstance(loss_sum.data, float)

class TestL1Loss:
    """Tests for L1 Loss"""
    
    def test_forward(self):
        """Test forward pass of L1 loss"""
        predictions = Tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = Tensor([[2.0, 1.0], [4.0, 3.0]])
        
        loss = L1Loss.apply(predictions, targets)
        expected = np.mean(np.abs(predictions.data - targets.data))
        assert np.allclose(loss.data, expected)
        
    def test_backward(self):
        """Test backward pass of L1 loss"""
        predictions = Tensor([1.0], requires_grad=True)
        targets = Tensor([2.0])
        
        loss = L1Loss.apply(predictions, targets)
        loss.backward()
        
        # Gradient should be sign(pred - target)
        expected_grad = np.sign(predictions.data - targets.data)
        assert np.allclose(predictions.grad, expected_grad)

class TestHuberLoss:
    """Tests for Huber Loss"""
    
    def test_forward(self):
        """Test forward pass of Huber loss"""
        predictions = Tensor([1.0, 2.0])
        targets = Tensor([0.0, 4.0])
        delta = 1.0
        
        loss = HuberLoss.apply(predictions, targets, delta)
        
        # Manually calculate expected loss
        diff = predictions.data - targets.data
        expected = np.mean(np.where(np.abs(diff) <= delta,
                                  0.5 * diff ** 2,
                                  delta * np.abs(diff) - 0.5 * delta ** 2))
        
        assert np.allclose(loss.data, expected)
        
    def test_backward(self):
        """Test backward pass of Huber loss"""
        predictions = Tensor([0.0], requires_grad=True)
        targets = Tensor([2.0])
        delta = 1.0
        
        loss = HuberLoss.apply(predictions, targets, delta)
        loss.backward()
        
        assert predictions.grad is not None
        assert not np.isnan(predictions.grad).any()

class TestKLDivLoss:
    """Tests for KL Divergence Loss"""
    
    def test_forward(self):
        """Test forward pass of KL divergence loss"""
        predictions = Tensor([[0.5, 0.5]])
        targets = Tensor([[0.8, 0.2]])
        
        loss = KLDivLoss.apply(predictions, targets)
        assert loss.data >= 0  # KL divergence is always non-negative
        
    def test_numerical_stability(self):
        """Test numerical stability with small probabilities"""
        predictions = Tensor([[0.999, 0.001]])
        targets = Tensor([[0.001, 0.999]])
        
        loss = KLDivLoss.apply(predictions, targets)
        assert not np.isnan(loss.data)
        assert not np.isinf(loss.data)

class TestCosineSimilarityLoss:
    """Tests for Cosine Similarity Loss"""
    
    def test_forward(self):
        """Test forward pass of cosine similarity loss"""
        x1 = Tensor([[1.0, 0.0]])
        x2 = Tensor([[0.0, 1.0]])
        
        loss = CosineSimilarityLoss.apply(x1, x2)
        # Orthogonal vectors should have cos_sim = 0, so loss = 1 - 0 = 1
        assert np.allclose(loss.data, 1.0), f"Expected 1.0, got {loss.data}"
        
    def test_identical_vectors(self):
        """Test with identical vectors"""
        x = Tensor([[1.0, 1.0]])
        loss = CosineSimilarityLoss.apply(x, x)
        # For identical vectors, cosine similarity is 1, so loss = 1 - 1 = 0
        assert np.allclose(loss.data, 0.0, atol=1e-7)
        
    def test_numerical_stability(self):
        """Test numerical stability with zero vectors"""
        x1 = Tensor([[0.0, 0.0]])
        x2 = Tensor([[1.0, 1.0]])
        
        loss = CosineSimilarityLoss.apply(x1, x2)
        assert not np.isnan(loss.data)

class TestHingeLoss:
    """Tests for Hinge Loss"""
    
    def test_forward(self):
        """Test forward pass of hinge loss"""
        predictions = Tensor([0.5, -0.5])
        targets = Tensor([1.0, 0.0])
        
        loss = HingeLoss.apply(predictions, targets)
        assert loss.data >= 0  # Hinge loss is non-negative
        
    def test_perfect_prediction(self):
        """Test with perfect predictions"""
        predictions = Tensor([1.0])
        targets = Tensor([1.0])
        
        loss = HingeLoss.apply(predictions, targets)
        assert np.allclose(loss.data, 0.0)  # Loss should be zero
        
    def test_margin(self):
        """Test different margin values"""
        predictions = Tensor([0.5])
        targets = Tensor([1.0])
        
        loss1 = HingeLoss.apply(predictions, targets, margin=1.0)
        loss2 = HingeLoss.apply(predictions, targets, margin=2.0)
        assert loss2.data > loss1.data  # Larger margin should give larger loss

class TestFocalLoss:
    """Tests for Focal Loss"""
    
    def test_forward(self):
        """Test forward pass of focal loss"""
        predictions = Tensor([0.7, 0.3])
        targets = Tensor([1.0, 0.0])
        
        loss = FocalLoss.apply(predictions, targets)
        assert loss.data >= 0  # Focal loss is non-negative
        
    def test_gamma_effect(self):
        """Test effect of gamma parameter"""
        predictions = Tensor([0.7])
        targets = Tensor([1.0])
        
        loss1 = FocalLoss.apply(predictions, targets, gamma=0.0)  # Equivalent to BCE
        loss2 = FocalLoss.apply(predictions, targets, gamma=2.0)  # Standard focal loss
        
        # Focal loss should be smaller than BCE for easy examples
        assert loss2.data < loss1.data
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme probabilities"""
        predictions = Tensor([0.999, 0.001])
        targets = Tensor([1.0, 0.0])
        
        loss = FocalLoss.apply(predictions, targets)
        assert not np.isnan(loss.data)
        assert not np.isinf(loss.data)

class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_shape_mismatch(self):
        """Test shape mismatch handling"""
        predictions = Tensor([[1.0, 2.0]])
        targets = Tensor([1.0])
        
        with pytest.raises(ValueError):
            MSELoss.apply(predictions, targets)
            
    def test_invalid_reduction(self):
        """Test invalid reduction method"""
        predictions = Tensor([1.0])
        targets = Tensor([1.0])
        
        with pytest.raises(ValueError):
            MSELoss.apply(predictions, targets, reduction='invalid')
            
    def test_negative_probabilities(self):
        """Test handling of negative probabilities"""
        predictions = Tensor([-0.1, 1.1])
        targets = Tensor([0.0, 1.0])
        
        with pytest.raises(ValueError):
            BinaryCrossEntropyLoss.apply(predictions, targets)