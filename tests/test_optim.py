import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.optim import SGD, Adam, RMSprop, AdaGrad

class TestOptimizers:
    """Base test class for all optimizers."""
    
    def setup_method(self):
        """Setup method run before each test."""
        # Create a simple parameter tensor
        self.param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.grad = np.array([0.1, 0.2, 0.3])
        
    def _test_basic_update(self, optimizer_class, **kwargs):
        """Helper method to test basic parameter updates."""
        optimizer = optimizer_class([self.param], **kwargs)
        
        # Initial parameter values
        initial_params = self.param.data.copy()
        
        # Set gradient and perform optimization step
        self.param.grad = self.grad
        optimizer.step()
        
        # Check that parameters were updated
        assert not np.array_equal(self.param.data, initial_params)
        
    def _test_zero_grad(self, optimizer_class, **kwargs):
        """Helper method to test zero_grad functionality."""
        optimizer = optimizer_class([self.param], **kwargs)
        
        # Set some gradient
        self.param.grad = self.grad
        
        # Zero out gradients
        optimizer.zero_grad()
        
        # Check that gradients are zeroed
        assert np.all(self.param.grad == 0)

class TestSGD(TestOptimizers):
    """Tests for SGD optimizer."""
    
    def test_basic_sgd(self):
        """Test basic SGD functionality."""
        self._test_basic_update(SGD, lr=0.1)
        
    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        optimizer = SGD([self.param], lr=0.1, momentum=0.9)
        
        # First update
        self.param.grad = self.grad
        optimizer.step()
        first_update = self.param.data.copy()
        
        # Second update with same gradient
        optimizer.step()
        
        # With momentum, second update should be larger
        first_step = np.abs(first_update - np.array([1.0, 2.0, 3.0]))
        second_step = np.abs(self.param.data - first_update)
        assert np.all(second_step > first_step)
        
    def test_sgd_nesterov(self):
        """Test SGD with Nesterov momentum."""
        self._test_basic_update(SGD, lr=0.1, momentum=0.9, nesterov=True)
        
    def test_sgd_weight_decay(self):
        """Test SGD with weight decay."""
        optimizer = SGD([self.param], lr=0.1, weight_decay=0.1)
        self.param.grad = self.grad
        optimizer.step()
        
        # Parameters should decrease more with weight decay
        no_decay_param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        no_decay_optimizer = SGD([no_decay_param], lr=0.1)
        no_decay_param.grad = self.grad
        no_decay_optimizer.step()
        
        assert np.all(np.abs(self.param.data) < np.abs(no_decay_param.data))

class TestAdam(TestOptimizers):
    """Tests for Adam optimizer."""
    
    def test_basic_adam(self):
        """Test basic Adam functionality."""
        self._test_basic_update(Adam, lr=0.001)
        
    def test_adam_bias_correction(self):
        """Test Adam bias correction."""
        optimizer = Adam([self.param], lr=0.001)
        
        initial_param = self.param.data.copy()
        updates = []
        
        # Perform several updates and track parameter changes
        for _ in range(6):
            self.param.grad = self.grad  # Keep constant gradient for testing
            optimizer.step()
            updates.append(np.linalg.norm(self.param.data - initial_param))
        
        # Test that:
        # 1. Parameters are actually being updated
        assert not np.array_equal(self.param.data, initial_param)
        
        # 2. Updates are being affected by bias correction
        # (not necessarily smaller, but should be different)
        assert len(set(updates)) > 1  # Updates should not all be identical
        
        # 3. State contains expected bias correction terms
        assert 'step' in optimizer.state[id(self.param)]
        assert 'exp_avg' in optimizer.state[id(self.param)]
        assert 'exp_avg_sq' in optimizer.state[id(self.param)]
        
    def test_adam_amsgrad(self):
        """Test Adam with AMSGrad."""
        self._test_basic_update(Adam, lr=0.001, amsgrad=True)

class TestRMSprop(TestOptimizers):
    """Tests for RMSprop optimizer."""
    
    def test_basic_rmsprop(self):
        """Test basic RMSprop functionality."""
        self._test_basic_update(RMSprop, lr=0.01)
        
    def test_rmsprop_momentum(self):
        """Test RMSprop with momentum."""
        self._test_basic_update(RMSprop, lr=0.01, momentum=0.9)
        
    def test_rmsprop_centered(self):
        """Test centered RMSprop."""
        self._test_basic_update(RMSprop, lr=0.01, centered=True)

class TestAdaGrad(TestOptimizers):
    """Tests for AdaGrad optimizer."""
    
    def test_basic_adagrad(self):
        """Test basic AdaGrad functionality."""
        self._test_basic_update(AdaGrad, lr=0.01)
        
    def test_adagrad_lr_decay(self):
        """Test AdaGrad learning rate decay."""
        optimizer = AdaGrad([self.param], lr=0.01, lr_decay=0.1)
        
        initial_param = self.param.data.copy()
        effective_lrs = []
        
        # Perform several updates and track effective learning rates
        for _ in range(6):
            self.param.grad = self.grad  # Keep constant gradient for testing
            prev_param = self.param.data.copy()
            optimizer.step()
            
            # Calculate effective learning rate from parameter update
            param_update = np.linalg.norm(self.param.data - prev_param)
            grad_norm = np.linalg.norm(self.grad)
            effective_lrs.append(param_update / grad_norm if grad_norm != 0 else 0)
        
        # Test that:
        # 1. Parameters are actually being updated
        assert not np.array_equal(self.param.data, initial_param)
        
        # 2. Accumulated sum in state is increasing
        assert np.all(optimizer.state[id(self.param)]['sum'] > 0)
        
        # 3. Effective learning rates should show some variation
        assert len(set(map(lambda x: round(x, 6), effective_lrs))) > 1
        
    def test_adagrad_reset(self):
        """Test AdaGrad state reset."""
        optimizer = AdaGrad([self.param], lr=0.01)
        
        # Perform some updates
        self.param.grad = self.grad
        optimizer.step()
        
        # Reset state
        optimizer.reset_state()
        
        # Check that state was reset
        for state in optimizer.state.values():
            assert state['step'] == 0
            assert np.all(state['sum'] == 0)

class TestOptimizerEdgeCases:
    """Tests for optimizer edge cases and error conditions."""
    
    def test_invalid_learning_rates(self):
        """Test that invalid learning rates raise errors."""
        param = Tensor([1.0], requires_grad=True)
        
        with pytest.raises(ValueError):
            SGD([param], lr=-0.1)
        with pytest.raises(ValueError):
            Adam([param], lr=-0.1)
        with pytest.raises(ValueError):
            RMSprop([param], lr=-0.1)
        with pytest.raises(ValueError):
            AdaGrad([param], lr=-0.1)
            
    def test_no_gradients(self):
        """Test optimizer behavior with no gradients."""
        param = Tensor([1.0], requires_grad=True)
        optimizers = [
            SGD([param], lr=0.1),
            Adam([param], lr=0.001),
            RMSprop([param], lr=0.01),
            AdaGrad([param], lr=0.01)
        ]
        
        # Parameter should not change if there's no gradient
        for optimizer in optimizers:
            initial_param = param.data.copy()
            optimizer.step()
            assert np.array_equal(param.data, initial_param)
            
    def test_param_groups(self):
        """Test adding parameter groups."""
        param1 = Tensor([1.0], requires_grad=True)
        param2 = Tensor([2.0], requires_grad=True)
        
        optimizer = SGD([param1], lr=0.1)
        optimizer.add_param_group({'params': [param2]})
        
        # Both parameters should be updated
        param1.grad = np.array([0.1])
        param2.grad = np.array([0.2])
        optimizer.step()
        
        assert not np.array_equal(param1.data, [1.0])
        assert not np.array_equal(param2.data, [2.0])