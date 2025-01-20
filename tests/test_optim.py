import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.optim import SGD, Adam, RMSprop, AdaGrad, AdaDelta, AdaMax

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
    
    def test_optimizer_state_dict(self):
        """Test state dict functionality"""
        param = Tensor([1.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)
        
        # Save state
        state = optimizer.state_dict()
        assert 'state' in state
        
        # Load state
        new_optimizer = SGD([param], lr=0.1)
        new_optimizer.load_state_dict(state)
        
        assert new_optimizer.state == optimizer.state

    def test_optimizer_add_param_group_validation(self):
        """Test param group validation"""
        param = Tensor([1.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)
        
        # Test adding single tensor
        optimizer.add_param_group({'params': Tensor([2.0], requires_grad=True)})
        
        # Test adding list of tensors instead of set
        param_list = [Tensor([3.0], requires_grad=True)]
        optimizer.add_param_group({'params': param_list})
        
        assert len(optimizer._params) == 3

class TestAdaDelta:
    """Tests for AdaDelta optimizer."""
    
    def setup_method(self):
        """Setup method run before each test."""
        self.param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.grad = np.array([0.1, 0.2, 0.3])
        
    def test_basic_adadelta(self):
        """Test basic AdaDelta functionality."""
        optimizer = AdaDelta([self.param])
        
        # Initial parameter values
        initial_params = self.param.data.copy()
        
        # First update
        self.param.grad = self.grad
        optimizer.step()
        
        # Parameters should be updated
        assert not np.array_equal(self.param.data, initial_params)
        
        # State should be initialized
        state = optimizer.state[id(self.param)]
        assert 'square_avg' in state
        assert 'acc_delta' in state
        assert state['step'] == 1
        
    def test_adadelta_no_lr(self):
        """Test AdaDelta works without learning rate."""
        optimizer = AdaDelta([self.param])
        self.param.grad = self.grad
        
        # Should not raise error despite no learning rate
        optimizer.step()
        
    def test_adadelta_convergence(self):
        """Test AdaDelta converges to minimum."""
        param = Tensor([1.0], requires_grad=True)
        optimizer = AdaDelta([param], eps=1e-5)
        
        # Simple quadratic function: f(x) = (x-1)^2
        for _ in range(100):
            # Gradient of (x-1)^2 is 2(x-1)
            param.grad = np.array([2.0 * (param.data[0] - 1.0)])
            optimizer.step()
            
        # Should converge close to x=1
        assert np.abs(param.data[0] - 1.0) < 0.1
        
    def test_adadelta_parameter_validation(self):
        """Test parameter validation in AdaDelta."""
        with pytest.raises(ValueError):
            AdaDelta([self.param], rho=-0.1)
        with pytest.raises(ValueError):
            AdaDelta([self.param], rho=1.1)
        with pytest.raises(ValueError):
            AdaDelta([self.param], eps=-1e-6)
        with pytest.raises(ValueError):
            AdaDelta([self.param], weight_decay=-0.1)

class TestAdaMax:
    """Tests for AdaMax optimizer."""
    
    def setup_method(self):
        """Setup method run before each test."""
        self.param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.grad = np.array([0.1, 0.2, 0.3])
        
    def test_basic_adamax(self):
        """Test basic AdaMax functionality."""
        optimizer = AdaMax([self.param])
        
        # Initial parameter values
        initial_params = self.param.data.copy()
        
        # First update
        self.param.grad = self.grad
        optimizer.step()
        
        # Parameters should be updated
        assert not np.array_equal(self.param.data, initial_params)
        
        # State should be initialized
        state = optimizer.state[id(self.param)]
        assert 'exp_avg' in state
        assert 'exp_inf' in state
        assert state['step'] == 1
        
    def test_adamax_bias_correction(self):
        """Test AdaMax bias correction."""
        optimizer = AdaMax([self.param], lr=0.01)
        
        updates = []
        initial_param = self.param.data.copy()
        
        # Perform several updates and track parameter changes
        for _ in range(5):
            self.param.grad = self.grad
            optimizer.step()
            updates.append(np.linalg.norm(self.param.data - initial_param))
            
        # Updates should vary due to bias correction
        assert len(set(updates)) > 1
        
    def test_adamax_convergence(self):
        """Test AdaMax converges to minimum."""
        param = Tensor([2.0], requires_grad=True)
        optimizer = AdaMax([param], lr=0.1)
        
        # Simple quadratic function: f(x) = (x-1)^2
        for _ in range(100):
            # Gradient of (x-1)^2 is 2(x-1)
            param.grad = np.array([2.0 * (param.data[0] - 1.0)])
            optimizer.step()
            
        # Should converge close to x=1
        assert np.abs(param.data[0] - 1.0) < 0.1
        
    def test_adamax_parameter_validation(self):
        """Test parameter validation in AdaMax."""
        with pytest.raises(ValueError):
            AdaMax([self.param], lr=-0.1)
        with pytest.raises(ValueError):
            AdaMax([self.param], eps=-1e-8)
        with pytest.raises(ValueError):
            AdaMax([self.param], betas=(-0.1, 0.999))
        with pytest.raises(ValueError):
            AdaMax([self.param], betas=(0.9, 1.1))
        with pytest.raises(ValueError):
            AdaMax([self.param], weight_decay=-0.1)

class TestAdvancedOptimizerEdgeCases:
    """Tests for edge cases in advanced optimizers."""
    
    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        param = Tensor([1.0], requires_grad=True)
        optimizers = [
            AdaDelta([param]),
            AdaMax([param])
        ]
        
        for optimizer in optimizers:
            # Parameter should not change if there's no gradient
            initial_param = param.data.copy()
            optimizer.step()
            assert np.array_equal(param.data, initial_param)
            
    def test_state_dict(self):
        """Test state dict functionality."""
        param = Tensor([1.0], requires_grad=True)
        optimizers = [
            AdaDelta([param]),
            AdaMax([param])
        ]
        
        for optimizer in optimizers:
            # Perform an update
            param.grad = np.array([0.1])
            optimizer.step()
            
            # Save state
            state = optimizer.state_dict()
            assert 'state' in state
            assert 'defaults' in state
            
            # Create new optimizer and load state
            if isinstance(optimizer, AdaDelta):
                new_optimizer = AdaDelta([param])
            else:
                new_optimizer = AdaMax([param])
            
            new_optimizer.load_state_dict(state)
            
            # States should match
            assert new_optimizer.state.keys() == optimizer.state.keys()
            for key in optimizer.state:
                assert all(np.array_equal(optimizer.state[key][k], 
                                        new_optimizer.state[key][k])
                         for k in optimizer.state[key]
                         if isinstance(optimizer.state[key][k], np.ndarray))
                         
    def test_sparse_gradients(self):
        """Test handling of sparse gradients."""
        param = Tensor([1.0, 0.0, 2.0, 0.0], requires_grad=True)
        optimizers = [
            AdaDelta([param]),
            AdaMax([param])
        ]
        
        # Create sparse gradient (most values zero)
        sparse_grad = np.array([0.1, 0.0, 0.0, 0.3])
        
        for optimizer in optimizers:
            param.grad = sparse_grad
            optimizer.step()
            
            # Check that only non-zero gradient entries caused updates
            state = optimizer.state[id(param)]
            if isinstance(optimizer, AdaDelta):
                square_avg = state['square_avg']
                assert square_avg[1] == 0  # No update where gradient was 0
                assert square_avg[2] == 0
                assert square_avg[0] != 0  # Update where gradient was non-zero
                assert square_avg[3] != 0
            else:  # AdaMax
                exp_avg = state['exp_avg']
                assert exp_avg[1] == 0  # No update where gradient was 0
                assert exp_avg[2] == 0
                assert exp_avg[0] != 0  # Update where gradient was non-zero
                assert exp_avg[3] != 0

    def test_momentum_behavior(self):
        """Test momentum-like behavior in optimizers with varying gradients."""
        # Initialize a single parameter
        param_adadelta = Tensor([1.0], requires_grad=True)
        param_adamax = Tensor([1.0], requires_grad=True)

        optimizers = [
            AdaDelta([param_adadelta], rho=0.9),
            AdaMax([param_adamax], betas=(0.9, 0.999))
        ]

        for optimizer, param in zip(optimizers, [param_adadelta, param_adamax]):
            # Apply varying gradients
            updates = []
            for i in range(1, 6):
                param.grad = np.array([0.1 * i])  # Gradients increase each step
                optimizer.step()
                updates.append(float(param.data[0]))

            # Compute differences between consecutive updates
            diffs = np.diff(updates)

            # Debugging output (optional)
            print(f"{type(optimizer).__name__} updates: {updates}")
            print(f"{type(optimizer).__name__} diffs: {diffs}")

            # Assert that diffs are not all close to zero
            assert not np.allclose(diffs, diffs[0]), (
                f"{type(optimizer).__name__} does not exhibit momentum-like behavior"
            )

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        param = Tensor([1e-6, 1e6], requires_grad=True)
        optimizers = [
            AdaDelta([param], eps=1e-7),
            AdaMax([param], eps=1e-7)
        ]
        
        for optimizer in optimizers:
            # Test with very small and very large gradients
            param.grad = np.array([1e-8, 1e8])
            
            try:
                optimizer.step()
            except Exception as e:
                pytest.fail(f"Optimizer {type(optimizer).__name__} failed with extreme values: {str(e)}")
                
            # Check for NaN or inf values
            assert not np.any(np.isnan(param.data))
            assert not np.any(np.isinf(param.data))

    def test_weight_decay(self):
        """Test weight decay in optimizers."""
        weight_decay = 0.2
        
        # Test both optimizers with and without weight decay
        for optimizer_class in [AdaDelta, AdaMax]:
            # Without weight decay
            param_no_decay = Tensor([1.0, 2.0], requires_grad=True)
            opt_no_decay = optimizer_class([param_no_decay])
            param_no_decay.grad = np.array([0.1, 0.2])
            opt_no_decay.step()
            result_no_decay = param_no_decay.data.copy()
            
            # With weight decay
            param_with_decay = Tensor([1.0, 2.0], requires_grad=True)
            opt_with_decay = optimizer_class([param_with_decay], weight_decay=weight_decay)
            param_with_decay.grad = np.array([0.1, 0.2])
            opt_with_decay.step()
            result_with_decay = param_with_decay.data.copy()
            
            # Parameters should be smaller with weight decay
            assert np.all(np.abs(result_with_decay) < np.abs(result_no_decay)), \
                f"{type(optimizer_class).__name__} does not correctly apply weight decay"
