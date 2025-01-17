import pytest
from DLpy.nn.modules import Module
from DLpy.core import Tensor
import numpy as np
from DLpy.nn.linear import Linear  

class TestModuleEdgeCases:
    """Tests for edge cases in Module functionality"""
    
    def test_premature_parameter_registration(self):
        """Test parameter registration before initialization"""
        with pytest.raises(TypeError):
            class BadModule(Module):
                def __init__(self):
                    self.param = Tensor([1.0])  # Before super().__init__()
            BadModule()

    def test_invalid_module_addition(self):
        """Test adding invalid modules"""
        module = Module()
        
        # Test adding None module
        module.add_module('none_module', None)
        assert module._modules['none_module'] is None
        
        # Test adding invalid type
        with pytest.raises(TypeError):
            module.add_module('invalid', "not a module")
            
        # Test adding before initialization
        with pytest.raises(TypeError):
            class BadModule(Module):
                def __init__(self):
                    self.add_module('test', Module())  # Before super().__init__()
            BadModule()

    def test_attribute_access(self):
        """Test attribute access edge cases"""
        # Test accessing non-existent attribute
        module = Module()
        with pytest.raises(AttributeError):
            _ = module.nonexistent_attr
        
        # Test accessing training attribute before initialization
        class BadModule(Module):
            def __init__(self):
                # Access training before super().__init__()
                try:
                    _ = self._parameters
                except AttributeError:
                    pass  # Expected
                    
                # Now try to get the training attribute which should fail
                _ = self.training
                super().__init__()
                
        with pytest.raises(AttributeError):
            BadModule()

    def test_module_buffer_operations(self):
        """Test buffer operations in detail"""
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('running_mean', Tensor([0.0]))
                self.register_buffer('running_var', None)
                
        module = TestModule()
        assert 'running_mean' in module._buffers
        assert module._buffers['running_var'] is None
        
        # Test buffer replacement
        module.register_buffer('running_mean', Tensor([1.0]))
        assert np.array_equal(module._buffers['running_mean'].data, [1.0])

    def test_module_state_dict(self):
        """Test state dict functionality"""
        class ComplexModule(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(2, 2)
                self.register_buffer('running_stats', Tensor([0.0]))
                
        module = ComplexModule()
        # Test parameter access
        params = dict(module.named_parameters())
        assert 'linear.weight' in params
        assert 'linear.bias' in params

    def test_nested_module_training(self):
        """Test training mode propagation in nested modules"""
        class NestedModule(Module):
            def __init__(self):
                super().__init__()
                self.sub1 = Linear(2, 2)
                self.sub2 = Linear(2, 2)
                
        module = NestedModule()
        module.train(False)
        assert not module.training
        assert not module.sub1.training
        assert not module.sub2.training