import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from DLpy.core import Module, Tensor
from DLpy.nn import Linear, ReLU
from DLpy.core.serialization import ModelSaver
from DLpy.optim import Adam

class SimpleModel(Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        # Initialize layers
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 1)
        
        # Manual initialization to ensure different weights
        np.random.seed()  # Reset seed for each instance
        bound1 = np.sqrt(2.0 / 10)  # He initialization
        bound2 = np.sqrt(2.0 / 5)  
        self.fc1.weight.data = np.random.uniform(-bound1, bound1, (5, 10))
        self.fc2.weight.data = np.random.uniform(-bound2, bound2, (1, 5))
        self.fc1.bias.data = np.random.uniform(-0.1, 0.1, (5,))
        self.fc2.bias.data = np.random.uniform(-0.1, 0.1, (1,))
        
    def forward(self, x):
        x = self.fc1(x)
        x = np.maximum(0, x.data)  # ReLU
        return self.fc2(Tensor(x))

class TestModelSaver:
    """Tests for ModelSaver functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
            
    def test_save_load_model(self, model, temp_dir):
        """Test saving and loading complete model."""
        save_path = temp_dir / "model.pkl"
        
        # Save model
        ModelSaver.save_model(model, save_path)
        assert save_path.exists()
        
        # Load model
        loaded_model = ModelSaver.load_model(save_path, 
            custom_classes={"SimpleModel": SimpleModel})        
        # Verify model structure
        assert isinstance(loaded_model, SimpleModel)
        assert hasattr(loaded_model, 'fc1')
        assert hasattr(loaded_model, 'fc2')  # Changed from 'relu' to 'fc2'
        
        # Verify parameters
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), 
                                    loaded_model.named_parameters()):
            assert n1 == n2
            assert np.array_equal(p1.data, p2.data)
            
    def test_save_load_state_dict(self, model, temp_dir):
        """Test saving and loading state dictionary."""
        save_path = temp_dir / "state.npz"
        
        # Initialize a second model with different parameters
        model2 = SimpleModel()
        
        # Verify models have different parameters
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert not np.array_equal(p1.data, p2.data)
            
        # Save state dict from first model
        ModelSaver.save_state_dict(model, save_path)
        assert save_path.exists()
        
        # Load state dict into second model
        ModelSaver.load_state_dict(model2, save_path)
        
        # Verify parameters are now the same
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert np.array_equal(p1.data, p2.data)
            
    def test_checkpoint_save_load(self, model, temp_dir):
        """Test saving and loading checkpoints."""
        save_path = temp_dir / "checkpoint.pkl"
        
        # Create optimizer and some training state
        optimizer = Adam(model.parameters())
        epoch = 10
        loss = 0.5
        additional_data = {'learning_rate': 0.001}
        
        # Save checkpoint
        ModelSaver.save_checkpoint(
            save_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            additional_data=additional_data
        )
        assert save_path.exists()
        
        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = Adam(new_model.parameters())
        
        # Load checkpoint
        checkpoint_data = ModelSaver.load_checkpoint(
            save_path,
            model=new_model,
            optimizer=new_optimizer
        )
        
        # Verify model parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert np.array_equal(p1.data, p2.data)
            
        # Verify checkpoint data
        assert checkpoint_data['epoch'] == epoch
        assert checkpoint_data['loss'] == loss
        assert checkpoint_data['additional_data'] == additional_data
        
    def test_custom_model_save_load(self, temp_dir):
        """Test saving and loading model with custom architecture."""
        class CustomModel(Module):
            def __init__(self):
                super().__init__()
                self.custom_param = Tensor([1.0, 2.0], requires_grad=True)
                
        model = CustomModel()
        save_path = temp_dir / "custom_model.pkl"
        
        # Save model
        ModelSaver.save_model(model, save_path)
        
        # Load model with custom class mapping
        loaded_model = ModelSaver.load_model(
            save_path,
            custom_classes={'CustomModel': CustomModel}
        )
        
        assert isinstance(loaded_model, CustomModel)
        assert np.array_equal(loaded_model.custom_param.data, model.custom_param.data)
            
    def test_invalid_paths(self, model):
        """Test error handling for invalid paths."""
        # Non-existent directory
        with pytest.raises(OSError):
            ModelSaver.save_model(model, "/nonexistent/path/model.pkl")
            
        # Invalid file path
        with pytest.raises(OSError):
            ModelSaver.load_model("/nonexistent/model.pkl")
            
    def test_incompatible_state_dict(self, model, temp_dir):
        """Test loading incompatible state dict."""
        save_path = temp_dir / "state.npz"
        
        # Create different model
        different_model = SimpleModel()
        different_model.fc1 = Linear(5, 2)  # Different architecture
        
        # Save state dict from first model
        ModelSaver.save_state_dict(model, save_path)
        
        # Loading should raise error due to incompatible architecture
        with pytest.raises(ValueError):
            ModelSaver.load_state_dict(different_model, save_path)
            
    def test_buffer_saving_loading(self, temp_dir):
        """Test saving and loading model with buffers."""
        class ModelWithBuffer(Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('running_mean', Tensor([0.0, 0.0]))
                self.register_buffer('running_var', Tensor([1.0, 1.0]))
                
        model = ModelWithBuffer()
        save_path = temp_dir / "model_with_buffer.pkl"
        
        # Save model
        ModelSaver.save_model(model, save_path)
        
        # Load model
        loaded_model = ModelSaver.load_model(
            save_path,
            custom_classes={'ModelWithBuffer': ModelWithBuffer}
        )
        
        # Verify buffers
        assert np.array_equal(loaded_model.running_mean.data, model.running_mean.data)
        assert np.array_equal(loaded_model.running_var.data, model.running_var.data)
        
    def test_optimizer_state_save_load(self, model, temp_dir):
        """Test saving and loading optimizer state."""
        save_path = temp_dir / "checkpoint.pkl"
        
        # Create optimizer and perform an update
        optimizer = Adam(model.parameters())
        x = Tensor(np.random.randn(1, 10))
        y = Tensor([[1.0]])
        
        output = model(x)
        loss = ((output - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save checkpoint
        ModelSaver.save_checkpoint(save_path, model, optimizer)
        
        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = Adam(new_model.parameters())
        
        # Load checkpoint
        ModelSaver.load_checkpoint(save_path, new_model, new_optimizer)
        
        # Verify optimizer state
        assert optimizer.state.keys() == new_optimizer.state.keys()
        for key in optimizer.state:
            old_state = optimizer.state[key]
            new_state = new_optimizer.state[key]
            for k in old_state:
                if isinstance(old_state[k], np.ndarray):
                    assert np.array_equal(old_state[k], new_state[k])
                else:
                    assert old_state[k] == new_state[k]