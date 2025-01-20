# test_data.py
import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.data import (
    Dataset, TensorDataset, DataLoader,
    SequentialSampler, RandomSampler
)

class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, size: int = 10):
        self.size = size
        self.data = list(range(size))
        
    def __getitem__(self, index: int):
        return self.data[index]
        
    def __len__(self):
        return self.size

class TestDataset:
    """Tests for Dataset base class and implementations."""
    
    def test_tensor_dataset(self):
        """Test TensorDataset functionality."""
        # Test creation with matching sizes
        x = Tensor([[1, 2], [3, 4], [5, 6]])
        y = Tensor([1, 2, 3])
        dataset = TensorDataset(x, y)
        
        assert len(dataset) == 3
        sample = dataset[0]
        assert len(sample) == 2
        assert np.array_equal(sample[0].data, [1, 2])
        assert np.array_equal(sample[1].data, 1)
        
        # Test creation with mismatched sizes
        x = Tensor([[1, 2], [3, 4]])
        y = Tensor([1, 2, 3])
        with pytest.raises(AssertionError):
            TensorDataset(x, y)
            
    def test_dataset_interface(self):
        """Test Dataset abstract base class."""
        # Should raise NotImplementedError when not implemented
        dataset = Dataset()
        with pytest.raises(NotImplementedError):
            len(dataset)
        with pytest.raises(NotImplementedError):
            dataset[0]

class TestDataLoader:
    """Tests for DataLoader functionality."""
    
    def test_sequential_loading(self):
        """Test sequential data loading."""
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        # Check number of batches
        assert len(loader) == 4  # 3 full batches + 1 partial
        
        # Check batch contents
        batches = list(loader)
        assert len(batches) == 4
        assert np.array_equal(batches[0].data, [0, 1, 2])
        assert len(batches[-1]) == 1  # Last batch should have 1 element
        
    def test_shuffle_loading(self):
        """Test shuffled data loading."""
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3, shuffle=True)
        
        # Get two epochs of data
        epoch1 = [item.data.tolist() for item in loader]
        epoch2 = [item.data.tolist() for item in loader]
        
        # Epochs should be different (note: there's a tiny chance they're the same)
        assert epoch1 != epoch2
        
        # All elements should still be present in each epoch
        assert sorted([x for batch in epoch1 for x in batch]) == list(range(10))
        
    def test_drop_last(self):
        """Test drop_last functionality."""
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3, drop_last=True)
        
        batches = list(loader)
        assert len(batches) == 3  # Should drop the last incomplete batch
        assert all(len(batch) == 3 for batch in batches)
        
    def test_custom_sampler(self):
        """Test custom sampling."""
        dataset = SimpleDataset(10)
        sampler = SequentialSampler(dataset)
        
        # Test that sampler and shuffle are mutually exclusive
        with pytest.raises(ValueError):
            DataLoader(dataset, sampler=sampler, shuffle=True)
            
        # Test custom sampler works
        loader = DataLoader(dataset, sampler=sampler, batch_size=2)
        batches = list(loader)
        assert len(batches) == 5
        assert np.array_equal(batches[0].data, [0, 1])

class TestSamplers:
    """Tests for sampler implementations."""
    
    def test_sequential_sampler(self):
        """Test SequentialSampler."""
        dataset = SimpleDataset(5)
        sampler = SequentialSampler(dataset)
        
        indices = list(sampler)
        assert indices == [0, 1, 2, 3, 4]
        assert len(sampler) == 5
        
    def test_random_sampler(self):
        """Test RandomSampler."""
        dataset = SimpleDataset(5)
        sampler = RandomSampler(dataset)
        
        # Check length
        assert len(sampler) == 5
        
        # Check randomization
        indices1 = list(sampler)
        indices2 = list(sampler)
        
        # Should contain all indices
        assert sorted(indices1) == list(range(5))
        # Should be in different order (note: tiny chance they're the same)
        assert indices1 != indices2

class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        dataset = SimpleDataset(0)
        loader = DataLoader(dataset, batch_size=2)
        
        assert len(loader) == 0
        assert list(loader) == []
        
    def test_single_element_dataset(self):
        """Test datasets with a single element."""
        dataset = SimpleDataset(1)
        loader = DataLoader(dataset, batch_size=2)  # Batch size larger than dataset
        
        batches = list(loader)
        assert len(batches) == 1
        assert np.array_equal(batches[0].data, [0])
        
    def test_batch_size_one(self):
        """Test batch size of 1."""
        dataset = SimpleDataset(3)
        loader = DataLoader(dataset, batch_size=1)
        
        batches = list(loader)
        assert len(batches) == 3
        assert all(len(batch) == 1 for batch in batches)
        
    def test_custom_collate(self):
        """Test custom collate function."""
        def custom_collate(batch):
            # Convert Tensors to values before summing
            return sum(item.data[0] if isinstance(item, Tensor) else item for item in batch)
        
        dataset = SimpleDataset(4)
        loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)
        
        batches = list(loader)
        assert batches[0] == 1  # 0 + 1
        assert batches[1] == 5  # 2 + 3

    def test_large_batch_size(self):
        """Test batch size larger than dataset."""
        dataset = SimpleDataset(3)
        loader = DataLoader(dataset, batch_size=5)
        
        batches = list(loader)
        assert len(batches) == 1
        assert np.array_equal(batches[0].data, [0, 1, 2])