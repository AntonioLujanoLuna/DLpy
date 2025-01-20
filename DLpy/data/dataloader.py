# dataloader.py
from typing import Iterator, Optional, Sequence, Any, Callable
import numpy as np
from .samplers import SequentialSampler, RandomSampler, Sampler
from .dataset import Dataset
from DLpy.core.tensor import Tensor

class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    
    Args:
        dataset: Dataset from which to load the data
        batch_size: How many samples per batch to load
        shuffle: Set to True to have the data reshuffled at every epoch
        sampler: Defines the strategy to draw samples from the dataset
        drop_last: If True, drop the last incomplete batch
        collate_fn: Merges a list of samples to form a mini-batch
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else self._default_collate
        
        if sampler is not None:
            if shuffle:
                raise ValueError("Cannot specify both shuffle and sampler")
            self.sampler = sampler
        else:
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)
                
    def _default_collate(self, batch: Sequence[Any]) -> Any:
        """Default collate function for batching samples."""
        elem = batch[0]
        
        if isinstance(elem, Tensor):
            return Tensor(np.stack([b.data for b in batch]))
        elif isinstance(elem, (int, float)):
            # Convert scalar values to a flat tensor
            return Tensor(np.array(batch))
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return type(elem)(*(self._default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, (list, tuple)):
            return [self._default_collate(samples) for samples in zip(*batch)]
        else:
            try:
                return Tensor(np.array(batch))
            except:
                return batch
            
    def __iter__(self) -> Iterator[Any]:
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)
            
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size