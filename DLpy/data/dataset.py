# dataset.py
from typing import Any, Sized, TypeVar, Tuple
from DLpy.core.tensor import Tensor

T_co = TypeVar('T_co', covariant=True)

class Dataset(Sized):
    """
    Abstract base class for all datasets.
    
    All datasets that represent a map from keys to data samples should subclass it.
    All subclasses must implement __getitem__() and __len__().
    """
    
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError
        
    def __len__(self) -> int:
        raise NotImplementedError
        
class TensorDataset(Dataset):
    """
    Dataset wrapping tensors.
    
    Each sample will be retrieved by indexing tensors along the first dimension.
    
    Args:
        *tensors: Tensors that have the same size of the first dimension.
    """
    
    def __init__(self, *tensors: Tensor):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), \
            "Size mismatch between tensors"
        self.tensors = tensors
        
    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)
        
    def __len__(self) -> int:
        return self.tensors[0].shape[0]