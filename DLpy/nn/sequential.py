# DLpy/nn/sequential.py

from typing import List, Union, Iterator
from ..core import Module, Tensor
from collections import OrderedDict

class Sequential(Module):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can be passed in.
    
    Example usage:
    >>> net = Sequential(
    >>>     Linear(10, 5),
    >>>     ReLU(),
    >>>     Linear(5, 1)
    >>> )
    """
    
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
                
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all modules in sequence."""
        for module in self._modules.values():
            x = module(x)
        return x
        
    def __getitem__(self, idx: Union[slice, int]) -> Union['Sequential', Module]:
        """Get a module or slice of modules."""
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return list(self._modules.values())[idx]
            
    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._modules)
        
    def __iter__(self) -> Iterator[Module]:
        """Iterate over modules."""
        return iter(self._modules.values())
        
    def append(self, module: Module) -> None:
        """Add a module to the end."""
        self.add_module(str(len(self)), module)
        
    def insert(self, index: int, module: Module) -> None:
        """Insert a module at specified index."""
        # Shift existing modules
        for i in range(len(self) - 1, index - 1, -1):
            if str(i) in self._modules:
                self._modules[str(i + 1)] = self._modules[str(i)]
                del self._modules[str(i)]
                
        # Insert new module
        self.add_module(str(index), module)
        
    def extend(self, modules: List[Module]) -> None:
        """Add multiple modules to the end."""
        for module in modules:
            self.append(module)
            
    def pop(self, index: int = -1) -> Module:
        """Remove and return module at index."""
        if index < 0:
            index = len(self) + index
            
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
            
        module = self[index]
        del self._modules[str(index)]
        
        # Shift remaining modules
        for i in range(index + 1, len(self) + 1):
            if str(i) in self._modules:
                self._modules[str(i - 1)] = self._modules[str(i)]
                del self._modules[str(i)]
                
        return module