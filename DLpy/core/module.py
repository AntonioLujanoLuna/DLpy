from typing import Iterator, Dict, Any, Optional, Union
from collections import OrderedDict
from ..core import Tensor

class Module:
    """
    Base class for all neural network modules.
    
    Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in
    a tree structure.
    """
    
    def __init__(self):
        """Initialize the module."""
        # First set these directly to avoid triggering __setattr__
        object.__setattr__(self, 'training', True)
        object.__setattr__(self, '_parameters', OrderedDict())
        object.__setattr__(self, '_buffers', OrderedDict())
        object.__setattr__(self, '_modules', OrderedDict())
        
    def register_parameter(self, name: str, param: Optional[Tensor]) -> None:
        """Add a parameter to the module.
        
        Args:
            name: Name of the parameter
            param: The parameter tensor to register
        """
        if '_parameters' not in self.__dict__:
            raise TypeError(
                "cannot assign parameter before Module.__init__() call"
            )
            
        if param is not None and not isinstance(param, Tensor):
            raise TypeError(f"Parameter {name} must be a Tensor, not {type(param)}")
            
        self._parameters[name] = param
        
    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """Add a persistent buffer to the module.
        
        Buffers are typically used for running statistics in modules like BatchNorm.
        
        Args:
            name: Name of the buffer
            tensor: The tensor to register as a buffer
        """
        if '_buffers' not in self.__dict__:
            raise TypeError(
                "cannot assign buffer before Module.__init__() call"
            )
            
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(f"Buffer {name} must be a Tensor, not {type(tensor)}")
            
        self._buffers[name] = tensor
        
    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Add a child module to the current module.
        
        Args:
            name: Name of the child module
            module: The module to add
        """
        if not isinstance(module, (Module, type(None))):
            raise TypeError(f"{name} is not a Module subclass")
            
        if '_modules' not in self.__dict__:
            raise TypeError(
                "cannot assign module before Module.__init__() call"
            )
            
        self._modules[name] = module
        
    def __getattr__(self, name: str) -> Any:
        """Custom getattr that looks through parameters, buffers, and modules."""
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
                
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
                
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
                
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def __setattr__(self, name: str, value: Any) -> None:
        """Custom setattr that handles parameter registration."""
        # Handle special module attributes first
        if name in ['training']:
            object.__setattr__(self, name, value)
            return
            
        if isinstance(value, Tensor):
            if not hasattr(self, '_parameters'):
                raise TypeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            if not hasattr(self, '_modules'):
                raise TypeError(
                    "cannot assign module before Module.__init__() call"
                )
            self.add_module(name, value)
        else:
            object.__setattr__(self, name, value)
            
    def parameters(self) -> Iterator[Tensor]:
        """Returns an iterator over module parameters."""
        for param in self._parameters.values():
            if param is not None:
                yield param
        for module in self._modules.values():
            if module is not None:
                yield from module.parameters()
                
    def named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself."""
        for name, param in self._parameters.items():
            if param is not None:
                yield name, param
        for mname, module in self._modules.items():
            if module is not None:
                for name, param in module.named_parameters():
                    yield f"{mname}.{name}", param
                    
    def train(self, mode: bool = True) -> 'Module':
        """Sets the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self
        
    def eval(self) -> 'Module':
        """Sets the module in evaluation mode."""
        return self.train(False)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        """Define the computation performed at every call."""
        raise NotImplementedError
        
    def __repr__(self):
        """Returns a string representation of the module."""
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
            
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            
        lines = extra_lines + child_lines
        
        main_str = self.__class__.__name__ + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str
        
    def extra_repr(self) -> str:
        """Set the extra representation of the module."""
        return ''

def _addindent(s_: str, numSpaces: int) -> str:
    """Helper for indenting multiline strings."""
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    return '\n'.join([first] + s)