from collections import OrderedDict
from typing import Iterator, List, Union, cast

from ..core import Module, Tensor


class Sequential(Module):
    """
    A sequential container for neural network modules.
    Modules are executed in the order they are passed to the constructor.

    The container can be initialized in two ways:
    1. With individual modules:
            Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    2. With an OrderedDict:
            Sequential(OrderedDict([('fc1', Linear(10, 5)), ('relu', ReLU())]))
    """

    def __init__(
        self,
        first_arg: Union[Module, OrderedDict[str, Module], None] = None,
        *rest: Module,
    ) -> None:
        """
        Initialize the sequential container with modules.

        Args:
            first_arg: Either an OrderedDict mapping names to modules,
                the first module, or None for empty initialization
            *rest: Additional modules when not using OrderedDict initialization
        """
        super().__init__()

        if first_arg is None:
            # Handle empty initialization
            if rest:
                raise TypeError(
                    "When initializing empty Sequential, no additional arguments "
                    "should be provided"
                )
            return

        if isinstance(first_arg, OrderedDict):
            # If we got an OrderedDict, we shouldn't have any additional arguments
            if rest:
                raise TypeError(
                    "When using OrderedDict initialization, no additional arguments "
                    "should be provided"
                )
            # Handle OrderedDict initialization
            for key, module in first_arg.items():
                if not isinstance(module, Module):
                    raise TypeError(
                        f"Value in OrderedDict must be Module, got {type(module)}"
                    )
                self.add_module(key, module)
        else:
            # Handle sequential Module initialization
            # first_arg should be treated as the first module
            if not isinstance(first_arg, Module):
                raise TypeError(f"Expected Module instance, got {type(first_arg)}")
            self.add_module("0", first_arg)

            # Add any remaining modules
            for idx, module in enumerate(rest, start=1):
                if not isinstance(module, Module):
                    raise TypeError(f"Expected Module instance, got {type(module)}")
                self.add_module(str(idx), module)

    def insert(self, index: int, module: Module) -> None:
        """
        Insert a module at a specified index in the sequence.

        Args:
            index: Position at which to insert the module
            module: Module to insert

        Raises:
            TypeError: If index is not an integer
            IndexError: If index is out of valid range
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index <= len(self):
            raise IndexError("Index out of range")

        # Store existing modules and rebuild the sequence
        modules = list(self._modules.values())
        self._modules.clear()

        for i in range(len(modules) + 1):
            if i < index:
                self.add_module(str(i), modules[i])
            elif i == index:
                self.add_module(str(i), module)
            else:
                self.add_module(str(i), modules[i - 1])

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through all modules in sequence.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all modules
        """
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx: Union[slice, int]) -> Union["Sequential", Module]:
        """
        Get a module or slice of modules.

        Args:
            idx: Integer index for single module or slice for multiple modules

        Returns:
            Either a single Module or a new Sequential containing the sliced modules
        """
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return cast(Module, list(self._modules.values())[idx])

    def __len__(self) -> int:
        """Return the number of modules in the sequence."""
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Return an iterator over the modules."""
        return iter(self._modules.values())

    def append(self, module: Module) -> None:
        """
        Add a module to the end of the sequence.

        Args:
            module: Module to append
        """
        self.add_module(str(len(self)), module)

    def extend(self, modules: List[Module]) -> None:
        """
        Add multiple modules to the end of the sequence.

        Args:
            modules: List of modules to append
        """
        for module in modules:
            self.append(module)

    def pop(self, index: int = -1) -> Module:
        """
        Remove and return the module at the specified index.

        Args:
            index: Position of module to remove (default: last module)

        Returns:
            The removed module

        Raises:
            IndexError: If index is out of range
        """
        if index < 0:
            index = len(self) + index

        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        module = self[index]
        del self._modules[str(index)]

        # Shift remaining modules to fill the gap
        for i in range(index + 1, len(self) + 1):
            if str(i) in self._modules:
                self._modules[str(i - 1)] = self._modules[str(i)]
                del self._modules[str(i)]

        return module
