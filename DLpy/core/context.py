from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Context:
    """
    Context class for storing information needed during the backward pass.

    The Context class serves as a storage mechanism for tensors and metadata that are
    needed during backpropagation. It's passed to both forward and backward functions
    to maintain state between the two passes.

    Attributes:
        _saved_tensors: List of tensors saved during forward pass for use in backward
            pass
        _non_tensor_args: Dictionary of additional arguments needed for backward pass
        _intermediate_values: Dictionary storing intermediate computations
    """

    _saved_tensors: List[Any] = field(default_factory=list)
    _non_tensor_args: Dict[str, Any] = field(default_factory=dict)
    _intermediate_values: Dict[str, Any] = field(default_factory=dict)

    def save_for_backward(self, *args: Any) -> None:
        """
        Saves tensors that will be needed for the backward pass.

        Args:
            *args: Variable number of tensors to save
        """
        self._saved_tensors = list(args)

    def save_arguments(self, **kwargs: Any) -> None:
        """
        Saves additional arguments that will be needed for the backward pass.

        Args:
            **kwargs: Keyword arguments to save
        """
        self._non_tensor_args.update(kwargs)

    def store_intermediate(self, name: str, value: Any) -> None:
        """
        Stores intermediate values computed during forward pass that may be
        useful during backward pass or for debugging.

        Args:
            name: Identifier for the intermediate value
            value: The value to store
        """
        self._intermediate_values[name] = value

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors as a tuple."""
        return tuple(self._saved_tensors)

    @property
    def saved_arguments(self) -> Dict[str, Any]:
        """Returns the saved non-tensor arguments."""
        return self._non_tensor_args.copy()

    def get_intermediate(self, name: str) -> Any:
        """
        Retrieves a stored intermediate value.

        Args:
            name: Identifier for the intermediate value

        Returns:
            The stored value

        Raises:
            KeyError: If no value exists for the given name
        """
        return self._intermediate_values[name]

    def clear(self) -> None:
        """Clears all saved data from the context."""
        self._saved_tensors.clear()
        self._non_tensor_args.clear()
        self._intermediate_values.clear()
