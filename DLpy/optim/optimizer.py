from typing import Any, Dict, Iterator, List, Union, cast

from ..core import Tensor

# Define more precise type aliases
OptState = Dict[
    int, Dict[str, Any]
]  # State dictionary maps parameter IDs to their states
OptDefaults = Dict[str, Any]  # Default hyperparameters dictionary
# StateDict needs to accurately represent our nested structure
StateDict = Dict[
    str, Union[OptState, OptDefaults]
]  # Top level has string keys mapping to either state or defaults


class Optimizer:
    """
    Base class for all optimizers.

    This class provides the basic infrastructure for parameter optimization
    in neural networks. It handles parameter management, gradient zeroing,
    and state tracking across optimization steps.

    Args:
        params: An iterable of parameters to optimize
        defaults: Dictionary of default hyperparameter values for the optimizer
    """

    def __init__(
        self, params: Union[Iterator[Tensor], List[Tensor]], defaults: OptDefaults
    ) -> None:
        self.defaults = defaults
        self._params: List[Tensor] = list(params)
        self.state: OptState = {}

        # Initialize state for each parameter
        for p in self._params:
            self.state[id(p)] = {}

    def zero_grad(self) -> None:
        """
        Clears the gradients of all optimized parameters.

        This should be called before computing gradients for the next step,
        to ensure we don't accumulate gradients from previous steps.
        """
        for p in self._params:
            if p.grad is not None:
                p.grad.fill(0)

    def step(self) -> None:
        """
        Performs a single optimization step.

        This method should be overridden by all optimizers to implement
        their specific parameter update rules.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Add a param group to the optimizer's param groups.

        This method is useful for adding parameters with different
        hyperparameters than the defaults.

        Args:
            param_group: Dictionary containing parameters and their specific options
        """
        params = param_group["params"]
        if isinstance(params, Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            param_group["params"] = list(params)

        # Now we know params is a List[Tensor]
        param_list = cast(List[Tensor], param_group["params"])

        for param in param_list:
            if id(param) not in self.state:
                self.state[id(param)] = {}
            self._params.append(param)

    def state_dict(self) -> StateDict:
        """
        Returns the state of the optimizer as a dictionary.

        The state dictionary has two main components:
        - 'state': Maps parameter IDs to their optimization state
        - 'defaults': Contains the default hyperparameters

        Returns:
            A dictionary containing the complete optimizer state
        """
        return {"state": self.state, "defaults": self.defaults}

    def load_state_dict(self, state_dict: StateDict) -> None:
        """
        Loads the optimizer state from a dictionary.

        This method carefully restores both the parameter states and default
        hyperparameters while maintaining type safety.

        Args:
            state_dict: Dictionary containing optimizer state and defaults
        """
        # Cast with more specific types for better type safety
        self.state = cast(OptState, state_dict["state"])
        self.defaults = cast(OptDefaults, state_dict["defaults"])
