from typing import Dict

from ..core import Tensor


class Optimizer:
    """
    Base class for all optimizers.

    Args:
        params: An iterable of parameters to optimize or a dict of parameter groups
        defaults: Dictionary of default hyperparameter values
    """

    def __init__(self, params, defaults: Dict):
        self.defaults = defaults
        self._params = list(params)  # Convert iterator to list
        self.state: Dict = {}  # State dict for optimizer states

        # Initialize state for each parameter
        for p in self._params:
            self.state[id(p)] = {}

    def zero_grad(self) -> None:
        """Clears the gradients of all optimized parameters."""
        for p in self._params:
            if p.grad is not None:
                p.grad.fill(0)

    def step(self) -> None:
        """Performs a single optimization step.

        This method should be overridden by all optimizers.
        """
        raise NotImplementedError

    def add_param_group(self, param_group: Dict) -> None:
        """Add a param group to the optimizer's param groups.

        Args:
            param_group (dict): Specifies parameters and parameter-specific options
        """
        params = param_group["params"]
        if isinstance(params, Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if id(param) not in self.state:
                self.state[id(param)] = {}
            self._params.append(param)

    def state_dict(self) -> Dict:
        """Returns the state of the optimizer as a Dict."""
        return {"state": self.state, "defaults": self.defaults}

    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads the optimizer state."""
        self.state = state_dict["state"]
        self.defaults = state_dict["defaults"]
