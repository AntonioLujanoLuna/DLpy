from typing import Any, Dict, Iterator, List, Union

import numpy as np

from ..core import Tensor
from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Implements stochastic gradient descent with momentum.

    Args:
        params: List or Iterator of parameters to optimize
        lr: Learning rate (default: 0.1)
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
    """

    def __init__(
        self,
        params: Union[Iterator[Tensor], List[Tensor]],
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults: Dict[str, Union[float, bool]] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a Dict.

        Returns a dictionary containing:
        - 'state': Dict mapping parameter IDs to their optimization state
        - 'defaults': Dict of optimizer hyperparameters
        """
        return {"state": self.state, "defaults": self.defaults}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the optimizer state.

        Args:
            state_dict: Dictionary containing optimizer state and parameters
        """
        self.state = state_dict["state"]
        self.defaults = state_dict["defaults"]

    def step(self) -> None:
        """Performs a single optimization step."""

        for p in self._params:
            if p.grad is None:
                continue

            grad = p.grad

            # Apply weight decay
            if self.defaults["weight_decay"] != 0:
                grad = grad + self.defaults["weight_decay"] * p.data

            # Get or initialize momentum buffer
            if "momentum_buffer" not in self.state[id(p)]:
                buf = self.state[id(p)]["momentum_buffer"] = np.zeros_like(p.data)
            else:
                buf = self.state[id(p)]["momentum_buffer"]

            # Update momentum buffer
            if self.defaults["momentum"] != 0:
                buf *= self.defaults["momentum"]
                if self.defaults["dampening"] != 0:
                    grad *= 1 - self.defaults["dampening"]
                buf += grad
            else:
                buf = grad

            # Nesterov momentum
            if self.defaults["nesterov"]:
                grad += self.defaults["momentum"] * buf
            else:
                grad = buf

            # Update parameters
            p.data -= self.defaults["lr"] * grad

            # Store updated momentum buffer
            self.state[id(p)]["momentum_buffer"] = buf
