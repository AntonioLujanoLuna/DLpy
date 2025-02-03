from typing import Any, Dict, Iterator, List, Union

import numpy as np

from ..core import Tensor
from .optimizer import Optimizer


class AdaGrad(Optimizer):
    """
    Implements AdaGrad algorithm.

    AdaGrad is an optimizer with parameter-specific learning rates,
    which are adapted based on historical gradient information. It performs
    smaller updates for frequently occurring features and larger updates
    for infrequent ones.

    Args:
        params: List or Iterator of parameters to optimize
        lr: Learning rate (default: 1e-2)
        lr_decay: Learning rate decay (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        eps: Term added to denominator to improve numerical stability (default: 1e-10)
        initial_accumulator_value: Initial value for accumulator (default: 0)
    """

    def __init__(
        self,
        params: Union[Iterator[Tensor], List[Tensor]],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
    ) -> None:
        # Input validation with descriptive error messages
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # Create defaults dictionary with explicit type annotation
        # All values in this dictionary are floats
        defaults: Dict[str, float] = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self._params:
            state = self.state[id(group)]
            state["step"] = 0
            # Use np.float64 for better numerical precision in accumulator
            state["sum"] = np.full_like(group.data, initial_accumulator_value, dtype=np.float64)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a Dict.

        The state dictionary contains:
        - 'state': A dictionary mapping parameter IDs to their optimization state
                  (including step counts and accumulated squared gradients)
        - 'defaults': The optimizer's hyperparameters

        Returns:
            A dictionary containing the complete optimizer state
        """
        return {"state": self.state, "defaults": self.defaults}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the optimizer state from a dictionary.

        This method allows saving and restoring optimizer state during training,
        which is useful for checkpointing or resuming training.

        Args:
            state_dict: Dictionary containing optimizer state and parameters.
                       Must have 'state' and 'defaults' keys.
        """
        self.state = state_dict["state"]
        self.defaults = state_dict["defaults"]

    def step(self) -> None:
        """
        Performs a single optimization step.

        For each parameter p, accumulates the square of the gradient and then
        updates the parameter using the formula:
        p = p - lr * g / (sqrt(accumulator) + eps)
        where g is the gradient.
        """
        for p in self._params:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[id(p)]

            state["step"] += 1

            if self.defaults["weight_decay"] != 0:
                grad = grad + self.defaults["weight_decay"] * p.data

            # Update accumulator with squared gradient
            state["sum"] += grad * grad

            # Compute the adaptive learning rate
            std = np.sqrt(state["sum"])

            # Add epsilon for numerical stability before division
            denom = std + self.defaults["eps"]

            # Apply learning rate decay if specified
            if self.defaults["lr_decay"] != 0:
                lr = self.defaults["lr"] / (1 + (state["step"] - 1) * self.defaults["lr_decay"])
            else:
                lr = self.defaults["lr"]

            # Update parameters
            p.data -= lr * grad / denom

    def reset_state(self) -> None:
        """
        Resets the state of the optimizer.

        This can be useful when you want to restart training or when you want to
        reset the accumulated gradients without creating a new optimizer instance.
        """
        initial_accumulator_value = self.defaults["initial_accumulator_value"]

        for group in self._params:
            state = self.state[id(group)]
            state["step"] = 0
            state["sum"] = np.full_like(group.data, initial_accumulator_value, dtype=np.float64)
