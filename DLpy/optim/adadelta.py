from typing import Any, Dict, Iterator, List, Union

import numpy as np

from ..core import Tensor
from .optimizer import Optimizer


class AdaDelta(Optimizer):
    """
    Implements AdaDelta algorithm.

    AdaDelta is a more robust extension of AdaGrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past squared gradients. The main advantage is that it doesn't need an
    initial learning rate.

    It works by maintaining two running averages:
    1. A running average of squared gradients (similar to RMSprop)
    2. A running average of squared parameter updates

    These averages allow AdaDelta to automatically adapt its learning rate for
    each parameter, making it particularly useful when the optimal learning rate
    is hard to determine.

    Args:
        params: List or Iterator of parameters to optimize
        rho: Coefficient for computing running averages (default: 0.9)
        eps: Term added to denominator for numerical stability (default: 1e-6)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params: Union[Iterator[Tensor], List[Tensor]],
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        # Validate input parameters with descriptive error messages
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}. Must be between 0 and 1")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}. Must be non-negative")
        if not 0.0 <= weight_decay:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}. Must be non-negative"
            )

        # Create defaults dictionary with explicit type annotation
        # All values in this dictionary are floats
        defaults: Dict[str, float] = dict(rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize state for each parameter with proper typing
        for group in self._params:
            state = self.state[id(group)]
            state["step"] = 0
            # Use float64 for better numerical precision in running averages
            state["square_avg"] = np.zeros_like(group.data, dtype=np.float64)  # E[g^2]
            state["acc_delta"] = np.zeros_like(group.data, dtype=np.float64)  # E[Δx^2]

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a Dict.

        The state dictionary contains:
        - 'state': A dictionary mapping parameter IDs to their optimization state
                  (including running averages of squared gradients and updates)
        - 'defaults': The optimizer's hyperparameters (rho, epsilon, weight decay)

        Returns:
            A dictionary containing the complete optimizer state
        """
        return {"state": self.state, "defaults": self.defaults}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the optimizer state from a dictionary.

        This method enables saving and restoring optimizer state during training,
        which is essential for checkpointing and resuming training sessions.

        Args:
            state_dict: Dictionary containing optimizer state and parameters.
                       Must have 'state' and 'defaults' keys.
        """
        self.state = state_dict["state"]
        self.defaults = state_dict["defaults"]

    def step(self) -> None:
        """
        Performs a single optimization step.

        For each parameter:
        1. Compute running average of squared gradients
        2. Compute parameter update using accumulated squared updates
        3. Update running average of squared updates
        4. Apply update to parameters
        """
        for p in self._params:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[id(p)]

            # Apply weight decay if specified
            if self.defaults["weight_decay"] != 0:
                grad = grad + self.defaults["weight_decay"] * p.data

            state["step"] += 1

            # Get parameters
            rho = self.defaults["rho"]
            eps = self.defaults["eps"]

            # Update running average of squared gradients
            square_avg = state["square_avg"]
            acc_delta = state["acc_delta"]

            # Update square_avg using numpy operations
            square_avg = rho * square_avg + (1 - rho) * grad * grad
            state["square_avg"] = square_avg

            # Compute update
            std = np.sqrt(acc_delta + eps)
            delta = np.sqrt(square_avg + eps)
            update = grad * std / delta

            # Update running average of squared updates
            acc_delta = rho * acc_delta + (1 - rho) * update * update
            state["acc_delta"] = acc_delta

            # Apply update
            p.data -= update
