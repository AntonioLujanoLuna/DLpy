from typing import Any, Dict, Iterator, List, Union

import numpy as np

from ..core import Tensor
from .optimizer import Optimizer


class RMSprop(Optimizer):
    """
    Implements RMSprop algorithm.

    Args:
        params: List or Iterator of parameters to optimize
        lr: Learning rate (default: 0.01)
        alpha: Smoothing constant (default: 0.99)
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        momentum: Momentum factor (default: 0)
        centered: If True, compute centered RMSprop with variance-normalized gradients
    """

    def __init__(
        self,
        params: Union[Iterator[Tensor], List[Tensor]],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ) -> None:
        # Parameter validation remains the same
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Create defaults dictionary with explicit type annotation
        defaults: Dict[str, Union[float, bool]] = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(params, defaults)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a Dict.

        Returns:
            A dictionary containing:
                - 'state': Dict mapping parameter IDs to their optimization state
                - 'defaults': Dict of optimizer hyperparameters
        """
        return {"state": self.state, "defaults": self.defaults}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the optimizer state.

        Args:
            state_dict: Dictionary containing optimizer state and parameters.
                       Should have 'state' and 'defaults' keys.
        """
        self.state = state_dict["state"]
        self.defaults = state_dict["defaults"]

    def step(self) -> None:
        """Performs a single optimization step."""
        for p in self._params:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[id(p)]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                state["square_avg"] = np.zeros_like(p.data)
                if self.defaults["momentum"] > 0:
                    state["momentum_buffer"] = np.zeros_like(p.data)
                if self.defaults["centered"]:
                    state["grad_avg"] = np.zeros_like(p.data)

            square_avg = state["square_avg"]
            alpha = self.defaults["alpha"]

            state["step"] += 1

            if self.defaults["weight_decay"] != 0:
                grad = grad + self.defaults["weight_decay"] * p.data

            # Update squared average
            square_avg = alpha * square_avg + (1 - alpha) * grad * grad

            if self.defaults["centered"]:
                grad_avg = state["grad_avg"]
                grad_avg = alpha * grad_avg + (1 - alpha) * grad
                avg = square_avg - grad_avg * grad_avg
                state["grad_avg"] = grad_avg
            else:
                avg = square_avg

            # Apply momentum if enabled
            if self.defaults["momentum"] > 0:
                buf = state.get("momentum_buffer", np.zeros_like(grad))
                buf = self.defaults["momentum"] * buf + grad / (np.sqrt(avg) + self.defaults["eps"])
                state["momentum_buffer"] = buf
                p.data -= self.defaults["lr"] * buf
            else:
                p.data -= self.defaults["lr"] * grad / (np.sqrt(avg) + self.defaults["eps"])

            # Save state
            state["square_avg"] = square_avg
