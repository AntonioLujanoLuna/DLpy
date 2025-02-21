from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np

from ..core import Tensor
from .optimizer import Optimizer


class AdaMax(Optimizer):
    """
    Implements AdaMax algorithm, a variant of Adam based on the infinity norm.

    AdaMax is a variant of Adam that adopts the infinity norm in place of the L2 norm.
    It tends to be more stable than Adam in some cases.

    Args:
        params: List or Iterator of parameters to optimize
        lr: Learning rate (default: 0.002)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params: Union[Iterator[Tensor], List[Tensor]],
        lr: float = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        # Validate input parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Create defaults dictionary with explicit type annotation
        defaults: Dict[str, Union[float, Tuple[float, float]]] = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self._params:
            state = self.state[id(group)]
            state["step"] = 0
            # Note: we explicitly specify dtype for numerical stability
            state["exp_avg"] = np.zeros_like(group.data, dtype=np.float64)  # m_t
            state["exp_inf"] = np.zeros_like(group.data, dtype=np.float64)  # u_t

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a Dict.

        Returns:
            A dictionary containing optimizer state and default parameters:
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
        """
        Performs a single optimization step.

        For each parameter:
        1. Update biased first moment estimate
        2. Update the exponentially weighted infinity norm
        3. Compute bias-corrected learning rate
        4. Update parameters
        """
        for p in self._params:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[id(p)]

            # Apply weight decay if specified
            if self.defaults["weight_decay"] != 0:
                grad = grad + self.defaults["weight_decay"] * p.data

            # Get parameters
            beta1, beta2 = self.defaults["betas"]
            lr = self.defaults["lr"]
            eps = self.defaults["eps"]

            state["step"] += 1
            bias_correction = 1 - beta1 ** state["step"]

            # Get momentum buffer
            exp_avg = state["exp_avg"]
            exp_inf = state["exp_inf"]

            # Update biased first moment estimate using numpy operations
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            state["exp_avg"] = exp_avg

            # Update the exponentially weighted infinity norm
            exp_inf = np.maximum(beta2 * exp_inf, np.abs(grad))
            state["exp_inf"] = exp_inf

            # Compute bias-corrected learning rate
            step_size = lr / bias_correction

            # Update parameters
            p.data -= step_size * exp_avg / (exp_inf + eps)
