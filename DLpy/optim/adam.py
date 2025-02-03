from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np

from ..core import Tensor
from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Implements Adam algorithm.

    The Adam optimizer combines ideas from RMSprop and momentum optimization:
    - It uses exponential moving averages of gradients (like momentum)
    - It uses exponential moving averages of squared gradients (like RMSprop)
    - It includes bias correction for more accurate initial steps

    Args:
        params: List or Iterator of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages of gradient and its square
            (default: (0.9, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        amsgrad: Whether to use the AMSGrad variant (default: False)
    """

    def __init__(
        self,
        params: Union[Iterator[Tensor], List[Tensor]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ) -> None:
        # Input validation with descriptive error messages
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Create defaults dictionary with explicit type annotation
        # Note that values can be float, tuple of floats, or boolean
        defaults: Dict[str, Union[float, Tuple[float, float], bool]] = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super().__init__(params, defaults)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a Dict.

        The state dictionary contains:
        - 'state': A dictionary mapping parameter IDs to their optimization state
                  (including momentum and adaptive learning rate information)
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
        """Performs a single optimization step."""
        for p in self._params:
            if p.grad is None:
                continue

            grad = p.grad

            # Get optimizer state
            state = self.state[id(p)]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = np.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = np.zeros_like(p.data)
                if self.defaults["amsgrad"]:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = np.zeros_like(p.data)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            if self.defaults["amsgrad"]:
                max_exp_avg_sq = state["max_exp_avg_sq"]
            beta1, beta2 = self.defaults["betas"]

            state["step"] += 1
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            if self.defaults["weight_decay"] != 0:
                grad = grad + self.defaults["weight_decay"] * p.data

            # Decay the first and second moment running average coefficient
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

            if self.defaults["amsgrad"]:
                # Maintains the maximum of all 2nd moment running avg. till now
                max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (np.sqrt(max_exp_avg_sq) / np.sqrt(bias_correction2)) + self.defaults["eps"]
            else:
                denom = (np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2)) + self.defaults["eps"]

            step_size = self.defaults["lr"] / bias_correction1

            p.data -= step_size * exp_avg / denom

            # Save state
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq
            if self.defaults["amsgrad"]:
                state["max_exp_avg_sq"] = max_exp_avg_sq
