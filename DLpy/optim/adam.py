from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Implements Adam algorithm.

    Args:
        params: Iterable of parameters to optimize
        lr (float): Learning rate (default: 0.001)
        betas (tuple): Coefficients for computing running averages of gradient and its square
            (default: (0.9, 0.999))
        eps (float): Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay (float): Weight decay (L2 penalty) (default: 0)
        amsgrad (bool): Whether to use the AMSGrad variant (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

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

    def state_dict(self) -> Dict:
        """Returns the state of the optimizer as a Dict."""
        return {"state": self.state, "defaults": self.defaults}

    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads the optimizer state."""
        self.state = state_dict["state"]
        self.defaults = state_dict["defaults"]
