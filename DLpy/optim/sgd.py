
import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Implements stochastic gradient descent with momentum.

    Args:
        params: Iterable of parameters to optimize
        lr (float): Learning rate (default: 0.1)
        momentum (float): Momentum factor (default: 0)
        weight_decay (float): Weight decay (L2 penalty) (default: 0)
        dampening (float): Dampening for momentum (default: 0)
        nesterov (bool): Enables Nesterov momentum (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

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
