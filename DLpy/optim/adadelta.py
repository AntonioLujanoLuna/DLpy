import numpy as np
from typing import Dict, Iterator, Optional
from .optimizer import Optimizer
from ..core import Tensor

class AdaDelta(Optimizer):
    """
    Implements AdaDelta algorithm.
    
    AdaDelta is a more robust extension of AdaGrad that adapts learning rates based on a 
    moving window of gradient updates, instead of accumulating all past squared gradients.
    The main advantage is that it doesn't need an initial learning rate.
    
    Args:
        params: Iterable of parameters to optimize
        rho (float): Coefficient for computing a running average of squared gradients (default: 0.9)
        eps (float): Term added to denominator to improve numerical stability (default: 1e-6)
        weight_decay (float): Weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, rho: float = 0.9, eps: float = 1e-6, weight_decay: float = 0):
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self._params:
            state = self.state[id(group)]
            state['step'] = 0
            state['square_avg'] = np.zeros_like(group.data, dtype=np.float64)  # E[g^2]
            state['acc_delta'] = np.zeros_like(group.data, dtype=np.float64)   # E[Δx^2]

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
            if self.defaults['weight_decay'] != 0:
                grad = grad + self.defaults['weight_decay'] * p.data

            state['step'] += 1

            # Get parameters
            rho = self.defaults['rho']
            eps = self.defaults['eps']

            # Update running average of squared gradients
            square_avg = state['square_avg']
            acc_delta = state['acc_delta']
            
            # Update square_avg using numpy operations
            square_avg = rho * square_avg + (1 - rho) * grad * grad
            state['square_avg'] = square_avg
            
            # Compute update
            std = np.sqrt(acc_delta + eps)
            delta = np.sqrt(square_avg + eps)
            update = grad * std / delta

            # Update running average of squared updates
            acc_delta = rho * acc_delta + (1 - rho) * update * update
            state['acc_delta'] = acc_delta
            
            # Apply update
            p.data -= update

    def state_dict(self) -> Dict:
        """Returns the state of the optimizer as a Dict."""
        return {
            'state': self.state,
            'defaults': self.defaults
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads the optimizer state."""
        self.state = state_dict['state']
        self.defaults = state_dict['defaults']