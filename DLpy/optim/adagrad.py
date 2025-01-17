import numpy as np
from typing import Dict, Iterator, Optional
from .optimizer import Optimizer
from ..core import Tensor

class AdaGrad(Optimizer):
    """
    Implements AdaGrad algorithm.
    
    AdaGrad is an optimizer with parameter-specific learning rates,
    which are adapted based on historical gradient information.
    
    Args:
        params: Iterable of parameters to optimize
        lr (float): Learning rate (default: 1e-2)
        lr_decay (float): Learning rate decay (default: 0)
        weight_decay (float): Weight decay (L2 penalty) (default: 0)
        eps (float): Term added to denominator to improve numerical stability (default: 1e-10)
        initial_accumulator_value (float): Initial value for accumulator (default: 0)
    """
    
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0,
                 weight_decay: float = 0, initial_accumulator_value: float = 0,
                 eps: float = 1e-10):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(f"Invalid initial_accumulator_value value: {initial_accumulator_value}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, 
                       weight_decay=weight_decay,
                       initial_accumulator_value=initial_accumulator_value)
        super().__init__(params, defaults)

        for group in self._params:
            state = self.state[id(group)]
            state['step'] = 0
            state['sum'] = np.full_like(group.data, initial_accumulator_value, dtype=np.float64)

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

            state['step'] += 1

            if self.defaults['weight_decay'] != 0:
                grad = grad + self.defaults['weight_decay'] * p.data

            # Update accumulator with squared gradient
            state['sum'] += grad * grad

            # Compute the adaptive learning rate
            std = np.sqrt(state['sum'])
            
            # Add epsilon for numerical stability before division
            denom = std + self.defaults['eps']

            # Apply learning rate decay if specified
            if self.defaults['lr_decay'] != 0:
                lr = self.defaults['lr'] / (1 + (state['step'] - 1) * self.defaults['lr_decay'])
            else:
                lr = self.defaults['lr']

            # Update parameters
            p.data -= lr * grad / denom

    def reset_state(self) -> None:
        """
        Resets the state of the optimizer.
        
        This can be useful when you want to restart training or when you want to 
        reset the accumulated gradients without creating a new optimizer instance.
        """
        initial_accumulator_value = self.defaults['initial_accumulator_value']
        
        for group in self._params:
            state = self.state[id(group)]
            state['step'] = 0
            state['sum'] = np.full_like(group.data, initial_accumulator_value, dtype=np.float64)

    def state_dict(self) -> Dict:
        """
        Returns the state of the optimizer as a Dict.
        
        The returned state dict contains two entries:
            * state - a dict holding current optimization state. Its content
                differs between optimizer classes.
            * param_groups - a dict containing all parameter groups
        """
        return {
            'state': self.state,
            'defaults': self.defaults
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Loads the optimizer state.
        
        Args:
            state_dict (dict): Optimizer state. Should be an object returned
                from a call to state_dict().
        """
        self.state = state_dict['state']
        self.defaults = state_dict['defaults']