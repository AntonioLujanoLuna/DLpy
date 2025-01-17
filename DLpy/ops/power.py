from typing import Dict
import numpy as np
from ..core import Function, Tensor

class Power(Function):
    @staticmethod
    def forward(ctx, base, exponent):
        if not isinstance(base, Tensor):
            base = Tensor(base)
        if not isinstance(exponent, (Tensor, int, float)):
            raise TypeError("Exponent must be a Tensor, int, or float")
            
        # Convert Tensor exponent to scalar if possible
        if isinstance(exponent, Tensor):
            if exponent.data.size == 1:
                exponent = float(exponent.data)
            else:
                raise ValueError("Only scalar exponents are supported")
                
        ctx.save_for_backward(base)
        ctx.save_arguments(exponent=exponent)
        
        return Tensor(np.power(base.data, exponent))
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        base, = ctx.saved_tensors
        exponent = ctx.saved_arguments['exponent']
        
        if base.requires_grad:
            # d/dx(x^n) = nx^(n-1)
            grad = grad_output * exponent * np.power(base.data, exponent - 1)
            grad_dict[id(base)] = grad

class Divide(Function):
    @staticmethod
    def forward(ctx, numerator, denominator):
        if not isinstance(numerator, Tensor):
            numerator = Tensor(numerator)
        if not isinstance(denominator, Tensor):
            denominator = Tensor(denominator)
            
        # Check for division by zero
        if np.any(denominator.data == 0):
            raise ValueError("Division by zero encountered")
            
        ctx.save_for_backward(numerator, denominator)
        return Tensor(numerator.data / denominator.data)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        numerator, denominator = ctx.saved_tensors
        
        if numerator.requires_grad:
            # d/dx(x/y) = 1/y
            grad_dict[id(numerator)] = grad_output / denominator.data
            
        if denominator.requires_grad:
            # d/dy(x/y) = -x/y^2
            grad_dict[id(denominator)] = -grad_output * numerator.data / (denominator.data ** 2)