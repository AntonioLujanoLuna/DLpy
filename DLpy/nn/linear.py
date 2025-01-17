from typing import Optional, Dict
import numpy as np
from ..core import Tensor, Function
from .modules import Module

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        # Save tensors needed for backward pass
        ctx.save_for_backward(input, weight, bias)
        
        # Compute output: y = xW^T + b
        output = input.data @ weight.data
        if bias is not None:
            output += bias.data
            
        return Tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        # Retrieve saved tensors
        input, weight, bias = ctx.saved_tensors
        
        # Compute gradient with respect to input: dx = dout @ W
        if input.requires_grad:
            grad_dict[id(input)] = grad_output @ weight.data.T
            
        # Compute gradient with respect to weight: dW = x^T @ dout
        if weight.requires_grad:
            grad_dict[id(weight)] = input.data.T @ grad_output
            
        # Compute gradient with respect to bias: db = sum(dout, dim=0)
        if bias is not None and bias.requires_grad:
            grad_dict[id(bias)] = grad_output.sum(axis=0)

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xW^T + b
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        
    Shape:
        - Input: (batch_size, in_features)
        - Output: (batch_size, out_features)
        
    Attributes:
        weight: the learnable weights of shape (in_features, out_features)
        bias: the learnable bias of shape (out_features,)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using He initialization
        bound = np.sqrt(2.0 / in_features)
        weight = Tensor(
            np.random.uniform(-bound, bound, (in_features, out_features)),
            requires_grad=True
        )
        self.register_parameter('weight', weight)
        
        if bias:
            bias = Tensor(np.zeros(out_features), requires_grad=True)
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the linear layer."""
        return LinearFunction.apply(input, self.weight, self.bias)
            
    def extra_repr(self) -> str:
        """Extra information to add to the string representation."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'