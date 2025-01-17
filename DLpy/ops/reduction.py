from typing import Dict, Optional, Union, Tuple
import numpy as np
from ..core import Function, Tensor

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
        if not isinstance(x, Tensor):
            x = Tensor(x)
            
        ctx.save_for_backward(x)
        ctx.save_arguments(axis=axis, keepdims=keepdims, input_shape=x.shape)
        
        return Tensor(np.sum(x.data, axis=axis, keepdims=keepdims))
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        x, = ctx.saved_tensors
        axis = ctx.saved_arguments['axis']
        keepdims = ctx.saved_arguments['keepdims']
        input_shape = ctx.saved_arguments['input_shape']
        
        if x.requires_grad:
            # If not keeping dims, need to reshape grad_output to match broadcast
            if not keepdims and axis is not None:
                grad_output = np.expand_dims(grad_output, axis=axis)
                
            # Broadcast gradient to match input shape
            grad = np.broadcast_to(grad_output, input_shape)
            grad_dict[id(x)] = grad

class Mean(Function):
    @staticmethod
    def forward(ctx, x, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
        if not isinstance(x, Tensor):
            x = Tensor(x)
            
        ctx.save_for_backward(x)
        ctx.save_arguments(axis=axis, keepdims=keepdims, input_shape=x.shape)
        
        return Tensor(np.mean(x.data, axis=axis, keepdims=keepdims))
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        x, = ctx.saved_tensors
        axis = ctx.saved_arguments['axis']
        keepdims = ctx.saved_arguments['keepdims']
        input_shape = ctx.saved_arguments['input_shape']
        
        if x.requires_grad:
            # If not keeping dims, need to reshape grad_output to match broadcast
            if not keepdims and axis is not None:
                grad_output = np.expand_dims(grad_output, axis=axis)
                
            # Calculate number of elements we're taking mean over
            if axis is None:
                n = np.prod(input_shape)
            else:
                n = np.prod([input_shape[i] for i in (axis,) if i < len(input_shape)])
                
            # Broadcast gradient to match input shape and divide by n
            grad = np.broadcast_to(grad_output, input_shape) / n
            grad_dict[id(x)] = grad

class Max(Function):
    @staticmethod
    def forward(ctx, x, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
        if not isinstance(x, Tensor):
            x = Tensor(x)
            
        result = np.amax(x.data, axis=axis, keepdims=True)
        ctx.save_for_backward(x)
        ctx.save_arguments(axis=axis, keepdims=keepdims, max_vals=result)
        
        if not keepdims:
            result = np.squeeze(result, axis=axis)
            
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        x, = ctx.saved_tensors
        axis = ctx.saved_arguments['axis']
        keepdims = ctx.saved_arguments['keepdims']
        max_vals = ctx.saved_arguments['max_vals']
        
        if x.requires_grad:
            # If not keeping dims, need to reshape grad_output
            if not keepdims and axis is not None:
                grad_output = np.expand_dims(grad_output, axis=axis)
                
            # Create gradient mask (1 where x equals max, 0 elsewhere)
            mask = (x.data == max_vals)
            
            # In case of multiple maxima, distribute gradient equally
            mask = mask / np.sum(mask, axis=axis, keepdims=True)
            
            grad_dict[id(x)] = grad_output * mask