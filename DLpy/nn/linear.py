from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..core import Function, Module, Tensor


class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xW^T + b

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using He initialization
        bound = np.sqrt(2.0 / in_features)
        weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)), requires_grad=True
        )
        self.register_parameter("weight", weight)

        if has_bias:  # Changed variable name to avoid conflict
            bias_tensor = Tensor(np.zeros(out_features), requires_grad=True)
            self.register_parameter("bias", bias_tensor)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the linear layer."""
        return LinearFunction.apply(input, self.weight, self.bias)


class LinearFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,  # Type hint for context
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        # Save tensors needed for backward pass
        ctx.save_for_backward(input, weight, bias)

        # Compute output: y = xW^T + b
        output = input.data @ weight.data.T
        if bias is not None:
            output = output + bias.data

        return Tensor(output)

    @staticmethod
    def backward(
        ctx: Any,  # Type hint for context
        grad_output: NDArray[Any],
        grad_dict: Dict[int, NDArray[Any]],
    ) -> None:
        input, weight, bias = ctx.saved_tensors

        if input.requires_grad:
            # For input gradient: (batch_size, out_features) @ (out_features, in_features)
            grad_dict[id(input)] = grad_output @ weight.data

        if weight.requires_grad:
            # For weight gradient: (in_features, batch_size) @ (batch_size, out_features)
            # Reshape grad_output to (batch_size, out_features) if needed
            if len(grad_output.shape) == 3:
                batch_size, seq_len, out_features = grad_output.shape
                grad_output = grad_output.reshape(-1, out_features)
                input_data = input.data.reshape(-1, input.data.shape[-1])
            else:
                input_data = input.data

            grad_dict[id(weight)] = input_data.T @ grad_output

        if bias is not None and bias.requires_grad:
            # Sum across batch dimension
            grad_dict[id(bias)] = grad_output.sum(axis=0)
            if len(grad_output.shape) == 3:
                grad_dict[id(bias)] = grad_dict[id(bias)].sum(axis=0)
