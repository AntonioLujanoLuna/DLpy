from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

from .context import Context
from .tensor import Tensor  # This will be implemented next


class Function(ABC):
    """
    Base class for all autograd operations.

    This class defines the interface for creating differentiable operations.
    Each operation should implement both a forward pass (computing the result)
    and a backward pass (computing gradients).

    The Function class follows a similar design pattern to PyTorch's autograd.Function,
    but with some simplifications and additional features for clarity and debugging.
    """

    requires_grad: bool = True

    @staticmethod
    @abstractmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> Tensor:
        """
        Performs the forward computation.

        Args:
            ctx: Context object for saving information needed in backward pass
            *args: Input tensors and other arguments
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the computation as a Tensor
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(ctx: Context, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        """
        Computes gradients of the operation with respect to its inputs.

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output: Gradient of the loss with respect to the output
            grad_dict: Dictionary mapping tensor IDs to their gradients
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Tensor:
        """
        Applies the function to the given inputs.

        This method:
        1. Creates a Context object for storing intermediate values
        2. Runs the forward pass
        3. Sets up the computational graph for gradient computation
        4. Returns the result
        """
        ctx = Context()
        result = cls.forward(ctx, *args, **kwargs)

        # Check if we need to compute gradients
        needs_grad = cls.requires_grad and any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        if needs_grad:

            def backward_fn(grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
                cls.backward(ctx, grad_output, grad_dict)

            result._backward_fn = backward_fn
            result.requires_grad_(True)

            # Get autograd engine and register edges
            from .autograd import get_autograd_engine

            engine = get_autograd_engine()
            for arg in args:
                if isinstance(arg, Tensor):
                    engine.add_edge(arg, result)

        return result  # Return result in all cases

    @staticmethod
    def verify_backward(
        forward_fn: Any, backward_fn: Any, inputs: Tuple[np.ndarray, ...], epsilon: float = 1e-6
    ) -> bool:
        """
        Verifies backward pass implementation using numerical gradients.

        This helper method compares analytically computed gradients with
        numerically computed gradients to check for correctness.

        Args:
            forward_fn: The forward pass function
            backward_fn: The backward pass function
            inputs: Tuple of input arrays
            epsilon: Small value for numerical gradient computation

        Returns:
            True if gradients match within tolerance, False otherwise
        """

        def compute_numerical_gradient(idx: int, inp: np.ndarray) -> np.ndarray:
            grad = np.zeros_like(inp)
            it = np.nditer(inp, flags=["multi_index"])

            while not it.finished:
                ix = it.multi_index
                old_value = inp[ix]

                # Compute f(x + epsilon)
                inp[ix] = old_value + epsilon
                pos_inputs = list(inputs)
                pos_inputs[idx] = inp.copy()
                pos_output = forward_fn(*pos_inputs)

                # Compute f(x - epsilon)
                inp[ix] = old_value - epsilon
                neg_inputs = list(inputs)
                neg_inputs[idx] = inp.copy()
                neg_output = forward_fn(*neg_inputs)

                # Restore original value
                inp[ix] = old_value

                # Compute numerical gradient
                grad[ix] = np.sum(pos_output - neg_output) / (2 * epsilon)
                it.iternext()

            return grad

        # Compute analytical gradients
        ctx = Context()
        output = forward_fn(*inputs)
        grad_output = np.ones_like(output)
        analytical_grads = backward_fn(ctx, grad_output)

        # Compute numerical gradients
        numerical_grads = tuple(
            compute_numerical_gradient(i, inp.copy()) for i, inp in enumerate(inputs)
        )

        # Compare gradients
        for analytical, numerical in zip(analytical_grads, numerical_grads):
            if analytical is not None:
                rel_error = np.max(
                    np.abs(analytical - numerical)
                    / (np.maximum(np.abs(analytical), np.abs(numerical)) + epsilon)
                )
                if rel_error > 1e-5:
                    return False

        return True
