from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray


class Tensor:
    """
    A multidimensional array with autograd capabilities.

    The Tensor class wraps numpy arrays and adds automatic differentiation
    capabilities. It tracks the computational graph and enables gradient
    computation through backpropagation.

    Attributes:
        data: The underlying numpy array holding the tensor's values
        grad: Gradient of the loss with respect to this tensor
        requires_grad: Whether to compute gradients for this tensor
        _prev: Set of immediate predecessor nodes in computational graph
        _backward_fn: Function to compute gradients during backpropagation
        _is_leaf: Whether this tensor is a leaf node (created by user)
    """

    def __init__(
        self,
        data: Union[NDArray[Any], List[Any], Number],
        requires_grad: bool = False,
        dtype: Optional[DTypeLike] = None,
    ):
        # Convert scalars to scalar arrays with shape ()
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=dtype or np.float64)  # Will have shape ()
        elif isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, list):
            self.data = np.array(data, dtype=dtype)
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=dtype) if dtype else np.array(data)
        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype) if dtype else data
        else:
            self.data = np.array(data, dtype=dtype)

        self.grad: Optional[NDArray[Any]] = None
        self._requires_grad = requires_grad
        self._backward_fn: Optional[Callable[[NDArray[Any], Dict[int, NDArray[Any]]], None]] = None

        self._prev: Set["Tensor"] = set()
        self._is_leaf = True

        # Register with autograd engine
        from .autograd import get_autograd_engine

        engine = get_autograd_engine()
        engine.register_tensor(self)

        if requires_grad:
            self.zero_grad()

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self.data.shape)

    @property
    def dtype(self) -> np.dtype[np.float64]:
        return np.dtype("float64")

    @property
    def requires_grad(self) -> bool:
        """Returns whether the tensor requires gradient computation."""
        return self._requires_grad

    def __getitem__(self, index: Union[int, slice, Tuple[Union[int, slice], ...]]) -> "Tensor":
        """Enable indexing for tensors."""
        return Tensor(self.data[index], requires_grad=self.requires_grad)

    def __len__(self) -> int:
        """Return length of first dimension."""
        return self.data.shape[0] if self.data.shape else 1

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """Sets gradient computation requirement and returns self."""
        self._requires_grad = requires_grad
        if requires_grad and self.grad is None:
            self.zero_grad()
        return self

    def zero_grad(self) -> None:
        """Zeros out the gradient."""
        if self.data.shape == ():  # For scalar tensors
            self.grad = np.zeros(1, dtype=np.float64)  # Force 1D array
        else:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self, gradient: Optional[NDArray[Any]] = None) -> None:
        """
        Computes gradients of the loss with respect to this tensor.
        """
        if not self.requires_grad:
            return

        # Handle default gradient for scalar tensors
        if gradient is None:
            if np.prod(self.shape) == 1:
                if self.shape == ():  # scalar tensor
                    gradient = np.array(1.0)
                else:
                    gradient = np.ones(self.shape)
            else:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")

        # Ensure gradient is numpy array
        if isinstance(gradient, (int, float)):
            gradient = np.array(gradient)

        # Ensure matching shapes for scalar case
        if self.shape == () and gradient.shape != ():
            gradient = gradient.sum()
        elif self.shape != () and gradient.shape == ():
            gradient = np.full(self.shape, gradient)

        # Get autograd engine and execute backward pass
        from .autograd import get_autograd_engine

        engine = get_autograd_engine()
        engine.backward(self, gradient)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # Basic arithmetic operations that will be connected to Function implementations
    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        from ..ops.basic import Add

        return Add.apply(self, other)

    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        from ..ops.basic import Multiply

        return Multiply.apply(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        from ..ops.basic import MatMul

        return MatMul.apply(self, other)

    def __neg__(self) -> "Tensor":
        from ..ops.basic import Multiply

        return Multiply.apply(self, Tensor(-1))

    def __sub__(self, other: Union["Tensor", Number]) -> "Tensor":
        if isinstance(other, (int, float)):  # Only handle concrete numeric types
            return self + Tensor(-other)  # No need for float() conversion
        elif isinstance(other, complex):
            raise TypeError("Cannot convert complex number to tensor")
        elif isinstance(other, Tensor):
            return self + (-other)
        else:
            raise TypeError(f"Cannot subtract {type(other).__name__} from Tensor")

    # Helper methods for numpy compatibility
    def numpy(self) -> NDArray[Any]:
        """Returns the underlying numpy array."""
        return self.data

    @classmethod
    def from_numpy(cls, array: NDArray[Any], requires_grad: bool = False) -> "Tensor":
        """Creates a Tensor from a numpy array."""
        return cls(array.copy(), requires_grad=requires_grad)

    def reshape(self, *shape: int) -> "Tensor":
        """Returns a tensor with the same data and new shape."""
        # Unwrap nested tuples if passed as single argument
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        # Convert all elements to integers (handles -1 specially)
        processed_shape = tuple(d if d != -1 else -1 for d in shape)

        from ..ops import Reshape

        return Reshape.apply(self, processed_shape)

    def permute(self, *dims: int) -> "Tensor":
        """Permutes the dimensions of the tensor."""
        from ..ops import Transpose

        return Transpose.apply(self, dims)

    def pow(self, exponent: Union["Tensor", float]) -> "Tensor":
        """Returns tensor raised to the power of exponent."""
        from ..ops import Power

        return Power.apply(self, exponent)

    def div(self, other: Union["Tensor", float]) -> "Tensor":
        """Returns self divided by other."""
        from ..ops import Divide

        return Divide.apply(self, other)

    def log(self) -> "Tensor":
        """Returns the natural logarithm of the tensor."""
        from ..ops import Log

        return Log.apply(self)

    def exp(self) -> "Tensor":
        """Returns e raised to the power of each element in the tensor."""
        from ..ops import Exp

        return Exp.apply(self)

    def softmax(self, dim: int = -1) -> "Tensor":
        """Applies the softmax function along the specified dimension."""
        from ..ops.basic import Softmax

        return Softmax.apply(self, dim)

    def sigmoid(self) -> "Tensor":
        from ..nn.activations import SigmoidFunction

        return SigmoidFunction.apply(self)

    def tanh(self) -> "Tensor":
        from ..nn.activations import TanhFunction

        return TanhFunction.apply(self)

    def clip(self, min_val: Union[float, int], max_val: Union[float, int]) -> "Tensor":
        """Clips tensor values between minimum and maximum."""
        from ..ops import Clip

        return Clip.apply(self, min_val, max_val)

    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        """Returns the sum of all elements in the tensor."""
        from ..ops import Sum

        return Sum.apply(self, axis, keepdims)

    def mean(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        """Returns the mean of all elements in the tensor."""
        from ..ops import Mean

        return Mean.apply(self, axis, keepdims)

    def max(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        """Returns the maximum value of all elements in the tensor."""
        from ..ops import Max

        return Max.apply(self, axis, keepdims)

    def min(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        """Returns the minimum value of all elements in the tensor."""
        from ..ops import Min

        return Min.apply(self, axis, keepdims)

    def t(self) -> "Tensor":
        """Returns the transpose of the tensor."""
        from ..ops import Transpose

        return Transpose.apply(self)

    def transpose(self, *axes: int) -> "Tensor":
        """Returns the transposed tensor."""
        from ..ops import Transpose

        return Transpose.apply(self, axes or None)

    # Comparison operations
    def __gt__(self, other: Union["Tensor", float]) -> "Tensor":
        from ..ops import Greater

        return Greater.apply(self, other)

    def __ge__(self, other: Union["Tensor", float]) -> "Tensor":
        from ..ops import GreaterEqual

        return GreaterEqual.apply(self, other)

    def __lt__(self, other: Union["Tensor", float]) -> "Tensor":
        from ..ops import Less

        return Less.apply(self, other)

    def __le__(self, other: Union["Tensor", float]) -> "Tensor":
        from ..ops import LessEqual

        return LessEqual.apply(self, other)

    @overload
    def __eq__(self, other: Union["Tensor", float]) -> "Tensor": ...
    @overload
    def __eq__(self, other: Any) -> bool: ...  # Use Any instead of object for the fallback case

    def __eq__(self, other: Any) -> Union["Tensor", bool]:
        if isinstance(other, (Tensor, float)):
            from ..ops import Equal

            return Equal.apply(self, other)
        return NotImplemented

    def __ne__(self, other: Any) -> Any:  # Use Any for maximum flexibility
        """
        Implements != operator for tensors. Returns a new tensor for tensor operations,
        and delegates to Python's default behavior for non-tensor types.
        """
        if isinstance(other, (Tensor, float)):
            from ..ops import NotEqual

            return NotEqual.apply(self, other)
        return NotImplemented

    def __truediv__(self, other: Union["Tensor", float]) -> "Tensor":
        """Implements division using the / operator."""
        from ..ops import Divide

        return Divide.apply(self, other)

    def __pow__(self, exponent: Union["Tensor", float]) -> "Tensor":
        """Implements power using the ** operator."""
        from ..ops import Power

        return Power.apply(self, exponent)

    def copy(self) -> "Tensor":
        """Creates a deep copy of the tensor."""
        new_tensor = Tensor(
            self.data.copy(),  # Create a new numpy array with copied data
            requires_grad=self.requires_grad,
            dtype=self.dtype,
        )
        if self.grad is not None:
            new_tensor.grad = self.grad.copy()
        return new_tensor
