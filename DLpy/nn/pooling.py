from typing import Optional, Tuple, Union

from ..core import Module, Tensor
from ..ops.pooling import AvgPool2dFunction, MaxPool2dFunction


class MaxPool2d(Module):
    """2D max pooling layer."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()

        # Convert scalar parameters to tuples
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (
            self.kernel_size
            if stride is None
            else (stride, stride) if isinstance(stride, int) else stride
        )
        self.padding = (padding, padding) if isinstance(padding, int) else padding

    def forward(self, x: Tensor) -> Tensor:
        return MaxPool2dFunction.apply(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool2d(Module):
    """2D average pooling layer."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()

        # Convert scalar parameters to tuples
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (
            self.kernel_size
            if stride is None
            else (stride, stride) if isinstance(stride, int) else stride
        )
        self.padding = (padding, padding) if isinstance(padding, int) else padding

    def forward(self, x: Tensor) -> Tensor:
        return AvgPool2dFunction.apply(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
