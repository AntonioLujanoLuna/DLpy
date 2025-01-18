from typing import Tuple, Optional, Union
import numpy as np
from ..core import Tensor
from .modules import Module
from ..ops.cnn import Conv2dFunction

def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert input to a pair of values."""
    if isinstance(x, tuple):
        return x
    return (x, x)

class Conv2d(Module):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1 
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        
    Shape:
        - Input: (N, C_in, H, W)
        - Output: (N, C_out, H_out, W_out)
          where
          H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
          W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        
        # Initialize weights using He initialization
        # Adjust fan_in to account for groups
        fan_in = in_channels // groups * self.kernel_size[0] * self.kernel_size[1]
        bound = np.sqrt(2.0 / fan_in)
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        weight = Tensor(
            np.random.uniform(-bound, bound, weight_shape),
            requires_grad=True
        )
        self.register_parameter('weight', weight)
        
        if bias:
            # Initialize bias to zero
            bias_data = np.zeros(out_channels)
            self.register_parameter('bias', Tensor(bias_data, requires_grad=True))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the convolution layer.
        
        Args:
            x: Input tensor of shape (N, C_in, H, W)
            
        Returns:
            Output tensor of shape (N, C_out, H_out, W_out)
        """
        return Conv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        
    def extra_repr(self) -> str:
        """Returns a string with extra representation information."""
        s = (f'{self.in_channels}, {self.out_channels}, '
             f'kernel_size={self.kernel_size}')
        
        if self.stride != (1, 1):
            s += f', stride={self.stride}'
        if self.padding != (0, 0):
            s += f', padding={self.padding}'
        if self.dilation != (1, 1):
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        return s

    @staticmethod
    def calc_output_shape(
        input_shape: Tuple[int, ...],
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate the output shape of the convolution.
        
        Args:
            input_shape: Input shape (N, C_in, H, W)
            out_channels: Number of output channels
            kernel_size: Size of the kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides
            dilation: Spacing between kernel elements
            
        Returns:
            Output shape (N, C_out, H_out, W_out)
        """
        N, _, H, W = input_shape
        
        H_out = ((H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) 
                // stride[0] + 1)
        W_out = ((W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) 
                // stride[1] + 1)
        
        return (N, out_channels, H_out, W_out)