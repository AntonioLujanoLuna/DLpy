import numpy as np

from ...core import Module, Tensor


class GroupNorm(Module):
    """
    Applies Group Normalization over a mini-batch of inputs.

    Group Normalization divides channels into groups and computes within each group
    the mean and variance for normalization.

    Args:
        num_groups: Number of groups to separate the channels into
        num_channels: Number of channels expected in input
        eps: Small value for numerical stability
        affine: If True, use learnable affine parameters
    """

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = Tensor(np.ones(num_channels), requires_grad=True)
            self.bias = Tensor(np.zeros(num_channels), requires_grad=True)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # Check input dimensions
        if len(x.shape) != 4:
            raise ValueError(f"expected 4D input, got {len(x.shape)}D input")

        N, C, H, W = x.shape
        if C != self.num_channels:
            raise ValueError(f"expected {self.num_channels} channels, got {C} channels")

        # Reshape input to (N, G, C/G, H, W)
        x_reshaped = x.data.reshape(N, self.num_groups, -1, H, W)

        # Calculate mean and variance over (C/G, H, W) for each group
        mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
        var = x_reshaped.var(axis=(2, 3, 4), keepdims=True, ddof=0)

        # Normalize
        x_norm = (x_reshaped - mean) / np.sqrt(var + self.eps)

        # Reshape back to (N, C, H, W)
        x_norm = x_norm.reshape(N, C, H, W)

        # Apply affine transform if specified
        if self.affine:
            x_norm = x_norm * self.weight.data.reshape(
                1, -1, 1, 1
            ) + self.bias.data.reshape(1, -1, 1, 1)

        return Tensor(x_norm, requires_grad=x.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, num_channels={self.num_channels}, "
            f"eps={self.eps}, affine={self.affine}"
        )
