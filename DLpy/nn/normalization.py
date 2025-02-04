import numpy as np

from ..core import Module, Tensor


class InstanceNorm2d(Module):
    """Applies Instance Normalization over a 4D input"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Tensor(np.ones(num_features), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features), requires_grad=True)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", Tensor(np.zeros(num_features)))
            self.register_buffer("running_var", Tensor(np.ones(num_features)))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, x: Tensor) -> Tensor:
        if self.training or not self.track_running_stats:
            # Calculate per-instance statistics
            batch_size, num_channels = x.shape[:2]
            x_reshaped = x.data.reshape(batch_size, num_channels, -1)

            mean = x_reshaped.mean(axis=2, keepdims=True)  # (N, C, 1)
            var = x_reshaped.var(axis=2, keepdims=True, ddof=0)  # (N, C, 1)

            # Reshape for broadcasting (N, C, 1, 1)
            mean = mean.reshape(batch_size, num_channels, 1, 1)
            var = var.reshape(batch_size, num_channels, 1, 1)

            if self.track_running_stats and self.training:
                # Update running stats with proper dimension handling
                self.running_mean.data = (
                    1 - self.momentum
                ) * self.running_mean.data + self.momentum * mean.mean(
                    axis=0
                ).squeeze()  # Remove extra dimensions
                self.running_var.data = (
                    1 - self.momentum
                ) * self.running_var.data + self.momentum * var.mean(axis=0).squeeze()
        else:
            # Use running stats with correct broadcasting shape
            mean = self.running_mean.data.reshape(1, -1, 1, 1)
            var = self.running_var.data.reshape(1, -1, 1, 1)

        # Normalization
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)

        # Apply affine transform if enabled
        if self.affine:
            x_norm = x_norm * self.weight.data.reshape(
                1, -1, 1, 1
            ) + self.bias.data.reshape(1, -1, 1, 1)

        return Tensor(x_norm, requires_grad=x.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


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
