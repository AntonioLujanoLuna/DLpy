# batch_norm.py
import numpy as np
from typing import Optional, Tuple
from ..core import Tensor, Module

class BatchNorm1d(Module):
    """
    Applies Batch Normalization over a 2D input (batch, features)

    Args:
        num_features: Number of features or channels
        eps: Small constant for numerical stability
        momentum: Value for running_mean and running_var computation
        affine: If True, use learnable affine parameters
        track_running_stats: If True, track running mean and variance
    """
    def __init__(self, 
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
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
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', Tensor(np.zeros(num_features)))
            self.register_buffer('running_var', Tensor(np.ones(num_features)))
            self.register_buffer('num_batches_tracked', Tensor(np.array([0])))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    exponential_average_factor = self.momentum

        # Calculate mean and variance for the current batch
        if self.training or not self.track_running_stats:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0, ddof=0)
            n = x.data.shape[0]
        else:
            mean = self.running_mean.data
            var = self.running_var.data
            n = None

        # Update running stats if tracking
        if self.training and self.track_running_stats:
            running_mean = self.running_mean.data
            running_var = self.running_var.data
            running_mean = running_mean * (1 - exponential_average_factor) + mean * exponential_average_factor
            running_var = running_var * (1 - exponential_average_factor) + var * exponential_average_factor
            self.running_mean.data = running_mean
            self.running_var.data = running_var

        # Normalize with fixed std calculation
        x_norm = (x.data - mean) / np.sqrt(var)
        
        # Apply affine transform if specified
        if self.affine:
            # Explicit broadcasting by adding batch dimension
            weight = self.weight.data
            bias = self.bias.data
            x_norm = x_norm * weight + bias

        return Tensor(x_norm, requires_grad=x.requires_grad)

class BatchNorm2d(BatchNorm1d):
    """
    Applies Batch Normalization over a 4D input (batch, channels, height, width)
    
    Args:
        num_features: Number of channels (C from input of size (N,C,H,W))
        eps: Small constant for numerical stability
        momentum: Value for running_mean and running_var computation
        affine: If True, use learnable affine parameters
        track_running_stats: If True, track running mean and variance
    """
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, f'expected 4D input, got {len(x.shape)}D input'
        
        # Reshape input: (N,C,H,W) -> (N*H*W,C)
        N, C, H, W = x.shape
        x_reshaped = x.data.transpose(0, 2, 3, 1).reshape(-1, C)
        x_normed = super().forward(Tensor(x_reshaped, requires_grad=x.requires_grad))
        
        # Reshape back: (N*H*W,C) -> (N,C,H,W)
        return Tensor(x_normed.data.reshape(N, H, W, C).transpose(0, 3, 1, 2),
                     requires_grad=x.requires_grad)