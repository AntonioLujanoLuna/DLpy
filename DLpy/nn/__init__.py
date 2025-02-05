# DLpy/nn/__init__.py
"""
Neural network module for DLpy.

This module contains all components needed for building neural networks.
Organized into submodules for different types of neural network architectures
and components.
"""

# Import base modules
from .base.activations import (
    ELU, GELU, LeakyReLU, ReLU, Sigmoid, Tanh,
    ELUFunction, GELUFunction, LeakyReLUFunction, ReLUFunction, SigmoidFunction, TanhFunction,
    relu, leaky_relu, elu, gelu, sigmoid, tanh
)
from .base.conv2d import Conv2d
from .base.dropout import Dropout, Dropout2d
from .base.linear import Linear
from .base.pooling import AvgPool2d, MaxPool2d
from .base.sequential import Sequential

# Import normalization modules
from .norm.batch_norm import BatchNorm1d, BatchNorm2d
from .norm.layer_norm import LayerNorm
from .norm.group_norm import GroupNorm
from .norm.instance_norm import InstanceNorm2d

# Import attention modules
from .attention.multihead import MultiHeadAttention
from .attention.utils import get_angles

# Import architectures
## Transformers
from .architectures.transformers.base import Transformer, generate_square_subsequent_mask
from .architectures.transformers.encoder import TransformerEncoder, TransformerEncoderLayer
from .architectures.transformers.decoder import TransformerDecoder, TransformerDecoderLayer
from .architectures.transformers.embedding import PositionalEncoding

## RNN
from .architectures.rnn.base import LSTM, GRU, LSTMCell, GRUCell

## Graph Neural Networks (placeholder for future)
# from .architectures.gnn import *

## Neural ODEs (placeholder for future)
# from .architectures.ode import *

## State Space Models (placeholder for future)
# from .architectures.state_space import *

# Define the public API
__all__ = [
    # Base Modules
    "Linear",
    "Conv2d",
    "Dropout",
    "Dropout2d",
    "Sequential",
    
    # Activations
    "ReLU",
    "LeakyReLU",
    "ELU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "ReLUFunction",
    "LeakyReLUFunction",
    "ELUFunction",
    "GELUFunction",
    "SigmoidFunction",
    "TanhFunction",
    "relu",
    "leaky_relu",
    "elu",
    "gelu",
    "sigmoid",
    "tanh",
    
    # Normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm2d",
    
    # Pooling
    "MaxPool2d",
    "AvgPool2d",
    
    # Attention
    "MultiHeadAttention",
    "get_angles",
    
    # Transformers
    "Transformer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "PositionalEncoding",
    "generate_square_subsequent_mask",
    
    # RNN
    "LSTM",
    "GRU",
    "LSTMCell",
    "GRUCell",
]

# DLpy/nn/architectures/__init__.py
"""
Neural network architectures submodule.
Contains implementations of various neural network architectures.
"""

from .transformers import *
from .rnn import *
# Future imports for GNN, ODE, etc.

# DLpy/nn/attention/__init__.py
"""
Attention mechanisms submodule.
Contains various attention implementations and utilities.
"""

from .multihead import MultiHeadAttention
from .utils import get_angles
from .linear import *  # When implemented

__all__ = [
    "MultiHeadAttention",
    "get_angles",
]

# DLpy/nn/base/__init__.py
"""
Base neural network components.
Contains fundamental building blocks for neural networks.
"""

from .activations import *
from .conv2d import Conv2d
from .dropout import Dropout, Dropout2d
from .linear import Linear
from .pooling import AvgPool2d, MaxPool2d
from .sequential import Sequential

__all__ = [
    "Conv2d",
    "Dropout",
    "Dropout2d",
    "Linear",
    "Sequential",
    "AvgPool2d",
    "MaxPool2d",
]

# DLpy/nn/norm/__init__.py
"""
Normalization layers submodule.
Contains various normalization implementations.
"""

from .batch_norm import BatchNorm1d, BatchNorm2d
from .layer_norm import LayerNorm
from .group_norm import GroupNorm
from .instance_norm import InstanceNorm2d

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm2d",
]