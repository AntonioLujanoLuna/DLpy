# DLpy/nn/__init__.py
"""
Neural network module for DLpy.

This module contains all components needed for building neural networks.
Organized into submodules for different types of neural network architectures
and components.
"""

# Import base modules
from .base.activations import (
    ELU,
    GELU,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Tanh,
    ELUFunction,
    GELUFunction,
    LeakyReLUFunction,
    ReLUFunction,
    SigmoidFunction,
    TanhFunction,
    relu,
    leaky_relu,
    elu,
    gelu,
    sigmoid,
    tanh,
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
# Transformers
from .architectures.transformers.base import (
    Transformer,
    generate_square_subsequent_mask,
)
from .architectures.transformers.encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .architectures.transformers.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from .architectures.transformers.embedding import PositionalEncoding

# RNN
from .architectures.rnn.base import LSTM, GRU, LSTMCell, GRUCell

# Graph Neural Networks (placeholder for future)
# from .architectures.gnn import *

# Neural ODEs (placeholder for future)
# from .architectures.ode import *

# State Space Models (placeholder for future)
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
