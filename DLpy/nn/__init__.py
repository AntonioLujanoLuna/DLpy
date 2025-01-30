"""
Neural network module for DLpy.

This module contains all components needed for building neural networks.
"""

from .activations import (
    ELU,
    GELU,
    ELUFunction,
    GELUFunction,
    LeakyReLU,
    LeakyReLUFunction,
    ReLU,
    ReLUFunction,
    Sigmoid,
    SigmoidFunction,
    Tanh,
    TanhFunction,
    elu,
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    tanh,
)
from .batch_norm import BatchNorm1d, BatchNorm2d
from .conv2d import Conv2d
from .dropout import Dropout, Dropout2d
from .layer_norm import LayerNorm
from .linear import Linear
from .normalization import GroupNorm, InstanceNorm2d
from .pooling import AvgPool2d, MaxPool2d
from .rnn import GRU, LSTM, GRUCell, LSTMCell
from .sequential import Sequential
from .transformer import (
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    generate_square_subsequent_mask,
    get_angles,
)

__all__ = [
    # Layers
    "Linear",
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "Dropout",
    "Dropout2d",
    "Sequential",
    # Activation functions
    "relu",
    "leaky_relu",
    "elu",
    "gelu",
    "sigmoid",
    "tanh",
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
    # Convolutional layers
    "Conv2d",
    # Pooling layers
    "MaxPool2d",
    "AvgPool2d",
    # Normalization layers
    "GroupNorm",
    "InstanceNorm2d",
    # RNN layers
    "LSTM",
    "GRU",
    "LSTMCell",
    "GRUCell",
    # Transformer layers
    "MultiHeadAttention",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "PositionalEncoding",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "Transformer",
    # Utility functions
    "generate_square_subsequent_mask",
    "get_angles",
]
