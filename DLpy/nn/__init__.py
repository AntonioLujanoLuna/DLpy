"""
Neural network module for DLpy.

This module contains all components needed for building neural networks.
"""

from .linear import Linear
from .batch_norm import BatchNorm1d, BatchNorm2d
from .layer_norm import LayerNorm
from .dropout import Dropout, Dropout2d
from .sequential import Sequential
from .activations import (
    relu, leaky_relu, elu, gelu, sigmoid, tanh,
    ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh,
    ReLUFunction, LeakyReLUFunction, ELUFunction, GELUFunction, SigmoidFunction, TanhFunction
)
from .conv2d import Conv2d 
from .pooling import MaxPool2d, AvgPool2d
from .normalization import GroupNorm, InstanceNorm2d
from .rnn import LSTM, GRU, LSTMCell, GRUCell
from .transformer import (
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerEncoder,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
    Transformer,
    generate_square_subsequent_mask
)

__all__ = [
    # Layers
    'Linear',
    'BatchNorm1d',
    'BatchNorm2d',
    'LayerNorm',
    'Dropout',
    'Dropout2d',
    'Sequential',
    
    # Activation functions
    'relu',
    'leaky_relu',
    'elu',
    'gelu',
    'sigmoid',
    'tanh',
    'ReLU',
    'LeakyReLU',
    'ELU',
    'GELU',
    'Sigmoid',
    'Tanh',
    'ReLUFunction',
    'LeakyReLUFunction',
    'ELUFunction',
    'GELUFunction',
    'SigmoidFunction',
    'TanhFunction',

    # Convolutional layers
    'Conv2d',

    # Pooling layers
    'MaxPool2d',
    'AvgPool2d',

    # Normalization layers
    'GroupNorm',
    'InstanceNorm2d',
    
    # RNN layers
    'LSTM',
    'GRU',
    'LSTMCell',
    'GRUCell',

    # Transformer layers
    'MultiHeadAttention',
    'TransformerEncoderLayer',
    'TransformerEncoder',
    'PositionalEncoding',
    'TransformerDecoderLayer',
    'TransformerDecoder',
    'Transformer',

    # Utility functions
    'generate_square_subsequent_mask'
]