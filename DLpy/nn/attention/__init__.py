# DLpy/nn/attention/__init__.py
"""
Attention mechanisms submodule.
Contains various attention implementations and utilities.
"""

from .multihead import MultiHeadAttention
from .utils import get_angles

# from .linear import *  # When implemented

__all__ = [
    "MultiHeadAttention",
    "get_angles",
]
