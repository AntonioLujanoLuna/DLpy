"""
DLpy: A Deep Learning Library with DAG-based Autograd

This library provides a PyTorch-like interface for building and training neural networks,
with a focus on clear implementation and educational value.
"""

from .core import Tensor, Function, Context
from .ops import Add, Multiply, Reshape

__version__ = "0.1.0"

__all__ = [
    'Tensor',
    'Function',
    'Context',
    'Add',
    'Multiply',
    'Reshape',
]