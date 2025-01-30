"""
Core functionality for DLpy.

This module contains the fundamental building blocks of the deep learning library.
"""

from .autograd import AutogradEngine, get_autograd_engine
from .context import Context
from .function import Function
from .module import Module
from .tensor import Tensor

__all__ = ["Tensor", "Function", "Context", "Module", "AutogradEngine", "get_autograd_engine"]
