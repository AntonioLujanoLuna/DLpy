"""
Utils module for DLpy.
"""

from .utils import calculate_fan_in_fan_out
from .decomposition import TensorDecomposition

__all__ = ['calculate_fan_in_fan_out',
           'TensorDecomposition'
           ]