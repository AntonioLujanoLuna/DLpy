"""
Utils module for DLpy.
"""

from .decomposition import TensorDecomposition
from .utils import calculate_fan_in_fan_out

__all__ = ["calculate_fan_in_fan_out", "TensorDecomposition"]
