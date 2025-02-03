"""
Operations module for DLpy.

This module contains all the mathematical operations that can be performed on tensors.
"""

from .basic import Add, Clip, MatMul, Multiply, Softmax
from .cnn import Conv2dFunction
from .elementwise import Exp, Log
from .loss import (
    BinaryCrossEntropyLoss,
    CosineSimilarityLoss,
    CrossEntropyLoss,
    FocalLoss,
    HingeLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MSELoss,
)
from .matrix import Equal, Greater, GreaterEqual, Less, LessEqual, NotEqual, Transpose
from .pooling import AvgPool2dFunction, MaxPool2dFunction
from .power import Divide, Power
from .reduction import Max, Min, Mean, Sum
from .reshape import Reshape

__all__ = [
    # Basic operations
    "Add",
    "Multiply",
    "MatMul",
    "Reshape",
    "Softmax",
    "Clip",
    # Power operations
    "Power",
    "Divide",
    # Element-wise operations
    "Log",
    "Exp",
    # Reduction operations
    "Sum",
    "Mean",
    "Max",
    "Min",
    # Matrix operations
    "Transpose",
    # Comparison operations
    "Greater",
    "GreaterEqual",
    "Less",
    "LessEqual",
    "Equal",
    "NotEqual",
    # Loss functions
    "MSELoss",
    "CrossEntropyLoss",
    "BinaryCrossEntropyLoss",
    "L1Loss",
    "HuberLoss",
    "KLDivLoss",
    "CosineSimilarityLoss",
    "HingeLoss",
    "FocalLoss",
    # CNN operations
    "Conv2dFunction",
    # Pooling operations
    "MaxPool2dFunction",
    "AvgPool2dFunction",
]
