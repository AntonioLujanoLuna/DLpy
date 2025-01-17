"""
Operations module for DLpy.

This module contains all the mathematical operations that can be performed on tensors.
"""

from .basic import Add, Multiply
from .reshape import Reshape
from .power import Power, Divide
from .elementwise import Log, Exp
from .reduction import Sum, Mean, Max
from .matrix import (
    Transpose,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual
)
from .loss import (
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    L1Loss,
    HuberLoss,
    KLDivLoss,
    CosineSimilarityLoss,
    HingeLoss,
    FocalLoss
)

__all__ = [
    # Basic operations
    'Add',
    'Multiply',
    'Reshape',
    
    # Power operations
    'Power',
    'Divide',
    
    # Element-wise operations
    'Log',
    'Exp',
    
    # Reduction operations
    'Sum',
    'Mean',
    'Max',
    
    # Matrix operations
    'Transpose',
    
    # Comparison operations
    'Greater',
    'GreaterEqual',
    'Less',
    'LessEqual',
    'Equal',
    'NotEqual',

    # Loss functions
    'MSELoss',
    'CrossEntropyLoss',
    'BinaryCrossEntropyLoss',
    'L1Loss',
    'HuberLoss',
    'KLDivLoss',
    'CosineSimilarityLoss',
    'HingeLoss',
    'FocalLoss'
]