"""
Graph Neural Network (GNN) module.

This module provides implementations of various Graph Neural Network components,
including convolution operations, pooling layers, and message passing mechanisms.
The implementations support both spatial and spectral approaches to graph learning.
"""

from .message_passing import MessagePassing, DefaultUpdate
from .convolution import (
    GCNConv,
    GATConv,
    GraphSAGEConv,
    EdgeConv,
)
from .pooling import (
    TopKPooling,
    SAGPooling,
    EdgePooling,
    DiffPooling,
)

__all__ = [
    # Base Classes
    "MessagePassing",
    # Convolution Operations
    "GCNConv",
    "GATConv",
    "GraphSAGEConv",
    "EdgeConv",
    # Pooling Operations
    "TopKPooling",
    "SAGPooling",
    "EdgePooling",
    "DiffPooling",
]
