"""
Graph Pooling Operations.

This module implements various graph pooling strategies for reducing graph size and
learning hierarchical representations. It includes:

    1. TopKPooling: Learns to select the most important nodes.
    2. SAGPooling: Self-Attention Graph Pooling.
    3. EdgePooling: Pools edges rather than nodes.
    4. DiffPooling: Differentiable Dense Graph Pooling.

These pooling operations help in:
    - Reducing computational complexity.
    - Learning hierarchical graph representations.
    - Focusing on the most relevant parts of the graph.
    - Enabling graph classification tasks.
"""

from typing import List, Tuple

import numpy as np

from ....core import Module, Tensor
from ....nn.base.activations import ReLU

# Import your own basic modules.
from ....nn.base.linear import Linear

###############################################################################
# 1. TopKPooling
###############################################################################


class TopKPooling(Module):
    """
    TopKPooling for Graphs.

    This layer learns a score for each node and selects the top fraction of nodes
    based on these scores. The pooled node features are those of the selected nodes.
    The edge index is filtered so that only edges connecting two selected nodes
    are kept.

    Args:
        in_channels (int): Dimensionality of node features.
        ratio (float): Fraction of nodes to keep (0 < ratio <= 1).
    """

    def __init__(self, in_channels: int, ratio: float = 0.5) -> None:
        super().__init__()
        if not (0 < ratio <= 1):
            raise ValueError("ratio must be between 0 and 1.")
        self.in_channels = in_channels
        self.ratio = ratio
        # Linear layer to compute a scalar score for each node.
        self.score_layer = Linear(in_channels, 1)

    def forward(
        self, x: Tensor, edge_index: Tuple[List[int], List[int]]
    ) -> Tuple[Tensor, Tuple[List[int], List[int]]]:
        """
        Args:
            x: Node feature tensor of shape (num_nodes, in_channels).
            edge_index: Tuple (src, tgt) of edge indices.

        Returns:
            A tuple containing:
              - Pooled node features Tensor.
              - New edge_index tuple for the pooled graph.
        """
        num_nodes = x.shape[0]
        # Compute node scores.
        scores = self.score_layer(x)  # shape: (num_nodes, 1)
        scores_np = scores.data.flatten()
        # Determine how many nodes to keep.
        k = int(np.ceil(self.ratio * num_nodes))
        # Get indices for top k nodes.
        topk_indices = np.argsort(-scores_np)[:k].tolist()
        # Pool features by selecting the top nodes.
        pooled_features = Tensor(x.data[topk_indices])
        # Filter edge_index: only keep edges with both endpoints in topk_indices.
        src, tgt = edge_index
        new_src = []
        new_tgt = []
        topk_set = set(topk_indices)
        for s, t in zip(src, tgt):
            if s in topk_set and t in topk_set:
                new_src.append(topk_indices.index(s))
                new_tgt.append(topk_indices.index(t))
        new_edge_index = (new_src, new_tgt)
        return pooled_features, new_edge_index


###############################################################################
# 2. SAGPooling
###############################################################################


class SAGPooling(Module):
    """
    Self-Attention Graph Pooling (SAGPooling).

    SAGPooling computes attention scores for nodes using a learnable transformation
    and a ReLU activation. The top fraction of nodes (as defined by the ratio) is then
    selected, and the edge index is filtered accordingly.

    Args:
        in_channels (int): Dimensionality of node features.
        ratio (float): Fraction of nodes to keep (0 < ratio <= 1).
    """

    def __init__(self, in_channels: int, ratio: float = 0.5) -> None:
        super().__init__()
        if not (0 < ratio <= 1):
            raise ValueError("ratio must be between 0 and 1.")
        self.in_channels = in_channels
        self.ratio = ratio
        self.att_layer = Linear(in_channels, 1)
        self.activation = ReLU()

    def forward(
        self, x: Tensor, edge_index: Tuple[List[int], List[int]]
    ) -> Tuple[Tensor, Tuple[List[int], List[int]]]:
        """
        Args:
            x: Node feature tensor (num_nodes, in_channels).
            edge_index: Tuple (src, tgt) of edge indices.

        Returns:
            A tuple of (pooled node features, new edge_index).
        """
        num_nodes = x.shape[0]
        # Compute attention scores.
        scores = self.att_layer(x)  # shape: (num_nodes, 1)
        scores = self.activation(scores)  # Ensure scores are nonnegative.
        scores_np = scores.data.flatten()
        k = int(np.ceil(self.ratio * num_nodes))
        topk_indices = np.argsort(-scores_np)[:k].tolist()
        pooled_features = Tensor(x.data[topk_indices])
        src, tgt = edge_index
        new_src = []
        new_tgt = []
        topk_set = set(topk_indices)
        for s, t in zip(src, tgt):
            if s in topk_set and t in topk_set:
                new_src.append(topk_indices.index(s))
                new_tgt.append(topk_indices.index(t))
        new_edge_index = (new_src, new_tgt)
        return pooled_features, new_edge_index


###############################################################################
# 3. EdgePooling
###############################################################################


class EdgePooling(Module):
    """
    EdgePooling for Graphs.

    Instead of directly pooling nodes, EdgePooling scores each edge based on the
    difference between node features and selects nodes that participate in high-
    scoring edges. The pooled node features are those from the union of nodes
    involved in the top scoring edges.

    Args:
        in_channels (int): Dimensionality of node features.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.edge_score_layer = Linear(in_channels, 1)
        self.activation = ReLU()

    def forward(
        self, x: Tensor, edge_index: Tuple[List[int], List[int]]
    ) -> Tuple[Tensor, Tuple[List[int], List[int]]]:
        """
        Args:
            x: Node feature tensor of shape (num_nodes, in_channels).
            edge_index: Tuple (src, tgt) of edge indices.

        Returns:
            A tuple containing the pooled node features and new edge index.
        """
        src, tgt = edge_index
        edge_messages = []

        # Compute edge scores
        for s, t in zip(src, tgt):
            diff = x[s] - x[t]
            score = self.edge_score_layer(diff)
            score = self.activation(score)
            edge_messages.append((score.data.item(), s, t))

        # Sort edges by descending score
        sorted_edges = sorted(edge_messages, key=lambda item: -item[0])

        # Create a set for efficient membership testing
        nodes_set = set()  # Type: Set[int]
        for _, s, t in sorted_edges:
            nodes_set.add(s)
            nodes_set.add(t)

        # Convert to sorted list for consistent ordering
        selected_nodes_list = sorted(list(nodes_set))  # Type: List[int]

        # Create the pooled features
        pooled_features = Tensor(x.data[selected_nodes_list])

        # Create new edge indices using the list for indexing
        new_src = []
        new_tgt = []
        for s, t in zip(src, tgt):
            if s in nodes_set and t in nodes_set:  # Use set for efficient lookup
                new_src.append(selected_nodes_list.index(s))  # Use list for indexing
                new_tgt.append(selected_nodes_list.index(t))

        new_edge_index = (new_src, new_tgt)
        return pooled_features, new_edge_index


###############################################################################
# 4. DiffPooling
###############################################################################


class DiffPooling(Module):
    """
    Differentiable Pooling (DiffPooling) for Graphs.

    DiffPooling learns a soft assignment matrix S that maps nodes to clusters.
    The pooled node features are computed as S^T * X, and the graph connectivity is
    coarsened accordingly.

    Args:
        in_channels (int): Dimensionality of input node features.
        assign_dim (int): Number of clusters to pool nodes into.
    """

    def __init__(self, in_channels: int, assign_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.assign_dim = assign_dim
        # Learnable assignment: maps node features to cluster scores.
        self.assign_linear = Linear(in_channels, assign_dim)
        # Optional transformation after pooling.
        self.transform_linear = Linear(in_channels, in_channels)
        self.activation = ReLU()

    def forward(
        self, x: Tensor, edge_index: Tuple[List[int], List[int]]
    ) -> Tuple[Tensor, Tuple[List[int], List[int]]]:
        """
        Args:
            x: Node feature tensor of shape (num_nodes, in_channels).
            edge_index: Tuple (src, tgt) of edge indices.

        Returns:
            A tuple containing:
              - Pooled node features of shape (assign_dim, in_channels).
              - A new edge index for the pooled graph.
        """
        # Compute assignment logits and then soft assignments.
        assign_logits = self.assign_linear(x)  # shape: (num_nodes, assign_dim)
        # Softmax along cluster dimension.
        exp_logits = np.exp(assign_logits.data)
        assign = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
        # Compute pooled features: S^T * X.
        pooled_features = Tensor(np.dot(assign.T, x.data))
        pooled_features = self.activation(self.transform_linear(pooled_features))
        # For simplicity, construct a new edge index as a complete graph among clusters.
        new_src = []
        new_tgt = []
        for i in range(self.assign_dim):
            for j in range(self.assign_dim):
                if i != j:
                    new_src.append(i)
                    new_tgt.append(j)
        new_edge_index = (new_src, new_tgt)
        return pooled_features, new_edge_index
