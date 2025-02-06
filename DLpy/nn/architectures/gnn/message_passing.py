"""
Message Passing Framework for Graph Neural Networks.

This module implements the Message Passing Neural Network (MPNN) framework,
which is a general architecture for learning on graphs. The framework consists
of two main phases:
  1. Message passing phase: Nodes exchange messages with their neighbors
  2. Update phase: Node features are updated based on accumulated messages

The implementation follows the mathematical formulation:
    m_{t}^{(j->i)} = M_t(h_i^t, h_j^t, e_{ij})
    m_t^i = Agg_{j ∈ N(i)} ( m_{t}^{(j->i)} )
    h_i^{(t+1)} = U_t( h_i^t, m_t^i )

By default, the message function simply returns the source node’s features,
the aggregation is an element‐wise sum, and the update phase is implemented via a
learnable transformation (if provided) or simply returns the aggregated message.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from ....core import Module, Tensor
from ...base import Linear, ReLU


class DefaultUpdate(Module):
    """
    Default update module for message passing.

    This module implements:
        h_i^(t+1) = ReLU( Linear( [h_i^t || m_i^t] ) )
    where || denotes concatenation.
    It assumes that both h and the aggregated message m have the same feature dimension.

    Args:
        in_channels (int): Feature dimension of h and m.
        out_channels (int): Desired output feature dimension.
                        Often, one may set out_channels = in_channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # The linear transformation maps the concatenated features (size 2*in_channels)
        # to out_channels.
        self.linear = Linear(2 * in_channels, out_channels)
        self.activation = ReLU()

    def forward(self, h: Tensor, m: Tensor) -> Tensor:
        # Concatenate the original node feature and its aggregated message along the
        # feature axis. Assuming h and m have shape (num_nodes, feature_dim),
        # concatenation is along axis=1.
        combined = Tensor(np.concatenate([h.data, m.data], axis=1))
        return Tensor(self.activation(self.linear(combined)))


class MessagePassing(Module):
    """
    Base class for message passing layers in Graph Neural Networks.

    This class implements the general propagation scheme:
      1. For each edge (i,j) the message is computed via:
             m_t^(j->i) = M_t(h_i^t, h_j^t, e_{ij})
      2. Messages are aggregated for each target node:
             m_t^i = Agg_{j ∈ N(i)}( m_t^(j->i) )
      3. Node features are updated:
             h_i^(t+1) = U_t( h_i^t, m_t^i )

    The update function U_t is provided via an update module. If no update module is
    provided upon instantiation, then update(x, m) returns m.

    Attributes:
        aggregate_fn: A callable to aggregate a list of messages (default: sum).
        update_module: A Module that implements the update function U_t.
    """

    def __init__(
        self,
        aggregate_fn: Optional[Callable[[List[Tensor]], Tensor]] = None,
        update_module: Optional[Module] = None,
    ) -> None:
        super().__init__()
        self.aggregate_fn = (
            aggregate_fn if aggregate_fn is not None else self._sum_aggregate
        )
        self.update_module = update_module

    def _sum_aggregate(self, messages: List[Tensor]) -> Tensor:
        """
        Aggregates a list of messages by summing them element–wise.

        Args:
            messages: A list of Tensor objects (messages from neighbors).

        Returns:
            A Tensor that is the element–wise sum of the messages.
        """
        # Assumes messages is non-empty; this method is used when at least one
        # message exists.
        result = messages[0]
        for msg in messages[1:]:
            result = result + msg
        return result

    def message(
        self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        """
        Computes the message from node j to node i.

        Args:
            x_i: Feature of the target node.
            x_j: Feature of the source node.
            edge_attr: Optional edge attribute tensor.

        Returns:
            A Tensor representing the message.
        """
        return x_j  # Default: simply return the source node's feature.

    def aggregate(
        self, messages: List[Tensor], index: List[int], num_nodes: int
    ) -> List[Tensor]:
        """
        Aggregates messages for each node based on target indices.

        Args:
            messages: A list of message Tensors.
            index: A list of integers representing the target node for each message.
            num_nodes: Total number of nodes in the graph.

        Returns:
            A list (length=num_nodes) where each element is the aggregated message for
                that node.
        """
        # Determine the feature dimension from the first message if available.
        # If there are no messages, we cannot infer the message shape.
        default_shape = messages[0].data.shape if messages else None

        # If no messages were sent, we will later handle aggregation in propagate.
        aggregated = []
        for _ in range(num_nodes):
            if default_shape is not None:
                aggregated.append(Tensor(np.zeros(default_shape, dtype=np.float64)))
            else:
                # Fallback: use a scalar zero.
                aggregated.append(Tensor(0.0))
        for msg, i in zip(messages, index):
            aggregated[i] = aggregated[i] + msg
        return aggregated

    def update(self, h: Tensor, m: Tensor) -> Tensor:
        """
        Updates node features based on the original feature and the aggregated message.

        Args:
            h: The original node feature tensor.
            m: The aggregated message tensor.

        Returns:
            A Tensor representing the updated node feature.
        """
        if self.update_module is not None:
            return Tensor(self.update_module(h, m))
        return m

    def propagate(
        self,
        edge_index: Tuple[List[int], List[int]],
        x: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Performs the full propagation step for the graph.

        Args:
            edge_index: Tuple (source_indices, target_indices) for all edges.
            x: Node feature tensor of shape (num_nodes, feature_dim).
            edge_attr: Optional edge attributes.

        Returns:
            A Tensor containing the updated node features.
        """
        src, tgt = edge_index
        messages = []
        index = []
        # Compute messages for each edge.
        for i, j in zip(tgt, src):
            msg = self.message(x_i=x[i], x_j=x[j], edge_attr=edge_attr)
            messages.append(msg)
            index.append(i)
        num_nodes = x.shape[0]
        if messages:
            aggr_messages_list = self.aggregate(messages, index, num_nodes)
            # Ensure that each aggregated message is a NumPy array of the same shape.
            aggr_messages = Tensor(np.stack([m.data for m in aggr_messages_list]))
        else:
            # If no messages were computed, return a zero tensor with shape
            # (num_nodes, feature_dim)
            aggr_messages = Tensor(
                np.zeros((num_nodes, x.data.shape[1]), dtype=x.data.dtype)
            )
        return self.update(x, aggr_messages)
