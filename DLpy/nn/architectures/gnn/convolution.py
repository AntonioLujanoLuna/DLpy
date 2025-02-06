"""
Graph Convolution Operations.

This module implements various types of graph convolutions, including:
  1. Graph Convolutional Network (GCN) layer
  2. Graph Attention Network (GAT) layer
  3. GraphSAGE convolution
  4. EdgeConv for geometric deep learning

The implementations support both spectral and spatial approaches to graph convolutions,
handling different types of graph structures and feature learning requirements.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ....core import Tensor
from ....nn.base.activations import LeakyReLU, ReLU

# Import our own linear and activation modules from DLpy
from ....nn.base.linear import Linear
from .message_passing import DefaultUpdate, MessagePassing

###############################################################################
# 1. Graph Convolutional Network (GCN) Layer
###############################################################################


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network (GCN) Layer.

    This layer computes:
        h_i' = (A_hat * x)_i * W
    where A_hat is the (optionally normalized) adjacency matrix with self-loops.

    Args:
        in_channels (int): Dimensionality of input node features.
        out_channels (int): Dimensionality of output node features.
        add_self_loops (bool): Whether to add self-loops (default: True).
        normalize (bool): Whether to apply symmetric normalization (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_self_loops: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        # Weight matrix: shape (in_channels, out_channels)
        self.weight = Tensor(
            np.random.randn(in_channels, out_channels), requires_grad=True
        )

    def forward(self, x: Tensor, edge_index: Tuple[List[int], List[int]]) -> Tensor:
        src, tgt = edge_index
        num_nodes = x.shape[0]
        # Optionally add self-loops
        if self.add_self_loops:
            self_loops = (list(range(num_nodes)), list(range(num_nodes)))
            src = src + self_loops[0]
            tgt = tgt + self_loops[1]

        # Compute degree for normalization if needed.
        if self.normalize:
            degree = np.zeros(num_nodes)
            for i in tgt:
                degree[i] += 1
            inv_sqrt_degree = 1.0 / np.sqrt(degree + 1e-10)
        else:
            inv_sqrt_degree = None

        messages = []
        index = []
        for i, j in zip(tgt, src):
            x_j = x[j]
            if self.normalize:
                norm = inv_sqrt_degree[i] * inv_sqrt_degree[j]
                msg = x_j * norm
            else:
                msg = x_j
            messages.append(msg)
            index.append(i)

        # Aggregate messages via simple sum.
        aggr_list = [Tensor(0.0) for _ in range(num_nodes)]
        for msg, i in zip(messages, index):
            aggr_list[i] = aggr_list[i] + msg
        aggr = Tensor(np.stack([m.data for m in aggr_list]))
        # Apply learnable weight matrix.
        out = aggr @ self.weight
        return out


###############################################################################
# 2. Graph Attention Network (GAT) Layer
###############################################################################


class GATConv(MessagePassing):
    """
    Graph Attention Network (GAT) Layer.

    This layer first linearly transforms node features and then computes for each edge
    an attention coefficient based on the concatenation of the source and target
    features.
    For each edge (j -> i):
        e_ij = LeakyReLU( a^T [h_i || h_j] )
    The attention coefficients are normalized via a softmax over the neighbors of i.
    Finally, the updated node feature is computed as the weighted sum (over heads) of
    the transformed neighbor features.

    Args:
        in_channels (int): Dimensionality of input features.
        out_channels (int): Dimensionality of output features per head.
        heads (int): Number of attention heads (default: 1).
        add_self_loops (bool): Whether to add self-loops (default: True).
        negative_slope (float): Negative slope for LeakyReLU (default: 0.2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        add_self_loops: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope
        # Weight transformation: maps input to (heads * out_channels)
        self.weight = Tensor(
            np.random.randn(in_channels, heads * out_channels), requires_grad=True
        )
        # Attention parameters: one vector per head of shape (2*out_channels,)
        self.att = Tensor(np.random.randn(heads, 2 * out_channels), requires_grad=True)
        self.leaky_relu = LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: Tensor, edge_index: Tuple[List[int], List[int]]) -> Tensor:
        num_nodes = x.shape[0]
        src, tgt = edge_index
        # Apply linear transformation.
        h = x @ self.weight  # shape: (num_nodes, heads*out_channels)
        # Reshape to (num_nodes, heads, out_channels)
        h = Tensor(h.data.reshape(num_nodes, self.heads, self.out_channels))
        # Optionally add self-loops.
        if self.add_self_loops:
            self_loops = (list(range(num_nodes)), list(range(num_nodes)))
            src = src + self_loops[0]
            tgt = tgt + self_loops[1]

        # Compute raw attention scores for each edge and each head.
        messages = []
        index = []
        for i, j in zip(tgt, src):
            h_i = h[i]  # shape: (heads, out_channels)
            h_j = h[j]  # shape: (heads, out_channels)
            # Concatenate h_i and h_j along feature dimension: (heads, 2*out_channels)
            concat = Tensor(np.concatenate([h_i.data, h_j.data], axis=1))
            # Compute attention scores per head.
            e = []
            for head in range(self.heads):
                att_vector = self.att.data[head]  # shape: (2*out_channels,)
                score = np.dot(att_vector, concat.data[head])
                # Apply LeakyReLU activation.
                activated = self.leaky_relu.forward(Tensor(score)).data
                e.append(activated)
            # e has shape (heads,)
            e = np.array(e)
            # Store both the raw scores and the corresponding neighbor feature.
            messages.append((Tensor(e), h_j))
            index.append(i)

        # Group messages by target node.
        messages_by_node: Dict[int, List[Tuple[Tensor, Tensor]]] = {
            i: [] for i in range(num_nodes)
        }

        for (score, feat), i in zip(messages, index):
            messages_by_node[i].append((score, feat))
        # For each node and each head, perform softmax normalization.
        aggregated_list = []
        for i in range(num_nodes):
            if not messages_by_node[i]:
                # No incoming message; use zeros.
                aggregated_list.append(np.zeros((self.heads, self.out_channels)))
                continue
            # Stack scores: shape (num_messages, heads)
            scores = np.stack([item[0].data for item in messages_by_node[i]], axis=0)
            # Stack features: shape (num_messages, heads, out_channels)
            feats = np.stack([item[1].data for item in messages_by_node[i]], axis=0)
            # For each head, compute softmax over the messages.
            agg_node = np.zeros((self.heads, self.out_channels))
            for head in range(self.heads):
                head_scores = scores[:, head]
                exp_scores = np.exp(head_scores - np.max(head_scores))
                softmax_coef = exp_scores / (np.sum(exp_scores) + 1e-10)
                # Multiply each message feature by its coefficient and sum.
                for idx in range(softmax_coef.shape[0]):
                    agg_node[head] += softmax_coef[idx] * feats[idx, head]
            aggregated_list.append(agg_node)
        # Stack aggregated messages: shape (num_nodes, heads, out_channels)
        aggr_tensor = Tensor(np.stack(aggregated_list, axis=0))
        # You may choose to concatenate the heads.
        out = aggr_tensor.data.reshape(num_nodes, self.heads * self.out_channels)
        return Tensor(out)


###############################################################################
# 3. GraphSAGE Convolution
###############################################################################


class GraphSAGEConv(MessagePassing):
    """
    GraphSAGE Convolution Layer.

    This layer aggregates neighbor features (using the default sum or a provided
    aggregator) and then updates node features via a learnable transformation.
    The update is defined as:
        h_i' = ReLU( Linear( [h_i || aggr] ) )
    where || denotes concatenation.

    Args:
        in_channels (int): Dimensionality of input node features.
        out_channels (int): Dimensionality of output node features.
        aggregator (Callable, optional): Custom aggregation function; if None, sum
            aggregation is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregator: Optional[Callable[[List[Tensor]], Tensor]] = None,
    ) -> None:
        # For the update phase we use our DefaultUpdate which concatenates x and the
        # aggregated message.
        update_module = DefaultUpdate(
            in_channels=in_channels, out_channels=out_channels
        )
        super().__init__(aggregate_fn=aggregator, update_module=update_module)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def message(
        self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        # For GraphSAGE, the message is simply the neighbor feature.
        return x_j

    def forward(self, x: Tensor, edge_index: Tuple[List[int], List[int]]) -> Tensor:
        return self.propagate(edge_index, x)


###############################################################################
# 4. EdgeConv Layer
###############################################################################


class EdgeConv(MessagePassing):
    """
    Edge Convolution Layer.

    In this layer, for each edge (i, j) the message is computed as:
        m_{ij} = ReLU( Linear( [h_i || (h_j - h_i)] ) )
    Then, the messages for each node are aggregated using a max-pooling operator.

    Args:
        in_channels (int): Dimensionality of input node features.
        out_channels (int): Dimensionality of output node features.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Linear layer maps concatenated features (2*in_channels) to out_channels
        self.linear = Linear(2 * in_channels, out_channels)
        self.activation = ReLU()

    def message(
        self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        # Compute the difference between neighbor and center features
        # Both x_i and x_j have shape (in_channels,)
        diff = x_j - x_i

        # Concatenate features: [h_i || (h_j - h_i)]
        # Result shape: (2 * in_channels,)
        concatenated = Tensor(np.concatenate([x_i.data, diff.data]))

        # Apply linear transformation and activation
        # Output shape: (out_channels,)
        transformed = self.linear(concatenated)
        return Tensor(self.activation(transformed))

    def aggregate(
        self, messages: List[Tensor], index: List[int], num_nodes: int
    ) -> List[Tensor]:
        # Initialize output tensors with proper shape
        # Each message has shape (out_channels,)
        if not messages:
            # Handle empty message case
            return [Tensor(np.zeros(self.out_channels)) for _ in range(num_nodes)]

        # Initialize with negative infinity for max pooling
        aggregated = [
            Tensor(np.full(self.out_channels, -np.inf)) for _ in range(num_nodes)
        ]

        # Perform max pooling over messages
        for msg, i in zip(messages, index):
            msg_data = msg.data
            # Ensure msg_data is 1D
            if len(msg_data.shape) > 1:
                msg_data = msg_data.reshape(-1)
            aggregated[i] = Tensor(np.maximum(aggregated[i].data, msg_data))

        # Replace any remaining -inf values with zeros
        for i in range(num_nodes):
            aggregated[i] = Tensor(
                np.where(aggregated[i].data == -np.inf, 0, aggregated[i].data)
            )

        return aggregated

    def forward(self, x: Tensor, edge_index: Tuple[List[int], List[int]]) -> Tensor:
        # Propagate messages
        out = self.propagate(edge_index, x)
        # Stack the list of tensors into a single tensor with shape
        # (num_nodes, out_channels)
        return Tensor(np.stack([o.data for o in out]))
