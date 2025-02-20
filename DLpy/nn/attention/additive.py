# File: DLpy/nn/attention/additive.py
from typing import Optional, Tuple

import numpy as np

from ...core import Module, Tensor
from ..base.linear import Linear


class AdditiveAttention(Module):
    """
    Additive (Bahdanau) Attention mechanism.

    Computes attention scores using a feedforward network over the query and key.

    Args:
        query_dim (int): Dimension of the query vector.
        key_dim (int): Dimension of the key vector.
        hidden_dim (int): Dimension of the hidden layer for score computation.
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        # Transformations for query and key.
        self.linear_query = Linear(query_dim, hidden_dim)
        self.linear_key = Linear(key_dim, hidden_dim)
        # Scoring vector (no bias).
        self.v = Linear(hidden_dim, 1, has_bias=False)

    def forward(
        self, query: Tensor, keys: Tensor, values: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute additive attention.

        Args:
            query: Tensor of shape (batch_size, query_len, query_dim).
            keys: Tensor of shape (batch_size, key_len, key_dim).
            values: Tensor of shape (batch_size, key_len, value_dim).
            mask: Optional mask tensor of shape (batch_size, query_len, key_len), where masked positions have large negative values.

        Returns:
            A tuple (context, attention_weights):
                context: Tensor of shape (batch_size, query_len, value_dim), the attended representation.
                attention_weights: Tensor of shape (batch_size, query_len, key_len), the attention distribution.
        """
        # Transform query and keys.
        query_transformed = self.linear_query(query)  # (B, Q, H)
        keys_transformed = self.linear_key(keys)  # (B, K, H)

        # Expand dimensions to combine query and key.
        # query_expanded: (B, Q, 1, H); keys_expanded: (B, 1, K, H)
        query_expanded = query_transformed.data[:, :, None, :]
        keys_expanded = keys_transformed.data[:, None, :, :]
        # Compute combined representation and apply non-linearity.
        combined = np.tanh(query_expanded + keys_expanded)  # (B, Q, K, H)
        # Compute scores via the final linear layer.
        scores = self.v(Tensor(combined)).data.squeeze(-1)  # (B, Q, K)

        if mask is not None:
            scores = (
                scores + mask.data
            )  # Add mask (assumed to have -inf in masked positions)

        # Numerically stable softmax along the key dimension.
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Compute context as weighted sum of values.
        context = np.matmul(attention_weights, values.data)  # (B, Q, value_dim)
        return Tensor(context), Tensor(attention_weights)
