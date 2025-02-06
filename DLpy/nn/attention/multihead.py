from typing import Optional, Tuple

from ...core import Module, Tensor
from ..base.dropout import Dropout
from ..base.linear import Linear


class MultiHeadAttention(Module):
    """
    Multi-head attention mechanism.

    This module splits the input into multiple heads, applies scaled dot-product
    attention independently on each head, and then concatenates the results.

    Args:
        embed_dim (int): Total dimension of the model
        num_heads (int): Number of parallel attention heads
        dropout (float): Dropout probability (optional)
        bias (bool): If True, use bias in linear layers
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        has_bias: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} not divisible "
                f"by num_heads {num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.w_q = Linear(
            embed_dim, embed_dim, has_bias=has_bias
        )  # (N, L, E) -> (N, L, E)
        self.w_k = Linear(
            embed_dim, embed_dim, has_bias=has_bias
        )  # (N, S, E) -> (N, S, E)
        self.w_v = Linear(
            embed_dim, embed_dim, has_bias=has_bias
        )  # (N, S, E) -> (N, S, E)
        self.w_o = Linear(
            embed_dim, embed_dim, has_bias=has_bias
        )  # (N, L, E) -> (N, L, E)

        self.attention_dropout = Dropout(self.dropout)

    def _reshape_for_heads(self, x: Tensor) -> Tensor:
        """Reshapes input for parallel head processing."""
        batch_size, seq_len, _ = x.shape
        # First reshape to separate head dimensions
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # Then transpose dimensions to (batch_size, num_heads, seq_len, head_dim)
        x = x.transpose(0, 2, 1, 3)  # Note: passing axes as a tuple
        return x

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, query_len, embed_dim)
            key: Key tensor of shape (batch_size, key_len, embed_dim)
            value: Value tensor of shape (batch_size, key_len, embed_dim)
            attention_mask: Optional mask tensor of shape (batch_size, num_heads,
                query_len, key_len)

        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, query_len, embed_dim)
                - Attention weights of shape (batch_size, num_heads, query_len, key_len)
        """
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape

        # Linear projections for Q, K, V
        q = self.w_q(query)  # (batch_size, query_len, embed_dim)
        k = self.w_k(key)  # (batch_size, key_len, embed_dim)
        v = self.w_v(value)  # (batch_size, key_len, embed_dim)

        # Reshape for multi-head attention
        q = self._reshape_for_heads(q)  # (batch_size, num_heads, query_len, head_dim)
        k = self._reshape_for_heads(k)  # (batch_size, num_heads, key_len, head_dim)
        v = self._reshape_for_heads(v)  # (batch_size, num_heads, key_len, head_dim)

        # Scale query
        q = q * self.scaling  # Scale by sqrt(head_dim)

        # Compute attention scores
        # Transpose key for matrix multiplication
        k_t = k.transpose(0, 1, 3, 2)  # (batch_size, num_heads, head_dim, key_len)
        attention_scores = q @ k_t  # (batch_size, num_heads, query_len, key_len)

        # Attention mask handling
        if attention_mask is not None:
            # Apply mask first - this ensures masked positions stay masked
            attention_scores = attention_scores + attention_mask

        # Apply numerical stabilization after masking
        # This prevents overflow while maintaining masked positions
        attention_scores = attention_scores - attention_scores.max(
            axis=-1, keepdims=True
        )

        # Clip values for additional numerical stability
        attention_scores = attention_scores.clip(-1e30, 1e30)

        # Apply softmax to get attention weights
        attention_weights = attention_scores.softmax(dim=-1)

        # Apply dropout during training
        if self.training:
            attention_weights = self.attention_dropout(attention_weights)

        # Apply attention weights to values
        output = attention_weights @ v  # (batch_size, num_heads, query_len, head_dim)

        # Reshape back to original dimensions
        # First transpose to get heads dimension next to head_dim
        output = output.transpose(
            0, 2, 1, 3
        )  # (batch_size, query_len, num_heads, head_dim)
        # Then combine heads with head_dim to get back to embed_dim
        output = output.reshape(batch_size, query_len, self.embed_dim)

        # Final linear projection
        output = self.w_o(output)  # (batch_size, query_len, embed_dim)

        return output, attention_weights


# TODO LinearAttention, FlashAttention
