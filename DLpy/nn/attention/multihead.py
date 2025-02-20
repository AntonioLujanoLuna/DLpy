from typing import Optional, Tuple

import numpy as np

from ...core import Module, Tensor
from ..base.dropout import Dropout
from ..base.linear import Linear


class MultiHeadAttention(Module):
    """
    Multi-head attention mechanism with optional KV caching for autoregressive decoding.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
        has_bias (bool): If True, use bias in linear layers.
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
                f"Embedding dimension {embed_dim} not divisible by num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.w_q = Linear(embed_dim, embed_dim, has_bias=has_bias)
        self.w_k = Linear(embed_dim, embed_dim, has_bias=has_bias)
        self.w_v = Linear(embed_dim, embed_dim, has_bias=has_bias)
        self.w_o = Linear(embed_dim, embed_dim, has_bias=has_bias)

        self.attention_dropout = Dropout(self.dropout)

    def _reshape_for_heads(self, x: Tensor) -> Tensor:
        """
        Reshapes input tensor for parallel head processing.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _concat_cache(self, past: Tensor, current: Tensor) -> Tensor:
        """
        Concatenates cached and current tensors along the sequence dimension.

        Args:
            past (Tensor): Cached tensor of shape (batch_size, num_heads, cached_len, head_dim)
            current (Tensor): New tensor of shape (batch_size, num_heads, cur_len, head_dim)

        Returns:
            Tensor: Concatenated tensor of shape (batch_size, num_heads, cached_len + cur_len, head_dim)
        """
        # Assuming Tensor.data holds the underlying NumPy array.
        concatenated = np.concatenate((past.data, current.data), axis=2)
        return Tensor(concatenated)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key: Optional[Tensor] = None,
        past_value: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass of multi-head attention with optional KV caching.

        Args:
            query (Tensor): Query tensor of shape (batch_size, query_len, embed_dim)
            key (Tensor): Key tensor of shape (batch_size, key_len, embed_dim)
            value (Tensor): Value tensor of shape (batch_size, key_len, embed_dim)
            attention_mask (Optional[Tensor]): Mask tensor of shape (batch_size, num_heads, query_len, key_len_total)
            past_key (Optional[Tensor]): Cached key tensor of shape (batch_size, num_heads, cached_len, head_dim)
            past_value (Optional[Tensor]): Cached value tensor of shape (batch_size, num_heads, cached_len, head_dim)
            use_cache (bool): Flag to indicate whether to return updated KV cache.

        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, query_len, embed_dim)
                - Attention weights of shape (batch_size, num_heads, query_len, key_len_total)
                - Updated KV cache tuple (updated_key, updated_value) if use_cache is True, else None.
        """
        # Compute linear projections
        q = self.w_q(query)  # (batch_size, query_len, embed_dim)
        k = self.w_k(key)  # (batch_size, key_len, embed_dim)
        v = self.w_v(value)  # (batch_size, key_len, embed_dim)

        # Reshape for multiple heads
        q = self._reshape_for_heads(q)  # (batch_size, num_heads, query_len, head_dim)
        k = self._reshape_for_heads(k)  # (batch_size, num_heads, key_len, head_dim)
        v = self._reshape_for_heads(v)  # (batch_size, num_heads, key_len, head_dim)

        # Scale query
        q = q * self.scaling

        # If cached keys/values are provided, concatenate them with current keys/values
        if past_key is not None and past_value is not None:
            k = self._concat_cache(past_key, k)
            v = self._concat_cache(past_value, v)

        # Compute attention scores
        k_t = k.transpose(
            0, 1, 3, 2
        )  # (batch_size, num_heads, head_dim, key_len_total)
        attention_scores = q @ k_t  # (batch_size, num_heads, query_len, key_len_total)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Numerical stabilization
        attention_scores = attention_scores - attention_scores.max(
            axis=-1, keepdims=True
        )
        attention_scores = attention_scores.clip(-1e30, 1e30)

        # Compute softmax to obtain attention weights
        attention_weights = attention_scores.softmax(dim=-1)

        if self.training:
            attention_weights = self.attention_dropout(attention_weights)

        # Apply attention weights to values
        output = attention_weights @ v  # (batch_size, num_heads, query_len, head_dim)

        # Reshape back to original dimensions
        output = output.transpose(0, 2, 1, 3)
        batch_size, query_len, _, _ = output.shape
        output = output.reshape(batch_size, query_len, self.embed_dim)

        # Final linear projection
        output = self.w_o(output)

        updated_cache = (k, v) if use_cache else None
        return output, attention_weights, updated_cache
