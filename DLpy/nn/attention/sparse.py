# File: DLpy/nn/attention/sparse.py
from typing import Optional, Tuple

import numpy as np

from ...core import Module, Tensor
from ..base.linear import Linear


class SparseAttention(Module):
    """
    Sparse Attention mechanism.

    Implements block-sparse (local) attention by allowing each position to attend only
    to positions within a fixed window (block_size). This reduces computation for long sequences.

    Args:
        embed_dim (int): Total model dimension.
        num_heads (int): Number of attention heads.
        block_size (int): Local window size for attention.
        dropout (float): Dropout probability (currently not applied).
    """

    def __init__(
        self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.0
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.w_q = Linear(embed_dim, embed_dim)
        self.w_k = Linear(embed_dim, embed_dim)
        self.w_v = Linear(embed_dim, embed_dim)
        self.w_o = Linear(embed_dim, embed_dim)

    def _reshape_for_heads(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        x_reshaped = x.data.reshape(B, L, self.num_heads, self.head_dim)
        x_reshaped = np.transpose(x_reshaped, (0, 2, 1, 3))
        return Tensor(x_reshaped)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute sparse (block-local) attention.

        Args:
            query: Tensor of shape (B, L, D).
            key: Tensor of shape (B, L, D).
            value: Tensor of shape (B, L, D).
            mask: Optional mask tensor.

        Returns:
            A tuple (output, attention_weights) where:
                output: Tensor of shape (B, L, D).
                attention_weights: Tensor of shape (B, num_heads, L, L).
        """
        B, L, D = query.shape
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        q = self._reshape_for_heads(q)  # (B, H, L, head_dim)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        # Scale query.
        scaling = self.head_dim**-0.5
        q = Tensor(q.data * scaling)

        # Compute raw attention scores: (B, H, L, L)
        k_t = Tensor(np.transpose(k.data, (0, 1, 3, 2)))
        scores = Tensor(np.matmul(q.data, k_t.data))

        # Create a block-sparse mask so that each position attends only within a window.
        local_mask = np.full((L, L), -np.inf)
        for i in range(L):
            start = max(0, i - self.block_size)
            end = min(L, i + self.block_size + 1)
            local_mask[i, start:end] = 0.0
        # Expand mask to (1, 1, L, L) and add.
        scores = Tensor(scores.data + local_mask[None, None, :, :])

        if mask is not None:
            scores = Tensor(scores.data + mask.data)

        # Apply softmax over the last dimension.
        scores_exp = np.exp(scores.data - np.max(scores.data, axis=-1, keepdims=True))
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Compute the output as weighted sum over values.
        output = np.matmul(attn_weights, v.data)  # (B, H, L, head_dim)
        output = np.transpose(output, (0, 2, 1, 3)).reshape(B, L, D)
        output = self.w_o(Tensor(output))
        return output, Tensor(attn_weights)
