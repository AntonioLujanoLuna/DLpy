from typing import Optional, Tuple

import numpy as np

from ...core import Module, Tensor
from ...nn.base.activations import elu
from ..base.linear import Linear


class LinearAttention(Module):
    """
    Linear Attention mechanism.

    Approximates softmax attention using a kernel feature map (f(x) = ELU(x)+1)
    to achieve linear time complexity in the sequence length.

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability (not applied in this implementation).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
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
        Compute linear attention.

        Args:
            query: Tensor of shape (B, L, D)
            key: Tensor of shape (B, L, D)
            value: Tensor of shape (B, L, D)
            mask: Not used in this implementation

        Returns:
            A tuple (output, attention_weights) where:
                output: Tensor of shape (B, L, D)
                attention_weights: Dummy tensor (zeros) of shape (B, num_heads, L, L)
        """
        B, L, D = query.shape
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # Reshape for multiple heads
        q = self._reshape_for_heads(q)  # (B, H, L, head_dim)
        k = self._reshape_for_heads(k)  # (B, H, L, head_dim)
        v = self._reshape_for_heads(v)  # (B, H, L, head_dim)

        # Apply kernel feature map: f(x) = ELU(x) + 1
        q_feature = elu(q.data).data + 1  # (B, H, L, head_dim)
        k_feature = elu(k.data).data + 1  # (B, H, L, head_dim)

        # Compute denominator by summing over sequence length
        k_sum = np.sum(k_feature, axis=2)  # (B, H, head_dim)

        # Initialize output storage
        output = np.zeros((B, self.num_heads, L, self.head_dim))

        # Compute for each batch and head
        for b in range(B):
            for h in range(self.num_heads):
                q_slice = q_feature[b, h]  # (L, head_dim)
                k_slice = k_feature[b, h]  # (L, head_dim)
                v_slice = v.data[b, h]  # (L, head_dim)

                denom = k_sum[b, h] + 1e-6  # (head_dim,)
                kv_sum = np.sum(k_slice[:, None] * v_slice, axis=0)  # (head_dim,)
                output[b, h] = q_slice * (kv_sum / denom)

        # Reshape back without squeezing
        output = np.transpose(output, (0, 2, 1, 3)).reshape(B, L, D)
        output = self.w_o(Tensor(output))

        # Return dummy attention weights for compatibility
        attn_weights = np.zeros((B, self.num_heads, L, L))
        return output, Tensor(attn_weights)
