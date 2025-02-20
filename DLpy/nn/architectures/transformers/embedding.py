import numpy as np
from numpy.typing import NDArray

from ....core import Module, Tensor
from ...attention.utils import get_angles
from ...base.dropout import Dropout


class Embedding(Module):
    """
    Simple embedding layer mapping token indices to embedding vectors.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the embeddings.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Random initialization for demonstration purposes.
        self.weight = Tensor(np.random.randn(vocab_size, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps token indices in x to embeddings.

        Args:
            x (Tensor): Tensor of shape (batch_size, seq_len) with token indices.

        Returns:
            Tensor: Embedding tensor of shape (batch_size, seq_len, d_model).
        """
        indices = x.data.astype(np.int64)
        return Tensor(self.weight.data[indices])


class PositionalEncoding(Module):
    """
    Positional Encoding module.

    Adds positional information to the input embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)

        # Create position array and dimension indices.
        position = np.arange(max_len)
        div_term_indices = np.arange(0, d_model, 2)

        # Calculate angles using the more stable method.
        angles = get_angles(position, div_term_indices, d_model)

        # Create positional encoding matrix.
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)

        # Register buffer (non-trainable).
        self.register_buffer("pe", Tensor(pe[np.newaxis, :, :]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.shape[1]]
        return Tensor(self.dropout(x))


class LearnedPositionalEmbedding(Module):
    """
    Learned positional embedding module.

    Instead of using fixed sinusoidal functions, this module learns a positional embedding
    as a parameter.

    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)
        # Initialize learned positional embeddings with small random values.
        pe = np.random.randn(max_len, d_model) * 0.02
        # Store as a parameter with shape (1, max_len, d_model).
        self.register_parameter("pe", Tensor(pe[np.newaxis, :, :]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor with learned positional embeddings added.
        """
        seq_len = x.shape[1]
        pos_embedding = self.pe[:, :seq_len, :]
        x = x + pos_embedding
        return Tensor(self.dropout(x))


class RotaryPositionalEmbedding(Module):
    """
    Rotary positional embedding module.

    Implements rotary embeddings (RoPE) which apply a rotation to the token embeddings.
    Typically used for query/key projections in multi-head attention.

    Args:
        d_model (int): Dimension of the model (should be even).
        max_len (int): Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for rotary embeddings. Got {d_model}."
            )
        self.d_model = d_model
        self.max_len = max_len
        # Precompute angles.
        position = np.arange(max_len)[:, None]  # (max_len, 1)
        dim = np.arange(d_model)[None, :]  # (1, d_model)
        angle_rates = 1.0 / np.power(10000, (2 * (dim // 2)) / d_model)
        angles = position * angle_rates  # (max_len, d_model)
        # Precompute cosine and sine matrices.
        cos = np.cos(angles)
        sin = np.sin(angles)
        # Register as buffers (non-trainable).
        self.register_buffer(
            "cos", Tensor(cos[np.newaxis, :, :])
        )  # shape (1, max_len, d_model)
        self.register_buffer(
            "sin", Tensor(sin[np.newaxis, :, :])
        )  # shape (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the rotary transformation to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of the same shape with rotary positional embeddings applied.
        """
        seq_len = x.shape[1]
        cos = self.cos[:, :seq_len, :]
        sin = self.sin[:, :seq_len, :]

        # Helper function to rotate half the dimensions.
        def rotate_half(x_array: NDArray) -> NDArray:
            x1 = x_array[..., : self.d_model // 2]
            x2 = x_array[..., self.d_model // 2 :]
            return np.concatenate([-x2, x1], axis=-1)

        # Apply rotary transformation: x * cos + rotate(x) * sin.
        x_rotated = x.data * cos.data + rotate_half(x.data) * sin.data
        return Tensor(x_rotated)


class ALiBiEmbedding(Module):
    """
    ALiBi (Attention with Linear Biases) module.

    Instead of adding positional embeddings to token embeddings, ALiBi adds a bias term
    directly to the attention scores. This bias is computed linearly based on the distance
    between tokens.

    Args:
        num_heads (int): Number of attention heads.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # Compute slopes for each head following a simple exponential decay.
        slopes = np.array([1.0 / (2 ** (i / num_heads)) for i in range(num_heads)])
        # Register slopes as a buffer (non-trainable).
        self.register_buffer("slopes", Tensor(slopes))

    def forward(self, seq_len: int) -> Tensor:
        """
        Computes the ALiBi bias matrix for a given sequence length.

        Args:
            seq_len (int): Sequence length.

        Returns:
            Tensor: Bias matrix of shape (1, num_heads, seq_len, seq_len) to be added to attention scores.
        """
        pos = np.arange(seq_len)
        # Create a distance matrix where each element [i, j] = max(j - i, 0).
        distance = pos[None, :] - pos[:, None]
        distance = np.maximum(distance, 0)
        slopes = self.slopes.data  # shape (num_heads,)
        # Compute bias for each head: bias = -slope * distance.
        bias = -np.einsum(
            "h,ij->hij", slopes, distance
        )  # shape (num_heads, seq_len, seq_len)
        bias = bias[
            np.newaxis, ...
        ]  # Add batch dimension: (1, num_heads, seq_len, seq_len)
        return Tensor(bias)
