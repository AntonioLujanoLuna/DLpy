import numpy as np

from ....core import Module, Tensor
from ...attention.utils import get_angles
from ...base.dropout import Dropout


class PositionalEncoding(Module):
    """
    Positional Encoding module.

    Adds positional information to the input embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model (int): Dimension of the model
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)

        # Create position array and dimension indices
        position = np.arange(max_len)
        div_term_indices = np.arange(0, d_model, 2)

        # Calculate angles using the more stable method
        angles = get_angles(position, div_term_indices, d_model)

        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)

        # Register buffer (not a parameter)
        self.register_buffer("pe", Tensor(pe[np.newaxis, :, :]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.shape[1]]
        return Tensor(self.dropout(x))


# TODO Other embedding types (learned positional, rotary, ALiBi)
