from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core import Module, Tensor
from ..nn.activations import ReLU
from ..nn.dropout import Dropout
from ..nn.layer_norm import LayerNorm
from ..nn.linear import Linear
from ..nn.sequential import Sequential
from ..utils import calculate_fan_in_fan_out

class TransformerEncoderLayer(Module):
    """
    Transformer Encoder Layer.

    Implements a single layer of the transformer encoder, consisting of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm

    Args:
        d_model (int): The dimension of the model
        nhead (int): Number of attention heads
        dim_feedforward (int): Dimension of feedforward network
        dropout (float): Dropout probability
        activation (str): Activation function to use
        layer_norm_eps (float): eps value in layer normalizations
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead. Got {d_model} and {nhead}"
            )
        if dim_feedforward < d_model:
            raise ValueError(
                f"dim_feedforward ({dim_feedforward}) < d_model ({d_model})"
            )

        super().__init__()
        # Create single instances of dropout layers
        self.attn_dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=0.0
        )  # Set to 0 as we handle dropout separately
        self.activation = ReLU()

        # Feed forward network with dropout
        self.ff = Sequential(
            Linear(d_model, dim_feedforward),
            self.activation,
            Dropout(
                dropout
            ),  # This is fine as Sequential handles the instance properly
            Linear(dim_feedforward, d_model),
        )

        self.norm1 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = LayerNorm([d_model], eps=layer_norm_eps)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Self attention block
        attn_output, _ = self.self_attn(x, x, x, mask)
        attn_output = self.attn_dropout(
            attn_output
        )  # Apply dropout to attention output
        x = x + self.dropout1(attn_output)  # Apply dropout to residual
        x = self.norm1(x)
        # Feedforward block
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)  # Apply dropout to residual
        x = self.norm2(x)

        return x

class TransformerEncoder(Module):
    """
    Transformer Encoder.

    A stack of N encoder layers.

    Args:
        encoder_layer: An instance of TransformerEncoderLayer
        num_layers (int): Number of encoder layers in the stack
        norm (Module, optional): Layer normalization component
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(*[encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer encoder.

        Args:
            src: Source sequence of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        output = src

        for layer in self.layers:
            output = layer(output, mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

