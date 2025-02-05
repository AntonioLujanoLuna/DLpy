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

class TransformerDecoderLayer(Module):
    """
    Transformer Decoder Layer.

    Implements a single layer of the transformer decoder, consisting of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head attention over encoder output
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm

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
        super().__init__()

        # Self attention mechanism
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        # Cross attention mechanism
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)

        # Create ReLU activation
        self.activation = ReLU()

        # Initialize all required layers
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Layer normalization layers
        self.norm1 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm3 = LayerNorm([d_model], eps=layer_norm_eps)

        # Dropout layers
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Feed-forward network
        self.ff = Sequential(self.linear1, self.activation, self.dropout, self.linear2)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of decoder layer.

        Args:
            x: Input tensor (target sequence)
            memory: Output from encoder
            tgt_mask: Mask for target sequence
            memory_mask: Mask for source sequence

        Returns:
            Output tensor
        """
        # Self-attention block
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention block
        attn_output, _ = self.multihead_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # Feedforward block
        ff_output = self.ff(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x

class TransformerDecoder(Module):
    """
    Transformer Decoder.

    A stack of N decoder layers with masking functionality.

    Args:
        decoder_layer: An instance of TransformerDecoderLayer
        num_layers (int): Number of decoder layers
        norm (Module, optional): Layer normalization component
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(*[decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of transformer decoder.

        Args:
            tgt: Target sequence
            memory: Output from encoder
            tgt_mask: Target sequence mask
            memory_mask: Source sequence mask

        Returns:
            Output tensor
        """
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
